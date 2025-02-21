from flask import Flask, render_template, request, jsonify, send_from_directory, Response, abort
import os
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from typing import Dict, List, Any, Optional, Tuple
import traceback
import shutil
import tempfile
from PIL import Image
from roboflow import Roboflow
from dds_cloudapi_sdk import Config, Client, TextPrompt
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
import cv2

# -----------------------------
# Constants
# -----------------------------
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
DEFAULT_PAGE_SIZE = 20
DATABASE_TIMEOUT = 30.0  # SQLite timeout in seconds

# -----------------------------
# Model Configuration
# -----------------------------
# Roboflow setup
rf_api_key = os.getenv("ROBOFLOW_API_KEY", "Otg64Ra6wNOgDyjuhMYU")
workspace = os.getenv("ROBOFLOW_WORKSPACE", "alat-pelindung-diri")
project_name = os.getenv("ROBOFLOW_PROJECT", "nescafe-4base")
model_version = int(os.getenv("ROBOFLOW_MODEL_VERSION", "66"))

rf = Roboflow(api_key=rf_api_key)
project = rf.workspace(workspace).project(project_name)
yolo_model = project.version(model_version).model

# DINO-X setup
DINOX_API_KEY = os.getenv("DINO_X_API_KEY", "af1a81e6517f5cafb12752f7de7f6be9")
DINOX_PROMPT = "beverage . bottle . cans . boxed milk . milk"

dinox_config = Config(DINOX_API_KEY)
dinox_client = Client(dinox_config)

# -----------------------------
# Logging Configuration
# -----------------------------
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("server.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

# -----------------------------
# Flask Application Setup
# -----------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'nestle-iot-monitoring-secret-key'),
        UPLOAD_FOLDER=os.path.join(os.getcwd(), 'uploads'),
        DATABASE=os.path.join(os.getcwd(), 'nestle_iot.db'),
        MAX_CONTENT_LENGTH=MAX_FILE_SIZE
    )
    
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Ensure frames folder exists
    os.makedirs(os.path.join(os.getcwd(), 'frames'), exist_ok=True)
    
    return app

app = create_app()
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# Database Setup
# -----------------------------
def get_db_connection() -> sqlite3.Connection:
    """Create a database connection with proper configuration and error handling."""
    try:
        conn = sqlite3.connect(
            app.config['DATABASE'],
            timeout=DATABASE_TIMEOUT,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

def init_db() -> None:
    """Initialize the database with proper error handling."""
    try:
        with get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS detection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    roboflow_outputs TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, device_id: str) -> str:
    """Save an uploaded file with proper error handling."""
    if not file or not allowed_file(file.filename):
        raise ValueError("Invalid file type")
    
    filename = secure_filename(file.filename)
    device_folder = os.path.join(app.config['UPLOAD_FOLDER'], device_id)
    os.makedirs(device_folder, exist_ok=True)
    
    file_path = os.path.join(device_folder, filename)
    file.save(file_path)
    return file_path

def parse_detection_data(roboflow_data: Dict) -> Tuple[int, int]:
    """Parse detection data and return counts for Nestle and unclassified items."""
    nestle_count = 0
    unclassified_count = 0
    
    try:
        if isinstance(roboflow_data, dict) and 'predictions' in roboflow_data:
            for pred in roboflow_data['predictions']:
                if pred.get('class', '').lower().startswith('nestle'):
                    nestle_count += 1
                else:
                    unclassified_count += 1
        elif isinstance(roboflow_data, list):
            for item in roboflow_data:
                if 'roboflow_predictions' in item:
                    for pred in item['roboflow_predictions']:
                        if pred.get('class', '').lower().startswith('nestle'):
                            nestle_count += 1
                        else:
                            unclassified_count += 1
                if 'dinox_predictions' in item:
                    unclassified_count += len(item['dinox_predictions'])
    except Exception as e:
        logger.error(f"Error parsing detection data: {e}")
        raise
        
    return nestle_count, unclassified_count

def get_default_sku_data() -> Dict:
    """Generate default SKU data structure."""
    current_date = datetime.now()
    dates = [(current_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    
    return {
        "daily_data": {
            "dates": dates,
            "values": [0] * 7
        },
        "nestle": {
            "max": {"count": 0, "date": dates[-1]},
            "min": {"count": 0, "date": dates[0]},
            "avg": 0
        },
        "competitor": {
            "max": {"count": 0, "date": dates[-1]},
            "min": {"count": 0, "date": dates[0]},
            "avg": 0
        },
        "market_share": {
            "labels": ["Nestle", "Competitor"],
            "values": [50, 50]
        },
        "top_products": [
            {"name": "Product A", "count": 0},
            {"name": "Product B", "count": 0},
            {"name": "Product C", "count": 0}
        ],
        "daily_count": {
            "product": "All Products",
            "dates": dates,
            "counts": [0] * 7
        }
    }

def copy_file_to_frames(source_path, target_filename):
    """Copy a file to the frames directory with a specific filename."""
    frames_dir = os.path.join(os.getcwd(), 'frames')
    target_path = os.path.join(frames_dir, target_filename)
    
    try:
        shutil.copy2(source_path, target_path)
        logger.info(f"Copied {source_path} to {target_path}")
        return target_path
    except Exception as e:
        logger.error(f"Error copying file {source_path} to {target_path}: {e}")
        return None

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    """Render the main dashboard page with default data."""
    try:
        sku_data = get_default_sku_data()
        return render_template('index.html', sku_data=sku_data)
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/receive_data', methods=['POST'])
def receive_data():
    """Handle incoming detection data with comprehensive error handling."""
    try:
        # Validate required fields
        required_fields = ['device_id', 'timestamp', 'roboflow_outputs']
        if not all(field in request.form for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        device_id = request.form['device_id']
        timestamp = request.form['timestamp']
        roboflow_outputs = request.form['roboflow_outputs']

        # Validate JSON data
        try:
            roboflow_data = json.loads(roboflow_outputs)
            logger.info(f"Received detection data: {roboflow_data}")
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON in roboflow_outputs'}), 400

        # Handle file uploads
        if not request.files:
            return jsonify({'error': 'No image files uploaded'}), 400

        saved_images = []
        frames_dir = os.path.join(os.getcwd(), 'frames')
        
        # Extract labeled image paths from detection data
        labeled_image_paths = []
        if isinstance(roboflow_data, list):
            for item in roboflow_data:
                if isinstance(item, dict) and 'labeled_image' in item:
                    labeled_image_paths.append(item['labeled_image'])
        
        logger.info(f"Found labeled image paths: {labeled_image_paths}")
        
        # Save uploaded files
        for key, file in request.files.items():
            try:
                if file and file.filename:
                    # Save to uploads folder
                    file_path = save_uploaded_file(file, device_id)
                    saved_images.append(file_path)
                    logger.info(f"Saved image: {file_path}")
                    
                    # For the first file, also save to frames directory
                    if key == 'image0' and labeled_image_paths:
                        # This means we received a detection that has labeled images
                        # First save the original frame
                        frame_index = 0
                        frame_filename = f"frame_{frame_index:03d}.jpg"
                        file.seek(0)  # Reset file pointer
                        frame_path = os.path.join(frames_dir, frame_filename)
                        with open(frame_path, 'wb') as f:
                            f.write(file.read())
                        logger.info(f"Saved frame to {frame_path}")
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {e}")
                continue

        if not saved_images:
            return jsonify({'error': 'Failed to save any images'}), 400

        # Store event in database
        try:
            with get_db_connection() as conn:
                cursor = conn.execute(
                    '''INSERT INTO detection_events 
                       (device_id, timestamp, roboflow_outputs, image_path, created_at) 
                       VALUES (?, ?, ?, ?, ?)''',
                    (device_id, timestamp, roboflow_outputs, 
                     json.dumps(saved_images), datetime.now().isoformat())
                )
                event_id = cursor.lastrowid
                conn.commit()

            # Emit socket event
            event_data = {
                'id': event_id,
                'device_id': device_id,
                'timestamp': timestamp,
                'image_count': len(saved_images)
            }
            socketio.emit('new_detection', event_data)

            return jsonify({
                'success': True,
                'message': f'Successfully received {len(saved_images)} images from device {device_id}',
                'event_id': event_id
            })

        except sqlite3.Error as e:
            logger.error(f"Database error in receive_data: {e}")
            return jsonify({'error': 'Database error'}), 500

    except Exception as e:
        logger.error(f"Error in receive_data: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/images/<path:filename>')
def get_image(filename):
    """Serve images with proper error handling and security checks."""
    try:
        # Validate filename
        if '..' in filename or filename.startswith('/'):
            abort(404)
        
        # Handle both direct uploads and frames from Raspberry Pi
        if filename.startswith('frames/'):
            # For images in the frames directory
            frames_dir = os.path.join(os.getcwd(), 'frames')
            file_path = filename.replace('frames/', '')
            return send_from_directory(frames_dir, file_path)
        elif os.path.exists(os.path.join(os.getcwd(), 'frames', filename)):
            # For direct frame references
            frames_dir = os.path.join(os.getcwd(), 'frames')
            return send_from_directory(frames_dir, filename)
        else:
            # For images in the normal upload directory
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
            
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        abort(404)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/events', methods=['GET'])
def get_events():
    """Get events with pagination and filtering, processing counts for frontend display."""
    try:
        device_id = request.args.get('device_id')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        page = max(1, request.args.get('page', 1, type=int))
        limit = min(50, request.args.get('limit', DEFAULT_PAGE_SIZE, type=int))

        query = 'SELECT * FROM detection_events WHERE 1=1'
        params = []

        # Build query with parameters
        if device_id:
            query += ' AND device_id = ?'
            params.append(device_id)
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)

        # Add pagination
        query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
        params.extend([limit, (page - 1) * limit])

        with get_db_connection() as conn:
            events = conn.execute(query, params).fetchall()
            
            # Get total count
            count_query = query.split('ORDER BY')[0].replace('SELECT *', 'SELECT COUNT(*)')
            total = conn.execute(count_query, params[:-2]).fetchone()[0]

        # Process results
        results = []
        for event in events:
            try:
                # Handle empty or invalid JSON data
                roboflow_outputs = event['roboflow_outputs']
                if not roboflow_outputs:
                    continue

                try:
                    roboflow_data = json.loads(roboflow_outputs)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in event {event['id']}: {roboflow_outputs}")
                    continue

                # Extract counts based on the structure of roboflow_data
                nestle_count = 0
                competitor_count = 0

                if isinstance(roboflow_data, dict):
                    # Handle case where roboflow_data is a dictionary
                    if 'roboflow_predictions' in roboflow_data:
                        nestle_count = len(roboflow_data['roboflow_predictions'])
                    if 'dinox_predictions' in roboflow_data:
                        competitor_count = len(roboflow_data['dinox_predictions'])
                elif isinstance(roboflow_data, dict) and 'predictions' in roboflow_data:
                    # Handle case where it's direct Roboflow output
                    nestle_count = len(roboflow_data['predictions'])
                
                results.append({
                    'id': event['id'],
                    'device_id': event['device_id'],
                    'timestamp': event['timestamp'],
                    'nestle_detections': nestle_count,
                    'unclassified_detections': competitor_count,
                    'created_at': event['created_at']
                })
            except Exception as e:
                logger.error(f"Error processing event {event['id']}: {str(e)}")
                continue

        return jsonify({
            'data': results,
            'pagination': {
                'total': total,
                'page': page,
                'limit': limit,
                'pages': (total + limit - 1) // limit
            }
        })

    except Exception as e:
        logger.error(f"Error in get_events: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/events/<int:event_id>')
def get_event_details(event_id):
    try:
        with get_db_connection() as conn:
            event = conn.execute('SELECT * FROM detection_events WHERE id = ?', (event_id,)).fetchone()
            
            if not event:
                return jsonify({'error': 'Event not found'}), 404

            # Parse roboflow outputs
            roboflow_data = json.loads(event['roboflow_outputs']) if event['roboflow_outputs'] else {}
            
            # Extract product counts
            nestle_products = {}
            competitor_products = {}
            
            if isinstance(roboflow_data, dict):
                if 'roboflow_predictions' in roboflow_data:
                    nestle_products = roboflow_data['roboflow_predictions']
                if 'dinox_predictions' in roboflow_data:
                    competitor_products = roboflow_data['dinox_predictions']

            # Get total counts
            nestle_count = sum(nestle_products.values() if isinstance(nestle_products, dict) else [1 for _ in nestle_products])
            comp_count = sum(competitor_products.values() if isinstance(competitor_products, dict) else [1 for _ in competitor_products])

            # Fix image path
            image_path = event['image_path']
            if image_path:
                # Remove absolute path if exists
                image_path = os.path.basename(image_path)
                # Ensure path starts with 'uploads/'
                if not image_path.startswith('uploads/'):
                    image_path = f'uploads/{image_path}'

            # Format response
            response = {
                'id': event['id'],
                'device_id': event['device_id'],
                'timestamp': event['timestamp'],
                'nestleCount': nestle_count,
                'compCount': comp_count,
                'products': {
                    'nestle_products': nestle_products,
                    'competitor_products': competitor_products
                },
                'image_path': image_path,
                'created_at': event['created_at']
            }
            
            return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error getting event details: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/dashboard_data')
def get_dashboard_data():
    try:
        current_date = datetime.now()
        dates = [(current_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
        
        # Initialize data structure
        dashboard_data = {
            "daily_data": {
                "dates": dates,
                "nestle_values": [0] * 7,
                "competitor_values": [0] * 7
            },
            "nestle": {
                "max": {"count": 0, "date": dates[-1]},
                "min": {"count": float('inf'), "date": dates[0]},  # Initialize with infinity
                "avg": {"count": 0, "period": "Last 7 days"}
            },
            "competitor": {
                "max": {"count": 0, "date": dates[-1]},
                "min": {"count": float('inf'), "date": dates[0]},
                "avg": {"count": 0, "period": "Last 7 days"}
            },
            "top_products": [
                {'name': 'No Product', 'count': 0},
                {'name': 'No Product', 'count': 0},
                {'name': 'No Product', 'count': 0}
            ]
        }

        # Tambahkan perhitungan untuk top 3 Nestle products
        product_counts = {}
        
        with get_db_connection() as conn:
            events = conn.execute(
                """SELECT * FROM detection_events 
                   WHERE timestamp >= ? 
                   ORDER BY timestamp""",
                (dates[0],)
            ).fetchall()
            
            # Process events to count products
            for event in events:
                try:
                    if not event['roboflow_outputs']:
                        continue

                    roboflow_data = json.loads(event['roboflow_outputs'])
                    
                    if isinstance(roboflow_data, dict) and 'roboflow_predictions' in roboflow_data:
                        nestle_products = roboflow_data['roboflow_predictions']
                        
                        # Count each product
                        if isinstance(nestle_products, dict):
                            for product, count in nestle_products.items():
                                if product not in product_counts:
                                    product_counts[product] = 0
                                product_counts[product] += count

                except Exception as e:
                    logger.error(f"Error processing event for top products: {str(e)}")
                continue
                    
        # Get top 3 products
        top_products = []
        if product_counts:
            sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_products[:3]
            
            # Fill remaining slots if less than 3 products
            while len(top_3) < 3:
                top_3.append(('No Product', 0))
            
            top_products = [
                {'name': product, 'count': count}
                for product, count in top_3
            ]
        else:
            # If no products found, fill with empty data
            top_products = [
                {'name': 'No Product', 'count': 0},
                {'name': 'No Product', 'count': 0},
                {'name': 'No Product', 'count': 0}
            ]

        dashboard_data = {
            "daily_data": {
                "dates": dates,
                "nestle_values": [0] * 7,
                "competitor_values": [0] * 7
            },
            "nestle": {
                "max": {"count": 0, "date": dates[-1]},
                "min": {"count": float('inf'), "date": dates[0]},  # Initialize with infinity
                "avg": {"count": 0, "period": "Last 7 days"}
            },
            "competitor": {
                "max": {"count": 0, "date": dates[-1]},
                "min": {"count": float('inf'), "date": dates[0]},
                "avg": {"count": 0, "period": "Last 7 days"}
            },
            "top_products": top_products
        }

        with get_db_connection() as conn:
            # Query events for the last 7 days
            events = conn.execute(
                """SELECT * FROM detection_events 
                   WHERE timestamp >= ? 
                   ORDER BY timestamp""",
                (dates[0],)
            ).fetchall()

            nestle_total = 0
            competitor_total = 0
            nestle_count = 0
            competitor_count = 0
            
            # Track all non-zero counts for proper minimum calculation
            nestle_counts = []
            competitor_counts = []

            # Process each event
            for event in events:
                try:
                    if not event['roboflow_outputs']:
                        continue
            
                    roboflow_data = json.loads(event['roboflow_outputs'])
                    event_date = datetime.fromisoformat(event['timestamp'].split('T')[0]).strftime("%Y-%m-%d")
                    
                    # Count items in this detection
                    current_nestle_count = 0
                    current_competitor_count = 0

                    if isinstance(roboflow_data, dict):
                        # Count Nestle products
                        if 'roboflow_predictions' in roboflow_data:
                            nestle_products = roboflow_data['roboflow_predictions']
                            current_nestle_count = sum(nestle_products.values() if isinstance(nestle_products, dict) 
                                                     else [1 for _ in nestle_products])
                            
                            if current_nestle_count > 0:
                                nestle_counts.append({
                                    'count': current_nestle_count,
                                    'date': event_date
                                })
                            
                            # Update daily total if date is in range
                            if event_date in dates:
                                date_index = dates.index(event_date)
                                dashboard_data['daily_data']['nestle_values'][date_index] += current_nestle_count

                        # Count Competitor products
                        if 'dinox_predictions' in roboflow_data:
                            competitor_products = roboflow_data['dinox_predictions']
                            current_competitor_count = sum(competitor_products.values() if isinstance(competitor_products, dict) 
                                                        else [1 for _ in competitor_products])
                            
                            if current_competitor_count > 0:
                                competitor_counts.append({
                                    'count': current_competitor_count,
                                    'date': event_date
                                })
                            
                            if event_date in dates:
                                date_index = dates.index(event_date)
                                dashboard_data['daily_data']['competitor_values'][date_index] += current_competitor_count

                    # Update maximum counts
                    if current_nestle_count > dashboard_data['nestle']['max']['count']:
                        dashboard_data['nestle']['max']['count'] = current_nestle_count
                        dashboard_data['nestle']['max']['date'] = event_date

                    if current_competitor_count > dashboard_data['competitor']['max']['count']:
                        dashboard_data['competitor']['max']['count'] = current_competitor_count
                        dashboard_data['competitor']['max']['date'] = event_date

                    # Add to totals for average calculation
                    if current_nestle_count > 0:
                        nestle_total += current_nestle_count
                        nestle_count += 1
                    if current_competitor_count > 0:
                        competitor_total += current_competitor_count
                        competitor_count += 1

                except Exception as e:
                    logger.error(f"Error processing event {event['id']} for dashboard: {str(e)}")
                    continue

            # Calculate minimum counts from collected non-zero counts
            if nestle_counts:
                min_nestle = min(nestle_counts, key=lambda x: x['count'])
                dashboard_data['nestle']['min']['count'] = min_nestle['count']
                dashboard_data['nestle']['min']['date'] = min_nestle['date']
            else:
                dashboard_data['nestle']['min']['count'] = 0
                
            if competitor_counts:
                min_competitor = min(competitor_counts, key=lambda x: x['count'])
                dashboard_data['competitor']['min']['count'] = min_competitor['count']
                dashboard_data['competitor']['min']['date'] = min_competitor['date']
            else:
                dashboard_data['competitor']['min']['count'] = 0

            # Calculate averages
            dashboard_data['nestle']['avg']['count'] = round(nestle_total / nestle_count if nestle_count > 0 else 0, 1)
            dashboard_data['competitor']['avg']['count'] = round(competitor_total / competitor_count if competitor_count > 0 else 0, 1)
        
        return jsonify(dashboard_data)
    
    except Exception as e:
        logger.error(f"Error generating dashboard data: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/export/csv')
def export_csv():
    """Export detection data as CSV."""
    try:
        with get_db_connection() as conn:
            events = conn.execute('SELECT * FROM detection_events ORDER BY timestamp DESC').fetchall()
        
        if not events:
            return Response("No data available for export", mimetype='text/plain')
        
        # Generate CSV content
        csv_data = "id,device_id,timestamp,nestle_detections,unclassified_detections,created_at\n"
        
        for event in events:
            try:
                roboflow_data = json.loads(event['roboflow_outputs'])
                nestle_count, unclassified_count = parse_detection_data(roboflow_data)
                
                csv_data += f"{event['id']},{event['device_id']},{event['timestamp']},{nestle_count},{unclassified_count},{event['created_at']}\n"
            except Exception as e:
                logger.error(f"Error processing event {event['id']} for CSV: {e}")
                continue
        
        # Create response with CSV file
        response = Response(
            csv_data,
            mimetype='text/csv',
            headers={"Content-Disposition": "attachment;filename=nestle_detection_data.csv"}
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        return jsonify({'error': 'Error generating CSV export'}), 500

@app.route('/api/devices')
def get_devices():
    """Get list of unique devices that have sent data."""
    try:
        with get_db_connection() as conn:
            devices = conn.execute(
                """SELECT DISTINCT device_id FROM detection_events"""
            ).fetchall()
            
        device_list = [{"id": row["device_id"], "name": row["device_id"]} for row in devices]
        
        return jsonify(device_list)
        
    except Exception as e:
        logger.error(f"Error getting devices: {str(e)}")
        return jsonify({'error': 'Error retrieving device list'}), 500

@app.route('/check_image', methods=['POST'])
def check_image():
    """Handle image upload and detection check."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name

            try:
                # Process image with both detectors
                # 1. YOLO Detection (Nestlé products)
                yolo_pred = yolo_model.predict(temp_path, confidence=50, overlap=80).json()
                
                # Process Nestle products (from Roboflow)
                nestle_products = {}
                nestle_boxes = []
                for pred in yolo_pred['predictions']:
                    class_name = pred['class']
                    nestle_products[class_name] = nestle_products.get(class_name, 0) + 1
                    nestle_boxes.append({
                        'x': pred['x'],
                        'y': pred['y'],
                        'width': pred['width'],
                        'height': pred['height'],
                        'class': class_name,
                        'confidence': pred['confidence']
                    })
                total_nestle = sum(nestle_products.values())

                # 2. DINO-X Detection (Competitor products)
                image_url = dinox_client.upload_file(temp_path)
                task = DinoxTask(
                    image_url=image_url,
                    prompts=[TextPrompt(text=DINOX_PROMPT)],
                    bbox_threshold=0.25,
                    targets=[DetectionTarget.BBox]
                )
                dinox_client.run_task(task)
                dinox_pred = task.result.objects

                # Process competitor products (from DINO-X)
                competitor_products = {}
                competitor_boxes = []
                
                for obj in dinox_pred:
                    category = obj.category.strip().lower()
                    # Only count as competitor if not overlapping with Nestle products
                    box = obj.bbox
                    is_overlapping = False
                    for nestle_box in nestle_boxes:
                        if is_overlap((box[0], box[1], box[2], box[3]), 
                                    [(nestle_box['x'], nestle_box['y'], 
                                      nestle_box['width'], nestle_box['height'])]):
                            is_overlapping = True
                            break
                    
                    if not is_overlapping:
                        competitor_products[category] = competitor_products.get(category, 0) + 1
                        competitor_boxes.append({
                            'box': box,
                            'class': category,
                            'confidence': obj.score
                        })

                total_competitor = sum(competitor_products.values())

                # Draw detections on image
                img = cv2.imread(temp_path)
                
                # Draw Nestlé products (blue)
                for box in nestle_boxes:
                    x, y, w, h = box['x'], box['y'], box['width'], box['height']
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f"Nestle: {box['class']}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.55, (255, 0, 0), 2)

                # Draw competitor products (red)
                for box in competitor_boxes:
                    x1, y1, x2, y2 = box['box']
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 0, 255), 2)
                    cv2.putText(img, f"Competitor: {box['class']}", 
                               (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.55, (0, 0, 255), 2)

                # Save labeled image
                labeled_filename = 'labeled_' + secure_filename(file.filename)
                labeled_path = os.path.join(app.config['UPLOAD_FOLDER'], labeled_filename)
                cv2.imwrite(labeled_path, img)

                # After processing the image and getting results, save to database
                with get_db_connection() as conn:
                    current_time = datetime.now().isoformat()
                    current_date = current_time.split('T')[0]
                    
                    detection_data = {
                        'roboflow_predictions': nestle_products,
                        'dinox_predictions': competitor_products,
                        'counts': {
                            'nestle': total_nestle,
                            'competitor': total_competitor,
                            'date': current_date
                        }
                    }
                    
                    cursor = conn.execute(
                        '''INSERT INTO detection_events 
                           (device_id, timestamp, roboflow_outputs, image_path, created_at) 
                           VALUES (?, ?, ?, ?, ?)''',
                        ('web_upload', current_time, json.dumps(detection_data), 
                         labeled_path, current_time)
                    )
                    event_id = cursor.lastrowid
                    conn.commit()

                # Emit socket event with complete data for real-time updates
                socketio.emit('new_detection', {
                    'id': event_id,
                    'device_id': 'web_upload',
                    'timestamp': current_time,
                    'date': current_date,
                    'nestle_count': total_nestle,
                    'competitor_count': total_competitor,
                    'type': 'new_detection'
                })

                # Return the result with additional data for graph update
                result = {
                    'nestle_products': nestle_products,
                    'competitor_products': competitor_products,
                    'total_nestle': total_nestle,
                    'total_competitor': total_competitor,
                    'labeled_image': f'uploads/{labeled_filename}',
                    'timestamp': current_time,
                    'date': current_date
                }

                return jsonify(result)

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        return jsonify({'error': str(e)}), 500

def is_overlap(box1, boxes2, threshold=0.3):
    """Check for overlap between boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    for b2 in boxes2:
        x2, y2, w2, h2 = b2
        x2_min = x2 - w2/2
        x2_max = x2 + w2/2
        y2_min = y2 - h2/2
        y2_max = y2 + h2/2

        dx = min(x1_max, x2_max) - max(x1_min, x2_min)
        dy = min(y1_max, y2_max) - max(y1_min, y2_min)
        if (dx >= 0) and (dy >= 0):
            area_overlap = dx * dy
            area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
            if area_box1 > 0 and (area_overlap / area_box1) > threshold:
                return True
    return False

@app.route('/api/all_products')
def get_all_products():
    """Get all Nestle products with their counts."""
    try:
        current_date = datetime.now()
        start_date = (current_date - timedelta(days=6)).strftime("%Y-%m-%d")
        
        product_counts = {}
        
        with get_db_connection() as conn:
            events = conn.execute(
                """SELECT * FROM detection_events 
                   WHERE timestamp >= ? 
                   ORDER BY timestamp""",
                (start_date,)
            ).fetchall()

            # Process all events to count products
            for event in events:
                try:
                    if not event['roboflow_outputs']:
                        continue

                    roboflow_data = json.loads(event['roboflow_outputs'])
                    
                    if isinstance(roboflow_data, dict) and 'roboflow_predictions' in roboflow_data:
                        nestle_products = roboflow_data['roboflow_predictions']
                        
                        # Count each product
                        if isinstance(nestle_products, dict):
                            for product, count in nestle_products.items():
                                if product not in product_counts:
                                    product_counts[product] = 0
                                product_counts[product] += count

                except Exception as e:
                    logger.error(f"Error processing event for all products: {str(e)}")
                    continue

        # Sort products by count
        sorted_products = [
            {'name': name, 'count': count}
            for name, count in sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        return jsonify(sorted_products)

    except Exception as e:
        logger.error(f"Error getting all products: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

# -----------------------------
# Main Application Entry
# -----------------------------
if __name__ == '__main__':
    try:
        init_db()
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Server startup error: {e}")