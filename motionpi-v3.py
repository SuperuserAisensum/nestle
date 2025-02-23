import cv2
import time
import json
import os
import gc  # Add garbage collection
from datetime import datetime
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import tempfile

# =========== Roboflow Setup ===========
# We'll initialize this only when needed to save memory
from roboflow import Roboflow

# Load environment variables
load_dotenv()

SERVER_URL = os.getenv('SERVER_URL', 'http://16.78.246.103:5000')
DEVICE_ID = os.getenv('DEVICE_ID', 'raspberry_pi_zero')

ROBOFLOW_API_KEY = "Otg64Ra6wNOgDyjuhMYU"
ROBOFLOW_WORKSPACE = "alat-pelindung-diri"
ROBOFLOW_PROJECT = "nescafe-4base"
ROBOFLOW_VERSION = 66

# Initialize models only when needed
rf = None
project = None
yolo_model = None

# =========== OWLv2 Setup ===========
OWLV2_API_KEY = "bjJkZXZrb2Y1cDMzMXh3OHdzbGl6OlFQOHVmS2JkZjBmQUs2bnF2OVJVdXFoNnc0ZW5kN1hH"
OWLV2_PROMPTS = ["bottle", "tetra pak", "cans", "carton drink"]

def is_overlap(box1, boxes2, threshold=0.3):
    """
    Check if box1 overlaps significantly with any box in boxes2
    box1: [x1, y1, x2, y2]
    boxes2: list of [x_center, y_center, width, height]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    for b2 in boxes2:
        x2, y2, w2, h2 = b2
        x2_min = x2 - w2 / 2
        x2_max = x2 + w2 / 2
        y2_min = y2 - h2 / 2
        y2_max = y2 + h2 / 2

        dx = min(x1_max, x2_max) - max(x1_min, x2_min)
        dy = min(y1_max, y2_max) - max(y1_min, y2_min)
        if (dx >= 0) and (dy >= 0):
            area_overlap = dx * dy
            area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
            if area_box1 > 0 and (area_overlap / area_box1) > threshold:
                return True
    return False

def initialize_models():
    """Initialize ML models only when needed to save memory"""
    global rf, project, yolo_model
    if rf is None:
        print("Initializing Roboflow model...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        yolo_model = project.version(ROBOFLOW_VERSION).model
        print("Roboflow model initialized")

def process_image(image_path):
    """Process a single image with both YOLO and OWLv2 detections"""
    try:
        # Force garbage collection before processing
        gc.collect()

        # 1) Read & fix orientation with PIL
        pil_img = Image.open(image_path).convert("RGB")

        # Resize to lower resolution for Pi Zero
        original_width, original_height = pil_img.size
        pil_img = pil_img.resize((640, 480), Image.LANCZOS)

        # 2) Save temporarily to JPEG file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            pil_img.save(temp_file, format="JPEG", quality=85)  # Lower quality to save memory
            temp_path = temp_file.name

        # Initialize models if needed
        initialize_models()

        # 3) YOLO Detection
        yolo_pred = yolo_model.predict(temp_path, confidence=50, overlap=80).json()

        # Free up memory after prediction
        gc.collect()

        nestle_class_count = {}
        nestle_boxes = []
        for pred in yolo_pred['predictions']:
            class_name = pred['class']
            nestle_class_count[class_name] = nestle_class_count.get(class_name, 0) + 1
            nestle_boxes.append((pred['x'], pred['y'], pred['width'], pred['height']))
        total_nestle = sum(nestle_class_count.values())

        # 4) OWLv2 Detection
        headers = {
            "Authorization": "Basic " + OWLV2_API_KEY,
        }
        data = {
            "prompts": OWLV2_PROMPTS,
            "model": "owlv2"
        }
        with open(temp_path, "rb") as f:
            files = {"image": f}
            response = requests.post(
                "https://api.landing.ai/v1/tools/text-to-object-detection",
                files=files,
                data=data,
                headers=headers
            )
        owlv2_result = response.json()

        # Free up memory
        gc.collect()

        owlv2_objects = owlv2_result['data'][0] if 'data' in owlv2_result else []

        competitor_class_count = {}
        competitor_boxes = []
        for obj in owlv2_objects:
            if 'bounding_box' in obj:
                bbox = obj['bounding_box']  # Format: [x1, y1, x2, y2]
                if not is_overlap(bbox, nestle_boxes):
                    class_name = obj.get('label', 'unknown').strip().lower()
                    competitor_class_count[class_name] = competitor_class_count.get(class_name, 0) + 1
                    competitor_boxes.append({
                        "class": "unclassified",
                        "box": bbox,
                        "confidence": obj.get("score", 0)
                    })

        total_competitor = sum(competitor_class_count.values())

        # 5) Draw bounding boxes with OpenCV
        cv_img = cv2.imread(temp_path)

        # Draw Nestle boxes (green)
        for pred in yolo_pred['predictions']:
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_img, f"{pred['class']} {pred['confidence']:.2f}",
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Competitor boxes (red)
        for comp in competitor_boxes:
            x1, y1, x2, y2 = comp['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(cv_img,
                       f"{comp['class']} {comp['confidence']:.2f}",
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Format data for server
        detection_data = {
            "nestle_products": nestle_class_count,
            "total_nestle": total_nestle,
            "total_competitor": total_competitor,
            "competitor_detections": competitor_boxes
        }

        print("\nDetection Summary:")
        print("Nestlé Products:")
        for class_name, count in nestle_class_count.items():
            print(f"  - {class_name}: {count}")
        print(f"Competitor Products (unclassified): {total_competitor}")

        return cv_img, detection_data

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(f"OWLv2 response: {owlv2_result if 'owlv2_result' in locals() else 'No response'}")
        return None, None

    finally:
        # Cleanup temporary file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        # Force garbage collection
        gc.collect()

def detect_motion_and_capture_frames(frames_folder, num_frames=5):
    """Capture frames using PiCamera2 - Alternative for Pi Zero"""
    try:
        from picamera2 import Picamera2

        # Initialize camera with lower resolution
        picam2 = Picamera2()
        config = picam2.create_still_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        # Increase warm-up time to prevent initialization errors
        time.sleep(3)  # Changed from 2 to 3 seconds
        print("Camera initialized at 640x480 resolution")

        # Capture frames
        captured_frames = []
        for i in range(num_frames):
            frame_filename = os.path.join(frames_folder, f"frame_{i:03d}.jpg")
            picam2.capture_file(frame_filename)
            captured_frames.append(frame_filename)
            print(f"Saved frame to {frame_filename}")
            time.sleep(1)

        picam2.close()  # Changed from stop() to close()
        return captured_frames

    except ImportError:
        # Fall back to OpenCV method if picamera2 is not available
        print("picamera2 not available, falling back to OpenCV")
        return detect_motion_and_capture_video(frames_folder, num_frames)
    except Exception as e:
        print(f"Error in camera capture: {e}")
        if 'picam2' in locals():
            try:
                picam2.stop()
            except:
                pass
        return []

def detect_motion_and_capture_video(frames_folder, num_frames=5, video_duration=5,
                                    motion_threshold=500000, max_wait=10):
    """Fallback: Capture frames when motion is detected using OpenCV (original function)"""
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return []

        # Set resolution to something lower for Pi Zero
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Webcam started for motion detection.")
        time.sleep(2)  # Give camera time to warm up

        # Read first frame for baseline
        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to grab first frame")
            cap.release()
            return []

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_found = False
        start_time = time.time()

        while time.time() - start_time < max_wait:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(prev_gray, gray)
            diff_sum = np.sum(frame_diff)

            if diff_sum > motion_threshold:
                motion_found = True
                print("Motion detected!")
                break
            prev_gray = gray
            time.sleep(0.1)

        if not motion_found:
            print("No significant motion detected within the wait time.")
            # Just capture frames anyway for Pi Zero
            motion_found = True

        captured_frames = []
        if motion_found:
            interval = video_duration / num_frames
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to grab frame {i}")
                    continue

                frame_filename = os.path.join(frames_folder, f"frame_{i:03d}.jpg")
                cv2.imwrite(frame_filename, frame)
                captured_frames.append(frame_filename)
                print(f"Saved frame to {frame_filename}")
                time.sleep(interval)

        cap.release()
        return captured_frames

    except Exception as e:
        print(f"Error in motion detection/video capture: {e}")
        return []

    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("Camera cleaned up successfully")

def send_to_server(detection_data, image_file, server_url=SERVER_URL):
    """Send detection results and image to server"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_id = DEVICE_ID

    try:
        # Format data exactly as server expects
        roboflow_outputs = {
            "roboflow_predictions": detection_data["nestle_products"],
            "total_nestle": detection_data["total_nestle"],
            "total_competitor": detection_data["total_competitor"],
            "competitor_detections": detection_data["competitor_detections"]
        }

        # Read image file as binary
        with open(image_file, 'rb') as img_file:
            image_data = img_file.read()

        # Prepare multipart form data - use 'image0' as expected by /receive_data
        files = {
            'image0': (
                os.path.basename(image_file),
                image_data,
                'image/jpeg'
            )
        }

        # Form data fields exactly as server expects
        form_data = {
            "device_id": device_id,
            "timestamp": timestamp,
            "roboflow_outputs": json.dumps(roboflow_outputs)
        }

        print(f"Sending data to server: {server_url}/receive_data")
        print(f"Device ID: {device_id}")
        print(f"Timestamp: {timestamp}")
        print(f"Image file: {os.path.basename(image_file)}")

        # First try the main endpoint for device captures
        response = requests.post(
            f"{server_url}/receive_data",
            files=files,
            data=form_data,
            timeout=30
        )

        print(f"Server response status: {response.status_code}")
        print(f"Server response text: {response.text[:200]}")

        if response.status_code == 200:
            print("Successfully sent data to server")
            try:
                response_data = response.json()
                if 'success' in response_data and response_data['success']:
                    print(f"Server confirmed success: {response_data.get('message', '')}")
                    if 'id' in response_data:
                        print(f"Detection event ID: {response_data['id']}")
                    if 'image_path' in response_data:
                        print(f"Image saved at: {response_data['image_path']}")
            except:
                pass
            return True
        else:
            # Try alternate endpoint
            print("Trying alternate endpoint: /check_image")

            # For /check_image endpoint, use 'image' as the file parameter name
            alt_files = {
                'image': (
                    os.path.basename(image_file),
                    image_data,
                    'image/jpeg'
                )
            }

            alt_response = requests.post(
                f"{server_url}/check_image",
                files=alt_files,  # Different file parameter name
                timeout=30
            )

            print(f"Alternate endpoint response: {alt_response.status_code}")
            print(f"Response text: {alt_response.text[:200]}")

            if alt_response.status_code == 200:
                print("Successfully sent data via alternate endpoint")
                return True
            else:
                print(f"Failed to send data. All endpoints returned errors.")
                return False

    except Exception as e:
        print(f"Error sending data to server: {e}")
        return False

def main():
    frames_folder = "frames"
    os.makedirs(frames_folder, exist_ok=True)

    print("Starting motion detection system (Pi Zero optimized)...")
    print(f"Server URL: {SERVER_URL}")
    print(f"Device ID: {DEVICE_ID}")

    # For Raspberry Pi, increase file limits
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
        print("Increased file limits for Raspberry Pi")
    except Exception as e:
        print(f"Could not increase file limits: {e}")

    while True:
        try:
            print("\nWaiting for motion...")
            # Force garbage collection before capture
            gc.collect()

            # Use the Pi Zero optimized capture function
            saved_frames = detect_motion_and_capture_frames(frames_folder)

            if not saved_frames:
                print("No frames captured, continuing to monitor...")
                time.sleep(1)
                continue

            print(f"Captured {len(saved_frames)} frames.")

            best_frame = None
            best_detection_count = -1
            best_detection_data = None
            best_labeled_image = None

            # Process each frame
            for frame_path in saved_frames:
                try:
                    # Force garbage collection before processing
                    gc.collect()

                    print(f"Processing frame: {frame_path}")
                    labeled_image, detection_data = process_image(frame_path)

                    if labeled_image is not None and detection_data is not None:
                        total_detections = detection_data["total_nestle"] + detection_data["total_competitor"]

                        if total_detections > best_detection_count:
                            best_detection_count = total_detections
                            best_frame = frame_path
                            best_detection_data = detection_data
                            best_labeled_image = labeled_image

                    # Force garbage collection after processing
                    gc.collect()
                except Exception as e:
                    print(f"Error processing frame {frame_path}: {str(e)}")
                    continue

            # Send best frame to server
            if best_frame and best_labeled_image is not None:
                # Save the best labeled image
                labeled_path = best_frame.replace("frame_", "labeled_")
                cv2.imwrite(labeled_path, best_labeled_image)
                print(f"Saved best labeled image to: {labeled_path}")

                # Send to server
                if send_to_server(best_detection_data, labeled_path):
                    print("Successfully sent data to server")
                else:
                    print("Failed to send data to server")

                # Cleanup old frames
                try:
                    for frame in saved_frames:
                        if os.path.exists(frame):
                            os.remove(frame)
                    print("Cleaned up temporary frames")
                except Exception as e:
                    print(f"Error cleaning up frames: {e}")
            else:
                print("No valid detections found in captured frames")

            # Force garbage collection
            gc.collect()

            # Short delay before next detection
            time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping motion detection system...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")