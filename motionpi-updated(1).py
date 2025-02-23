import cv2
import time
import json
import os
from datetime import datetime
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import tempfile

# =========== Roboflow Setup ===========
from roboflow import Roboflow

# Load environment variables
load_dotenv()

SERVER_URL = os.getenv('SERVER_URL', 'http://16.78.246.103:5000/receive_data')
DEVICE_ID = os.getenv('DEVICE_ID', 'default_camera_01')

ROBOFLOW_API_KEY = "Otg64Ra6wNOgDyjuhMYU"
ROBOFLOW_WORKSPACE = "alat-pelindung-diri"
ROBOFLOW_PROJECT = "nescafe-4base"
ROBOFLOW_VERSION = 66

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
yolo_model = project.version(ROBOFLOW_VERSION).model

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

def process_image(image_path):
    """Process a single image with both YOLO and OWLv2 detections"""
    try:
        # 1) Baca & perbaiki orientasi dengan PIL
        pil_img = Image.open(image_path).convert("RGB")
        
        # 2) Simpan sementara ke file JPEG
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            pil_img.save(temp_file, format="JPEG")
            temp_path = temp_file.name

        # 3) YOLO Detection
        yolo_pred = yolo_model.predict(temp_path, confidence=50, overlap=80).json()

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

        # 5) Visualisasi bounding box dengan OpenCV
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

        # Format data untuk server
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

def detect_motion_and_capture_video(frames_folder, num_frames=5, video_duration=5,
                                    motion_threshold=500000, max_wait=10):
    """Capture frames when motion is detected"""
    picam2 = None
    try:
        # Tambah delay sebelum start camera
        time.sleep(1)
        
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_still_configuration(main={"size": (1920, 1080)})
        picam2.configure(config)
        picam2.start()
        print("Picam2 camera started for motion detection.")
        time.sleep(2)  # Tunggu kamera stabil

        # Batasi memory usage
        max_retries = 3
        for retry in range(max_retries):
            try:
                prev_frame = picam2.capture_array()
                break
            except Exception as e:
                print(f"Error capturing initial frame (attempt {retry+1}): {e}")
                time.sleep(1)
                if retry == max_retries - 1:
                    raise

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (640, 480))  # Resize untuk hemat memory
        motion_found = False
        start_time = time.time()

        print("Monitoring for motion...")
        while time.time() - start_time < max_wait:
            try:
                frame = picam2.capture_array()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (640, 480))  # Resize untuk analisis motion
                
                frame_diff = cv2.absdiff(prev_gray, gray)
                diff_sum = np.sum(frame_diff)
                
                if diff_sum > motion_threshold:
                    motion_found = True
                    print("Motion detected!")
                    break
                
                prev_gray = gray
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error during motion detection: {e}")
                time.sleep(0.5)
                continue

        if not motion_found:
            print("No significant motion detected within the wait time.")
            return []

        # Capture frames with memory management
        captured_frames = []
        interval = video_duration / num_frames
        
        for i in range(num_frames):
            try:
                frame = picam2.capture_array()
                if frame is not None:
                    frame_filename = os.path.join(frames_folder, f"frame_{i:03d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    captured_frames.append(frame_filename)
                    print(f"Saved frame {i+1}/{num_frames}")
                    
                    # Clear memory
                    del frame
                    time.sleep(interval)
                else:
                    print(f"Failed to capture frame {i+1}")
            except Exception as e:
                print(f"Error capturing frame {i+1}: {e}")
                continue

        return captured_frames

    except Exception as e:
        print(f"Error in motion detection/video capture: {e}")
        return []
        
    finally:
        if picam2:
            try:
                picam2.stop()
                picam2.close()
                print("Camera cleaned up successfully")
            except Exception as e:
                print(f"Error cleaning up camera: {e}")

def send_to_server(detection_data, image_file, server_url=SERVER_URL):
    """Send detection results and image to server"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_id = DEVICE_ID

    # Format data untuk server
    roboflow_outputs = {
        "roboflow_predictions": detection_data["nestle_products"],
        "total_nestle": detection_data["total_nestle"],
        "total_competitor": detection_data["total_competitor"],
        "competitor_detections": detection_data["competitor_detections"]
    }

    try:
        # Baca gambar dan encode ke base64
        with open(image_file, 'rb') as img_file:
            image_data = img_file.read()
        
        # Persiapkan multipart form data
        files = {
            'image0': (
                os.path.basename(image_file),
                image_data,
                'image/jpeg'
            )
        }
        
        data = {
            "device_id": device_id,
            "timestamp": timestamp,
            "roboflow_outputs": json.dumps(roboflow_outputs)
        }

        # Set headers
        headers = {
            'Accept': 'application/json',
        }

        print(f"Sending data to server: {server_url}")
        print(f"Image file: {image_file}")
        print(f"Detection data: {json.dumps(roboflow_outputs, indent=2)}")

        # Kirim request dengan timeout yang lebih lama
        response = requests.post(
            server_url,
            data=data,
            files=files,
            headers=headers,
            timeout=30  # 30 detik timeout
        )

        print(f"Server response status: {response.status_code}")
        print(f"Server response text: {response.text}")

        if response.status_code == 200:
            print("Successfully sent data and image to server")
            return True
        else:
            print(f"Failed to send data. Server returned status code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Network error when sending data: {e}")
        return False
    except Exception as e:
        print(f"Error sending data to server: {e}")
        return False

def main():
    frames_folder = "frames"
    os.makedirs(frames_folder, exist_ok=True)

    print("Starting motion detection system...")
    print(f"Server URL: {SERVER_URL}")
    print(f"Device ID: {DEVICE_ID}")

    consecutive_errors = 0
    max_consecutive_errors = 5

    while True:
        try:
            if consecutive_errors >= max_consecutive_errors:
                print(f"Too many consecutive errors ({consecutive_errors}). Waiting 60 seconds...")
                time.sleep(60)
                consecutive_errors = 0

            print("\nWaiting for motion...")
            saved_frames = detect_motion_and_capture_video(frames_folder)
            
            if not saved_frames:
                print("No motion detected, continuing to monitor...")
                time.sleep(1)
                continue

            # Process frames with memory management
            best_frame = None
            best_detection_count = -1
            best_detection_data = None
            best_labeled_image = None

            for frame_path in saved_frames:
                try:
                    labeled_image, detection_data = process_image(frame_path)
                    if labeled_image is not None and detection_data is not None:
                        total_detections = detection_data["total_nestle"] + detection_data["total_competitor"]
                        
                        if total_detections > best_detection_count:
                            if best_labeled_image is not None:
                                del best_labeled_image
                            best_detection_count = total_detections
                            best_frame = frame_path
                            best_detection_data = detection_data
                            best_labeled_image = labeled_image.copy()
                            
                        del labeled_image  # Clear memory

                except Exception as e:
                    print(f"Error processing frame {frame_path}: {str(e)}")
                    consecutive_errors += 1
                    continue

            # Cleanup frames immediately after processing
            for frame in saved_frames:
                if os.path.exists(frame):
                    try:
                        os.remove(frame)
                    except Exception as e:
                        print(f"Error removing frame {frame}: {e}")

            # Send best frame if found
            if best_frame and best_labeled_image is not None:
                labeled_path = best_frame.replace("frame_", "labeled_")
                cv2.imwrite(labeled_path, best_labeled_image)
                
                if send_to_server(best_detection_data, labeled_path):
                    print("Successfully sent data to server")
                    consecutive_errors = 0
                else:
                    print("Failed to send data to server")
                    consecutive_errors += 1

                # Cleanup labeled image
                if os.path.exists(labeled_path):
                    try:
                        os.remove(labeled_path)
                    except Exception as e:
                        print(f"Error removing labeled image: {e}")

            time.sleep(1)  # Short delay before next detection

        except KeyboardInterrupt:
            print("\nStopping motion detection system...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            consecutive_errors += 1
            time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")