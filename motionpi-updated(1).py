import cv2
import time
import json
import os
from datetime import datetime
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv

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

# =========== OWLv2 Setup ===========
OWLV2_API_KEY = "bjJkZXZrb2Y1cDMzMXh3OHdzbGl6OlFQOHVmS2JkZjBmQUs2bnF2OVJVdXFoNnc0ZW5kN1hH"
OWLV2_PROMPTS = ["bottle", "tetra pak", "cans", "carton drink"]

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
yolo_model = project.version(ROBOFLOW_VERSION).model

# =========== Dino-X Setup ===========
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt

DINOX_API_KEY = "af1a81e6517f5cafb12752f7de7f6be9"
DINOX_PROMPT = "beverage . bottle . cans . boxed milk . milk"

dinox_config = Config(DINOX_API_KEY)
dinox_client = Client(dinox_config)

def is_overlap(box1, boxes2, threshold=0.3):
    """
    Check if box1 overlaps with any box in boxes2
    box1: [x1, y1, x2, y2]
    boxes2: list of [x_center, y_center, width, height]
    """
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

def detect_combined(frame_path):
    """Detect both Nestle and competitor products"""
    try:
        # ========== [1] YOLO: Deteksi Produk Nestlé ==========
        yolo_pred = yolo_model.predict(frame_path, confidence=50, overlap=80).json()
        
        # Simpan bounding box Nestle (format: (x_center, y_center, width, height))
        nestle_boxes = []
        for pred in yolo_pred['predictions']:
            nestle_boxes.append((pred['x'], pred['y'], pred['width'], pred['height']))

        # ========== [2] OWLv2: Deteksi Kompetitor ==========
        headers = {
            "Authorization": "Basic " + OWLV2_API_KEY,
        }
        data = {
            "prompts": OWLV2_PROMPTS,
            "model": "owlv2"
        }
        with open(frame_path, "rb") as f:
            files = {"image": f}
            response = requests.post(
                "https://api.landing.ai/v1/tools/text-to-object-detection",
                files=files,
                data=data,
                headers=headers
            )
        owlv2_result = response.json()

        # Process competitor detections
        competitor_boxes = []
        if 'data' in owlv2_result and owlv2_result['data']:
            for obj in owlv2_result['data'][0]:
                if 'bounding_box' in obj:
                    bbox = obj['bounding_box']  # [x1, y1, x2, y2]
                    if not is_overlap(bbox, nestle_boxes):
                        competitor_boxes.append({
                            "box": bbox,
                            "confidence": obj.get("score", 0),
                            "class": "unclassified"
                        })

        # Prepare detection data
        detection_data = {
            'roboflow_predictions': yolo_pred['predictions'],
            'dinox_predictions': competitor_boxes,
            'counts': {
                'nestle': len(nestle_boxes),
                'competitor': len(competitor_boxes)
            }
        }

        return detection_data

    except Exception as e:
        print(f"Error in combined detection: {e}")
        return None

def detect_motion_and_capture_video(frames_folder):
    """Capture frames when motion is detected"""
    picam2 = None
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_still_configuration(main={"size": (1920, 1080)})
        picam2.configure(config)
        picam2.start()
        print("Picam2 camera started for motion detection.")
        time.sleep(2)

        prev_frame = picam2.capture_array()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

        captured_frames = []
        frame_count = 0

        while frame_count < 5:  # Capture 5 frames
            current_frame = picam2.capture_array()
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

            frame_diff = cv2.absdiff(prev_gray, current_gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            motion_detected = np.sum(thresh) > 500000

            if motion_detected:
                frame_path = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, current_frame)
                captured_frames.append(frame_path)
                frame_count += 1

            prev_gray = current_gray
            time.sleep(0.1)

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

def main():
    frames_folder = "frames"
    os.makedirs(frames_folder, exist_ok=True)

    print("Starting motion detection system...")
    print(f"Server URL: {SERVER_URL}")
    print(f"Device ID: {DEVICE_ID}")

    # Increase system file limit
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    print(f"Increased file limit from {soft} to {hard}")

    while True:
        try:
            print("\nWaiting for motion...")
            saved_frames = detect_motion_and_capture_video(frames_folder)
            
            if not saved_frames:
                print("No motion detected, continuing to monitor...")
                time.sleep(1)  # Add delay to prevent tight loop
                continue

            print(f"Motion detected! Processing {len(saved_frames)} frames...")

            # Process each frame for detection
            best_frame = None
            best_detections = None
            max_total_objects = 0

            for frame in saved_frames:
                detections = detect_combined(frame)
                if detections:
                    total_objects = (detections['counts']['nestle'] + 
                                   detections['counts']['competitor'])
                    if total_objects > max_total_objects:
                        max_total_objects = total_objects
                        best_frame = frame
                        best_detections = detections

            if best_frame and best_detections:
                # Send data to server
                timestamp = datetime.now().isoformat()
                
                with open(best_frame, 'rb') as f:
                    files = {'image0': f}
                    data = {
                        'device_id': DEVICE_ID,
                        'timestamp': timestamp,
                        'roboflow_outputs': json.dumps(best_detections)
                    }
                    
                    try:
                        response = requests.post(SERVER_URL, files=files, data=data)
                        if response.status_code == 200:
                            print("Data sent successfully!")
                            print(f"Nestle products: {best_detections['counts']['nestle']}")
                            print(f"Competitor products: {best_detections['counts']['competitor']}")
                        else:
                            print(f"Error sending data: {response.status_code}")
                    except Exception as e:
                        print(f"Error in server communication: {e}")

            # Cleanup frames
            try:
                for frame in saved_frames:
                    if os.path.exists(frame):
                        os.remove(frame)
                print("Cleaned up temporary frames")
            except Exception as e:
                print(f"Error cleaning up frames: {e}")

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