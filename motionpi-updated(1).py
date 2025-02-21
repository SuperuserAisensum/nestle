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
    """Process a single image with both YOLO and DINO-X detections"""
    try:
        # 1) Read image with PIL
        pil_img = Image.open(image_path).convert("RGB")
        
        # 2) YOLO Detection
        yolo_pred = yolo_model.predict(image_path, confidence=50, overlap=80).json()

        nestle_class_count = {}
        nestle_boxes = []
        for pred in yolo_pred['predictions']:
            class_name = pred['class']
            nestle_class_count[class_name] = nestle_class_count.get(class_name, 0) + 1
            nestle_boxes.append((pred['x'], pred['y'], pred['width'], pred['height']))
        total_nestle = sum(nestle_class_count.values())

        # 3) DINO-X Detection
        image_url = dinox_client.upload_file(image_path)
        task = DinoxTask(
            image_url=image_url,
            prompts=[TextPrompt(text=DINOX_PROMPT)],
            bbox_threshold=0.25,
            targets=[DetectionTarget.BBox]
        )
        dinox_client.run_task(task)
        dinox_pred = task.result.objects

        competitor_boxes = []
        competitor_class_count = {}
        unclassified_classes = ["beverage", "cans", "bottle", "boxed milk", "milk"]

        for obj in dinox_pred:
            dinox_box = obj.bbox
            class_name = obj.category.strip().lower()

            if not is_overlap(dinox_box, nestle_boxes):
                if any(cls in class_name for cls in unclassified_classes):
                    competitor_class_count[class_name] = competitor_class_count.get(class_name, 0) + 1
                    competitor_boxes.append({
                        "box": dinox_box,
                        "confidence": obj.score,
                        "class": "unclassified"
                    })

        total_competitor = sum(competitor_class_count.values())

        # 4) Draw bounding boxes
        cv_img = cv2.imread(image_path)

        # Draw Nestle boxes (green)
        for pred in yolo_pred['predictions']:
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_img, pred['class'], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Draw Competitor boxes (red)
        for comp in competitor_boxes:
            x1, y1, x2, y2 = comp['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(cv_img,
                        f"{comp['class']} {comp['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # 5) Prepare detection data
        detection_data = {
            "roboflow_predictions": yolo_pred.get('predictions', []),
            "dinox_predictions": competitor_boxes,
            "total_detections": total_nestle + total_competitor
        }

        return cv_img, detection_data

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None

def detect_motion_and_capture_video(frames_folder, num_frames=5, video_duration=5,
                                    motion_threshold=500000, max_wait=10):
    """Capture frames when motion is detected"""
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
        motion_found = False
        start_time = time.time()

        while time.time() - start_time < max_wait:
            frame = picam2.capture_array()
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
            picam2.stop()
            return []

        captured_frames = []
        interval = video_duration / num_frames
        for i in range(num_frames):
            frame = picam2.capture_array()
            frame_filename = os.path.join(frames_folder, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_filename, frame)
            captured_frames.append(frame_filename)
            print(f"Saved frame to {frame_filename}")
            time.sleep(interval)
        picam2.stop()
        return captured_frames
    except Exception as e:
        print(f"Error in motion detection/video capture: {e}")
        return []

def send_to_server(detection_data, image_file, server_url=SERVER_URL):
    """Send detection results and image to server"""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    device_id = DEVICE_ID

    data = {
        "device_id": device_id,
        "timestamp": timestamp,
        "roboflow_outputs": json.dumps(detection_data)
    }

    filename = f"{device_id}_{timestamp}_frame.jpg"
    files = [('image0', (filename, open(image_file, 'rb'), 'image/jpeg'))]

    try:
        print(f"Sending data to server: {server_url}")
        response = requests.post(server_url, data=data, files=files)
        print(f"Server response status: {response.status_code}")
        print(f"Server response text: {response.text}")
        return response.ok
    except Exception as e:
        print(f"Error sending data to server: {e}")
        return False

def main():
    frames_folder = "frames"
    os.makedirs(frames_folder, exist_ok=True)

    saved_frames = detect_motion_and_capture_video(frames_folder)
    if not saved_frames:
        print("Error: No motion detected or frames captured!")
        return

    print(f"Captured {len(saved_frames)} frames after motion detection.")

    best_frame = None
    best_detection_count = -1
    best_detection_data = None
    best_labeled_image = None

    for frame_path in saved_frames:
        try:
            print(f"Processing frame: {frame_path}")
            labeled_image, detection_data = process_image(frame_path)
            
            if labeled_image is not None and detection_data is not None:
                total_detections = detection_data["total_detections"]
                
                if total_detections > best_detection_count:
                    best_detection_count = total_detections
                    best_frame = frame_path
                    best_detection_data = detection_data
                    best_labeled_image = labeled_image

        except Exception as e:
            print(f"Error processing frame {frame_path}: {str(e)}")

    if best_frame and best_labeled_image is not None:
        # Save the best labeled image
        labeled_path = best_frame.replace("frame_", "labeled_")
        cv2.imwrite(labeled_path, best_labeled_image)
        print(f"Saved best labeled image to: {labeled_path}")

        # Send to server
        send_to_server(best_detection_data, labeled_path)
    else:
        print("No valid detections found in any frame!")

if __name__ == "__main__":
    main()