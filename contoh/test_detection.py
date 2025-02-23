import cv2
import json
import os
import tempfile
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image

# =========== Roboflow & DINO-X Setup ===========
from roboflow import Roboflow
from dds_cloudapi_sdk import Config, Client, TextPrompt
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget

load_dotenv()

# --- Roboflow Configuration ---
rf_api_key = os.getenv("ROBOFLOW_API_KEY")
workspace = os.getenv("ROBOFLOW_WORKSPACE")
project_name = os.getenv("ROBOFLOW_PROJECT")
model_version = int(os.getenv("ROBOFLOW_MODEL_VERSION"))

rf = Roboflow(api_key=rf_api_key)
project = rf.workspace(workspace).project(project_name)
yolo_model = project.version(model_version).model

# --- DINO-X Configuration ---
DINOX_API_KEY = os.getenv("DINO_X_API_KEY")
DINOX_PROMPT = "beverage . bottle . cans . boxed milk . milk"

dinox_config = Config(DINOX_API_KEY)
dinox_client = Client(dinox_config)

# =========== Helper Functions (Diambil dari Gradio Code) ===========

def is_overlap(box1, boxes2, threshold=0.3):
    """
    LOGIC DARI GRADIO CODE: 
    - Hitung overlap relatif terhadap area box1 (DINO-X)
    - Ini penting untuk menghindari misclassification
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
            area_box1 = (x1_max - x1_min) * (y1_max - y1_min)  # Perubahan kunci di sini
            if area_overlap / area_box1 > threshold:
                return True
    return False

def convert_video_to_mp4(input_path, output_path):
    """Sama seperti di Gradio Code"""
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        return None

# =========== Detection Functions (Diperbaiki dengan logika Gradio) ===========

def process_image(image_path):
    """
    Memproses 1 gambar:
    1. Baca gambar pakai PIL, normalkan orientasi.
    2. Simpan sementara ke file JPEG.
    3. Prediksi YOLO + DINO-X.
    4. Gambar bounding box di OpenCV.
    5. Return (cv2_image, result_text, data_json).
    """
    try:
        # 1) Baca & perbaiki orientasi dengan PIL
        pil_img = Image.open(image_path).convert("RGB")
        # pil_img = fix_orientation(pil_img)

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

        # 4) DINO-X Detection
        image_url = dinox_client.upload_file(temp_path)
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
            # DINO-X return format [x1, y1, x2, y2]
            dinox_box = obj.bbox
            class_name = obj.category.strip().lower()

            # Pastikan tidak overlap dengan deteksi Nestle
            if not is_overlap(dinox_box, nestle_boxes):
                # Hanya masukkan jika "beverage, cans, bottle, milk, etc."
                if any(cls in class_name for cls in unclassified_classes):
                    competitor_class_count[class_name] = competitor_class_count.get(class_name, 0) + 1
                    competitor_boxes.append({
                        "box": dinox_box,
                        "confidence": obj.score,
                        "class": "unclassified"
                    })

        total_competitor = sum(competitor_class_count.values())

        # 5) Visualisasi bounding box dengan OpenCV
        cv_img = cv2.imread(temp_path)  # Baca file sementara yang sudah terjamin orientasinya

        # Nestle (Hijau)
        for pred in yolo_pred['predictions']:
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_img, pred['class'], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Kompetitor (Merah)
        for comp in competitor_boxes:
            x1, y1, x2, y2 = comp['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(cv_img,
                        f"{comp['class']} {comp['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Format output text
        result_text = "Product Nestle\n\n"
        for class_name, count in nestle_class_count.items():
            result_text += f"{class_name}: {count}\n"
        result_text += f"\nTotal Products Nestle: {total_nestle}\n\n"
        if total_competitor > 0:
            result_text += f"Total Unclassified Products: {total_competitor}\n"
        else:
            result_text += "No Unclassified Products detected\n"

        detection_data = {
            "nestle_products": nestle_class_count,
            "total_nestle": total_nestle,
            "total_competitor": total_competitor,
            "competitor_detections": competitor_boxes
        }

        return cv_img, result_text, detection_data

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, str(e), None

    finally:
        # Bersihkan file sementara
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

def detect_objects_in_video(video_path):
    """
    Process video frames with YOLO detection and overlay object counts.
    Returns the path to the output (labeled) video.
    """
    temp_output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
    temp_frames_dir = tempfile.mkdtemp()
    frame_count = 0

    try:
        # Convert video to MP4 if necessary.
        if not video_path.lower().endswith(".mp4"):
            converted = convert_video_to_mp4(video_path, temp_output_path)
            if not converted:
                return None, "Video conversion failed."
            video_path = converted

        video = cv2.VideoCapture(video_path)
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(temp_output_path, fourcc, frame_rate, frame_size)

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Save the current frame temporarily.
            frame_path = os.path.join(temp_frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            # Run YOLO detection on the frame.
            predictions = yolo_model.predict(frame_path, confidence=50, overlap=80).json()

            # Count and draw boxes.
            object_counts = {}
            for pred in predictions['predictions']:
                cls = pred['class']
                object_counts[cls] = object_counts.get(cls, 0) + 1
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Overlay counts text.
            count_text = ""
            total_count = 0
            for cls, count in object_counts.items():
                count_text += f"{cls}: {count}\n"
                total_count += count
            count_text += f"\nTotal: {total_count}"

            y_offset = 20
            for line in count_text.split("\n"):
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30

            output_video.write(frame)
            frame_count += 1

        video.release()
        output_video.release()

        return temp_output_path, "Video processed successfully."

    except Exception as e:
        return None, f"An error occurred: {e}"

# =========== Main Function ===========
def main():
    test_images_folder = "test_images"   # Folder with input images
    output_folder = "test_output"          # Folder to save results
    os.makedirs(output_folder, exist_ok=True)

    # Process each image file in the test_images folder.
    for image_file in os.listdir(test_images_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_images_folder, image_file)
            print(f"\nProcessing image: {image_file}")

            labeled_img, result_text, detection_data = process_image(image_path)
            if labeled_img is not None:
                # Save labeled image.
                output_image_path = os.path.join(output_folder, f"labeled_{image_file}")
                cv2.imwrite(output_image_path, labeled_img)
                print(f"Saved labeled image to: {output_image_path}")

                # Save result text.
                text_path = os.path.join(output_folder, f"results_{os.path.splitext(image_file)[0]}.txt")
                with open(text_path, 'w') as f:
                    f.write(result_text)
                print(f"Saved results text to: {text_path}")

                # Save JSON detection data.
                json_path = os.path.join(output_folder, f"data_{os.path.splitext(image_file)[0]}.json")
                with open(json_path, 'w') as f:
                    json.dump(detection_data, f, indent=2)
                print(f"Saved detection data to: {json_path}")
            else:
                print(f"Failed to process {image_file}")

    # (Optional) Process a video file if available.
    video_file = "test_video.mp4"  # Change to your video file path if needed.
    if os.path.exists(video_file):
        print(f"\nProcessing video: {video_file}")
        video_output, msg = detect_objects_in_video(video_file)
        if video_output:
            video_save_path = os.path.join(output_folder, "labeled_video.mp4")
            os.replace(video_output, video_save_path)
            print(f"Saved processed video to: {video_save_path}")
        else:
            print(f"Video processing failed: {msg}")

if __name__ == "__main__":
    main()
