import gradio as gr
from dotenv import load_dotenv
from roboflow import Roboflow
import tempfile
import os
import requests
import cv2
import numpy as np
import subprocess

# ========== Konfigurasi ==========
load_dotenv()

# Roboflow Config
rf_api_key = os.getenv("ROBOFLOW_API_KEY")
workspace = os.getenv("ROBOFLOW_WORKSPACE")
project_name = os.getenv("ROBOFLOW_PROJECT")
model_version = int(os.getenv("ROBOFLOW_MODEL_VERSION"))

# OWLv2 Config
OWLV2_API_KEY = os.getenv("COUNTGD_API_KEY")
OWLV2_PROMPTS = ["bottle", "tetra pak","cans", "carton drink"]

# Inisialisasi Model YOLO
rf = Roboflow(api_key=rf_api_key)
project = rf.workspace(workspace).project(project_name)
yolo_model = project.version(model_version).model

# ========== Fungsi Deteksi Kombinasi ==========
def detect_combined(image):
    # Simpan gambar input ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file, format="JPEG")
        temp_path = temp_file.name

    try:
        # ========== [1] YOLO: Deteksi Produk Nestlé (Per Class) ==========
        yolo_pred = yolo_model.predict(temp_path, confidence=50, overlap=80).json()

        # Hitung per class Nestlé dan simpan bounding box (format: (x_center, y_center, width, height))
        nestle_class_count = {}
        nestle_boxes = []
        for pred in yolo_pred['predictions']:
            class_name = pred['class']
            nestle_class_count[class_name] = nestle_class_count.get(class_name, 0) + 1
            nestle_boxes.append((pred['x'], pred['y'], pred['width'], pred['height']))

        total_nestle = sum(nestle_class_count.values())

        # ========== [2] OWLv2: Deteksi Kompetitor ==========
        headers = {
            "Authorization": "Basic " + OWLV2_API_KEY,
        }
        data = {
            "prompts": OWLV2_PROMPTS,
            "model": "owlv2"
        }
        with open(temp_path, "rb") as f:
            files = {"image": f}
            response = requests.post("https://api.landing.ai/v1/tools/text-to-object-detection", files=files, data=data, headers=headers)
        result = response.json()
        owlv2_objects = result['data'][0] if 'data' in result else []

        competitor_class_count = {}
        competitor_boxes = []
        for obj in owlv2_objects:
            if 'bounding_box' in obj:
                bbox = obj['bounding_box']  # Format: [x1, y1, x2, y2]
                # Filter objek yang sudah terdeteksi oleh YOLO (Overlap detection)
                if not is_overlap(bbox, nestle_boxes):
                    class_name = obj.get('label', 'unknown').strip().lower()
                    competitor_class_count[class_name] = competitor_class_count.get(class_name, 0) + 1
                    competitor_boxes.append({
                        "class": class_name,
                        "box": bbox,
                        "confidence": obj.get("score", 0)
                    })

        total_competitor = sum(competitor_class_count.values())

        # ========== [3] Format Output ==========
        result_text = "Product Nestle\n\n"
        for class_name, count in nestle_class_count.items():
            result_text += f"{class_name}: {count}\n"
        result_text += f"\nTotal Products Nestle: {total_nestle}\n\n"
        if competitor_class_count:
            result_text += f"Total Unclassified Products: {total_competitor}\n"
        else:
            result_text += "No Unclassified Products detected\n"

        # ========== [4] Visualisasi ==========
        img = cv2.imread(temp_path)

        # Gambar bounding box untuk produk Nestlé (Hijau)
        for pred in yolo_pred['predictions']:
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
            cv2.putText(img, pred['class'], (int(x - w/2), int(y - h/2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Gambar bounding box untuk kompetitor (Merah) dengan label 'unclassified' jika sesuai
        for comp in competitor_boxes:
            x1, y1, x2, y2 = comp['box']
            unclassified_classes = ["cans"]
            display_name = "unclassified" if any(cls in comp['class'].lower() for cls in unclassified_classes) else comp['class']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"{display_name} {comp['confidence']:.2f}",
                        (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        output_path = "/tmp/combined_output.jpg"
        cv2.imwrite(output_path, img)

        return output_path, result_text

    except Exception as e:
        return temp_path, f"Error: {str(e)}"
    finally:
        os.remove(temp_path)

def is_overlap(box1, boxes2, threshold=0.3):
    """
    Fungsi untuk mendeteksi overlap bounding box.
    Parameter:
      - box1: Bounding box pertama dengan format (x1, y1, x2, y2)
      - boxes2: List bounding box lainnya dengan format (x_center, y_center, width, height)
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
            if area_overlap / area_box1 > threshold:
                return True
    return False

# ========== Fungsi untuk Deteksi Video ==========
def convert_video_to_mp4(input_path, output_path):
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        return None, f"Error converting video: {e}"

def detect_objects_in_video(video_path):
    temp_output_path = "/tmp/output_video.mp4"
    temp_frames_dir = tempfile.mkdtemp()
    all_class_count = {}  # Untuk menyimpan total hitungan objek dari semua frame
    nestle_total = 0
    frame_count = 0

    try:
        # Convert video ke MP4 jika perlu
        if not video_path.endswith(".mp4"):
            video_path, err = convert_video_to_mp4(video_path, temp_output_path)
            if not video_path:
                return None, f"Video conversion error: {err}"

        # Membaca dan memproses frame video
        video = cv2.VideoCapture(video_path)
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)

        # VideoWriter untuk output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(temp_output_path, fourcc, frame_rate, frame_size)

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Simpan frame untuk prediksi
            frame_path = os.path.join(temp_frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            # Proses prediksi untuk frame
            predictions = yolo_model.predict(frame_path, confidence=60, overlap=80).json()

            # Update hitungan objek untuk frame ini
            frame_class_count = {}
            for prediction in predictions['predictions']:
                class_name = prediction['class']
                frame_class_count[class_name] = frame_class_count.get(class_name, 0) + 1
                cv2.rectangle(frame, (int(prediction['x'] - prediction['width']/2),
                                      int(prediction['y'] - prediction['height']/2)),
                              (int(prediction['x'] + prediction['width']/2),
                               int(prediction['y'] + prediction['height']/2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, class_name, (int(prediction['x'] - prediction['width']/2),
                                                int(prediction['y'] - prediction['height']/2 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update hitungan kumulatif
            for class_name, count in frame_class_count.items():
                all_class_count[class_name] = all_class_count.get(class_name, 0) + count

            nestle_total = sum(all_class_count.values())

            # Overlay teks hitungan pada frame
            count_text = "Cumulative Object Counts\n"
            for class_name, count in all_class_count.items():
                count_text += f"{class_name}: {count}\n"
            count_text += f"\nTotal Product Nestlé: {nestle_total}"

            y_offset = 20
            for line in count_text.split("\n"):
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30

            output_video.write(frame)
            frame_count += 1

        video.release()
        output_video.release()

        return temp_output_path

    except Exception as e:
        return None, f"An error occurred: {e}"

# ========== Gradio Interface ==========
with gr.Blocks(theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate")) as iface:
    gr.Markdown("""<div style="text-align: center;"><h1>NESTLE - STOCK COUNTING</h1></div>""")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
        with gr.Column():
            output_image = gr.Image(label="Detect Object")
        with gr.Column():
            output_text = gr.Textbox(label="Counting Object")
    
    # Tombol untuk memproses input
    detect_button = gr.Button("Detect")
    
    # Hubungkan tombol dengan fungsi deteksi
    detect_button.click(
        fn=detect_combined, 
        inputs=input_image, 
        outputs=[output_image, output_text]
    )

iface.launch()