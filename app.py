import streamlit as st
import cv2
import easyocr
import mysql.connector
import pandas as pd
import os
import re
import tempfile
import numpy as np
from PIL import Image
from db_config import DB_CONFIG

st.set_page_config(
    page_title="Vehicle Plate Detection System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Plate Detection System")


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource
def load_reader():
    return easyocr.Reader(["en"], gpu=False)

reader = load_reader()

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def save_to_db(plate_text, source_type, file_path=""):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
        INSERT INTO plate_records (plate_text, source_type, image_path)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (plate_text, source_type, file_path))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

def fetch_records():
    try:
        conn = get_connection()
        df = pd.read_sql("SELECT * FROM plate_records ORDER BY detected_at DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Fetch Error: {e}")
        return pd.DataFrame()

def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

def is_plate_like(text):
    text = clean_text(text)

    if len(text) < 6 or len(text) > 12:
        return False

    has_letter = any(ch.isalpha() for ch in text)
    has_digit = any(ch.isdigit() for ch in text)

    if not has_letter or not has_digit:
        return False

    return True

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray

def detect_plate_text(image):
    processed = preprocess_image(image)

    results = reader.readtext(processed)

    detections = []

    for bbox, text, conf in results:
        cleaned = clean_text(text)

        if conf >= 0.20 and is_plate_like(cleaned):
            detections.append({
                "text": cleaned,
                "confidence": conf,
                "bbox": bbox
            })

    return detections

def draw_boxes(image, detections):
    output = image.copy()

    for det in detections:
        bbox = det["bbox"]
        text = det["text"]
        conf = det["confidence"]

        points = np.array(bbox).astype(int)

        cv2.polylines(output, [points], True, (0, 255, 0), 2)

        x = points[0][0]
        y = points[0][1] - 10

        cv2.putText(
            output,
            f"{text} {conf:.2f}",
            (x, max(y, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return output

mode = st.sidebar.radio(
    "Choose Detection Mode",
    ["Image Upload", "Video Upload", "Live Webcam", "Database Records"]
)

if mode == "Image Upload":
    st.subheader("Image Upload")

    uploaded_file = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image_np, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Plate Text"):
            detections = detect_plate_text(image_np)

            if detections:
                result_image = draw_boxes(image_np, detections)
                st.image(result_image, caption="Detected Result", use_container_width=True)

                st.success("Detected plate text:")

                saved_texts = set()

                for det in detections:
                    plate_text = det["text"]
                    confidence = det["confidence"]

                    st.write(f"### `{plate_text}`")
                    st.write(f"Confidence: `{confidence:.2f}`")

                    if plate_text not in saved_texts:
                        save_to_db(plate_text, "image", file_path)
                        saved_texts.add(plate_text)

            else:
                st.warning("No plate text detected. Try a clearer image or crop closer to plate.")

elif mode == "Video Upload":
    st.subheader("Video Upload")

    uploaded_video = st.file_uploader(
        "Upload vehicle video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        video_path = os.path.join(UPLOAD_DIR, uploaded_video.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(video_path)

        rotate_option = st.selectbox(
            "If video is rotated, choose correction",
            ["No Rotate", "Rotate Clockwise", "Rotate Counter Clockwise"]
        )

        if st.button("Detect Plate Text From Video"):
            cap = cv2.VideoCapture(video_path)

            frame_count = 0
            detected_set = set()

            frame_area = st.empty()
            crop_area = st.empty()
            text_area = st.empty()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.info(f"Total frames in video: {total_frames}")

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Check every 2nd frame
                if frame_count % 2 != 0:
                    continue

                # Convert BGR to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Manual rotate correction
                if rotate_option == "Rotate Clockwise":
                    rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)

                elif rotate_option == "Rotate Counter Clockwise":
                    rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Resize for OCR
                rgb = cv2.resize(
                    rgb,
                    None,
                    fx=1.8,
                    fy=1.8,
                    interpolation=cv2.INTER_CUBIC
                )

                h, w, _ = rgb.shape

                # Crop center-lower area
                y1 = int(h * 0.30)
                y2 = int(h * 0.80)
                x1 = int(w * 0.10)
                x2 = int(w * 0.90)

                crop = rgb[y1:y2, x1:x2]

                frame_area.image(
                    rgb,
                    caption=f"Checking frame {frame_count}",
                    channels="RGB",
                    use_container_width=True
                )

                crop_area.image(
                    crop,
                    caption="Plate search area",
                    channels="RGB",
                    use_container_width=True
                )

                # OCR on full frame and crop
                detections_full = detect_plate_text(rgb)
                detections_crop = detect_plate_text(crop)

                all_detections = detections_full + detections_crop

                if all_detections:
                    for det in all_detections:
                        plate_text = det["text"]

                        if plate_text not in detected_set:
                            detected_set.add(plate_text)
                            save_to_db(plate_text, "video", video_path)

                    text_area.success("Detected: " + ", ".join(detected_set))

            cap.release()

            if detected_set:
                st.success("Video detection completed")
                st.write("Detected plates:")
                for plate in detected_set:
                    st.write(f"- `{plate}`")
            else:
                st.warning("No plate text detected in video. Try clearer/longer video or upload image frame.")           
                           
elif mode == "Live Webcam":
    st.subheader("Live Webcam")

    st.warning("For best result, keep plate close to camera and use good lighting.")

    run_camera = st.checkbox("Start Webcam")

    frame_area = st.empty()
    text_area = st.empty()

    detected_set = set()

    if run_camera:
        cap = cv2.VideoCapture(0)

        frame_count = 0

        while run_camera:
            ret, frame = cap.read()

            if not ret:
                st.error("Camera not working")
                break

            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_count % 30 == 0:
                detections = detect_plate_text(rgb)

                if detections:
                    result_frame = draw_boxes(rgb, detections)
                    frame_area.image(result_frame, channels="RGB", use_container_width=True)

                    for det in detections:
                        plate_text = det["text"]

                        if plate_text not in detected_set:
                            detected_set.add(plate_text)
                            save_to_db(plate_text, "live", "webcam")

                    text_area.success("Detected: " + ", ".join(detected_set))
                else:
                    frame_area.image(rgb, channels="RGB", use_container_width=True)
            else:
                frame_area.image(rgb, channels="RGB", use_container_width=True)

        cap.release()

else:
    st.subheader("Database Records")

    df = fetch_records()

    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No records found.")