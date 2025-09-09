import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile, os
import psycopg2
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from sentence_transformers import SentenceTransformer
import boto3

# Load YOLO model
model = YOLO("No_Helmet_best.pt")
class_names = ['no helmet']

# AWS S3 config
AWS_CONFIG = {
    "bucket_name": "bucket",
    "region": "ap-south-8"
}
s3_client = boto3.client('s3', region_name=AWS_CONFIG["region"])

# PostgreSQL config
DB_CONFIG = {
    "host": "123",
    "port": "5432",
    "user": "post",
    "password": "****",
    "dbname": "123"
}

# Email alert config
email_config = {
    "sender": "xyz",
    "recipient": "lmn",
    "smtp_server": "google",
    "smtp_port": 587,
    "password": "cvvn"
}

# Embedding model
embedder = SentenceTransformer('123')

# Upload to S3
def upload_to_s3(file_path, filename):
    try:
        s3_client.upload_file(file_path, AWS_CONFIG["bucket_name"], filename)
        return f"https://{AWS_CONFIG['bucket_name']}.s3.{AWS_CONFIG['region']}.amazonaws.com/{filename}"
    except Exception as e:
        st.error(f"S3 upload error: {e}")
        return None

# Generate embedding
def generate_embedding(metadata):
    text = f"{metadata['class_name']} {metadata['confidence']} {metadata['bbox']} {metadata['timestamp']}"
    return embedder.encode(text).tolist()

# Log to PostgreSQL
def log_to_db(data):
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            dbname=DB_CONFIG["dbname"]
        )
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS helmet_logs (
                id SERIAL PRIMARY KEY,
                s3_url TEXT,
                timestamp TIMESTAMP,
                frame_number INT,
                class_name TEXT,
                confidence FLOAT,
                bbox TEXT,
                embedding VECTOR(384)
            )
        """)
        cursor.execute("""
            INSERT INTO helmet_logs (s3_url, timestamp, frame_number, class_name, confidence, bbox, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            data["s3_url"],
            data["timestamp"],
            data.get("frame_number", None),
            data["class_name"],
            data["confidence"],
            str(data["bbox"]),
            data["embedding"]
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

# Send email alert
def send_email_alert(filename, detection_info):
    try:
        msg = MIMEText(f"No Helmet detected in {filename} with confidence {detection_info['confidence']:.2f}")
        msg["Subject"] = "🚨 No Helmet Detection Alert"
        msg["From"] = email_config["sender"]
        msg["To"] = email_config["recipient"]

        with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
            server.starttls()
            server.login(email_config["sender"], email_config["password"])
            server.send_message(msg)
    except Exception as e:
        st.error(f"Email error: {e}")

# Image detection
def detect_image(image, filename):
    img = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img, (640, 640))
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    image.save(temp_path)
    s3_url = upload_to_s3(temp_path, filename)

    results = model(img_resized, stream=True)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_names[cls_id]} {conf:.2f}"
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img_resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            metadata = {
                "s3_url": s3_url,
                "timestamp": datetime.now(),
                "frame_number": None,
                "class_name": class_names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "embedding": generate_embedding({
                    "class_name": class_names[cls_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": datetime.now()
                })
            }
            log_to_db(metadata)
            send_email_alert(filename, metadata)

    return img_resized

# Video detection
def detect_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    output_frames = []
    frame_count = 0
    s3_url = upload_to_s3(video_path, filename)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_resized = cv2.resize(frame, (640, 640))
        results = model(frame_resized, stream=True)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame_resized, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                metadata = {
                    "s3_url": s3_url,
                    "timestamp": datetime.now(),
                    "frame_number": frame_count,
                    "class_name": class_names[cls_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "embedding": generate_embedding({
                        "class_name": class_names[cls_id],
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "timestamp": datetime.now()
                    })
                }
                log_to_db(metadata)
                send_email_alert(filename, metadata)

        output_frames.append(frame_resized)
        frame_count += 1

    cap.release()
    return output_frames

# Streamlit UI
st.title("🛵 No Helmet Detection Dashboard")
st.markdown("Upload an image or video to detect helmet violations and log them to AWS")

option = st.radio("Choose input type:", ["Image", "Video"])

if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_column_width=True)
        result_img = detect_image(image, uploaded_image.name)
        st.image(result_img, caption="Detected Violations", use_column_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_path = tmp_file.name

        st.video(uploaded_video)
        st.markdown("Processing video... This may take a moment.")
        frames = detect_video(tmp_path, uploaded_video.name)

        st.markdown("Preview of detected frames:")
        for i, frame in enumerate(frames[:10]):
            st.image(frame, caption=f"Frame {i+1}", use_column_width=True)

        os.remove(tmp_path)