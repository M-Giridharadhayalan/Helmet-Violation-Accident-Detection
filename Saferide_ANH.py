import streamlit as st
from ultralytics import YOLO
import boto3
import psycopg2
import requests
from datetime import datetime
import os
from PIL import Image

# Configuration
YOLO_MODEL_PATH = "xxyx"  
S3_BUCKET = "ABCD"
RDS_CONFIG = {
    "host": "post",
    "database": "post",
    "user": "post",
    "password": "post"
}
TELEGRAM_BOT_TOKEN = "token"
TELEGRAM_CHAT_ID = "chatid"
LOCATION = "world"

# Ulity
def is_video(file_name):
    return file_name.lower().endswith((".mp4", ".avi", ".mov"))

def save_uploaded_file(uploaded_file):
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def detect_objects(file_path):
    model = YOLO(YOLO_MODEL_PATH)
    results = model.predict(file_path, save=True, conf=0.5)
    return results

def get_annotated_output(results, original_file):
    save_dir = results[0].save_dir
    base_name = os.path.basename(original_file)

    # Try exact match
    annotated_path = os.path.join(save_dir, base_name)
    if os.path.exists(annotated_path):
        return annotated_path

    # Fallback: find any video file in save_dir
    for file in os.listdir(save_dir):
        if file.lower().endswith((".mp4", ".avi", ".mov")):
            return os.path.join(save_dir, file)

    raise FileNotFoundError(f"❌ Annotated video not found in {save_dir}")

def upload_to_s3(file_path, bucket_name, object_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Cannot upload: file not found at {file_path}")
    
    s3 = boto3.client("s3")
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        s3_link = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        st.success("✅ Uploaded to S3 successfully!")
        return s3_link
    except Exception as e:
        st.error(f"❌ S3 upload failed: {e}")
        return None

def extract_metadata(results, s3_link, location):
    metadata = []
    timestamp = datetime.now().isoformat()
    for result in results:
        for box in result.boxes:
            cls = result.names[int(box.cls)]
            conf = round(float(box.conf), 2)
            metadata.append({
                "timestamp": timestamp,
                "location": location,
                "class_label": cls,
                "confidence": conf,
                "s3_link": s3_link
            })
    return metadata

def log_to_postgres(metadata_list, config):
    try:
        conn = psycopg2.connect(**config)
        cur = conn.cursor()

        # ✅ Ensure table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                location TEXT,
                class_label TEXT,
                confidence FLOAT,
                s3_link TEXT
            );
        """)

        # 📥 Insert metadata
        for data in metadata_list:
            cur.execute("""
                INSERT INTO detections (timestamp, location, class_label, confidence, s3_link)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                data["timestamp"],
                data["location"],
                data["class_label"],
                data["confidence"],
                data["s3_link"]
            ))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("❌ PostgreSQL logging failed:", e)

def send_telegram_alert(metadata, bot_token, chat_id):
    for data in metadata:
        emoji = "🪖" if data["class_label"] == "No Helmet" else "💥" if data["class_label"] == "Accident" else "⚠️"
        message = (
            f"{emoji} Detection Alert\n"
            f"📍 Location: {data['location']}\n"
            f"🕒 Time: {data['timestamp']}\n"
            f"🧠 Class: {data['class_label']}\n"
            f"📊 Confidence: {data['confidence']}\n"
            f"🖼️ Proof: {data['s3_link']}"
        )
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": message})

# Streamlit 
st.set_page_config(layout="wide")
st.title("🛡️ Saferide AI Predictor v1.0")

uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    results = detect_objects(file_path)
    annotated_path = get_annotated_output(results, file_path)
    s3_link = upload_to_s3(annotated_path, S3_BUCKET, uploaded_file.name)

    if is_video(uploaded_file.name):
        st.video(annotated_path)
    else:
        st.image(annotated_path, use_column_width=True)

    metadata = extract_metadata(results, s3_link, LOCATION)

    # Optional: Filter by class
    selected_class = st.selectbox("Filter by class", ["All", "No Helmet", "Accident"])
    if selected_class != "All":
        metadata = [m for m in metadata if m["class_label"] == selected_class]

    # Log and notify
    log_to_postgres(metadata, RDS_CONFIG)
    send_telegram_alert(metadata, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    # Summary
    st.success("✅ Detection pipeline completed successfully!")
    st.write("### Detection Summary")
    for m in metadata:
        st.write(f"- {m['class_label']} ({m['confidence']}) → [Proof]({m['s3_link']})")