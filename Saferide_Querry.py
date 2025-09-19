import streamlit as st
import pandas as pd
import psycopg2
import requests
import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

# Environment Loading
load_dotenv()
DB_HOST = os.getenv("")
DB_NAME = os.getenv("")
DB_USER = os.getenv("")
DB_PASS = os.getenv("")

OLLAMA_API = os.getenv("")
OLLAMA_MODEL = os.getenv("", "")

YAHOO_EMAIL = os.getenv("")
YAHOO_PASSWORD = os.getenv("")
EMAIL_RECIPIENT = os.getenv("")

# Database Utils
def fetch_detections():
    conn = psycopg2.connect(
        host=,
        database=,
        user=,
        password=,
        port=""
    )
    df = pd.read_sql("SELECT * FROM detections", conn)
    conn.close()
    return df

# Rag Chatbot
def query_rag(prompt):
    df = fetch_detections()
    context = df.to_markdown(index=False)

    full_prompt = f"""
You are a safety analytics assistant. Use the following accident and helmet violation data to answer the question.

Context:
{context}

Question:
{prompt}
"""

    response = requests.post(f"{OLLAMA_API}/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False
    })

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"❌ Error: {response.text}"

# Email Summary
def send_summary_email():
    df = fetch_detections()
    accidents = df[df["class_label"] == "Accident"].shape[0]
    no_helmet = df[df["class_label"] == "No Helmet"].shape[0]

    summary = f"""
🚨 Safety Summary Report
-------------------------
💥 Total Accidents: {accidents}
🪖 Helmet Violations: {no_helmet}
"""

    msg = MIMEText(summary)
    msg["Subject"] = "Daily Safety Summary"
    msg["From"] = YAHOO_EMAIL
    msg["To"] = EMAIL_RECIPIENT

    with smtplib.SMTP("smtp.mail.yahoo.com", 587) as server:
        server.starttls()
        server.login(YAHOO_EMAIL, YAHOO_PASSWORD)
        server.send_message(msg)

# Streamlit
st.set_page_config(page_title="🛡️ Safety RAG Dashboard", layout="wide")
st.title("🛡️ SafeRide AI Chatbot v1.0")
# 💬 RAG Chatbot
st.subheader("🧠 Ask a question ")
query = st.text_area("Your question", placeholder="e.g. Show helmet violations in Chennai last week")

if st.button("Run Query"):
    with st.spinner("Thinking..."):
        response = query_rag(query)
        st.markdown("### 🧠 Response")
        st.write(response)

# 📧 Email Trigger
st.subheader("📧 Send Email")
if st.button("Send Email Now"):
    send_summary_email()
    st.success("✅ Summary email sent successfully!")