import streamlit as st
import pandas as pd
import psycopg2
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_key = os.getenv("open")

# PostgreSQL config
DB_CONFIG = {
    "host": os.getenv("123"),
    "port": os.getenv("5432"),
    "dbname": os.getenv("xyz"),
    "user": os.getenv("post"),
    "password": os.getenv("***")
}

# UI setup
st.set_page_config(page_title="🧠 Safety Log Chatbot", layout="wide")
st.title("🚦 Chatbot for Accident & Helmet Detection Logs")

log_type = st.radio("Choose log type:", ["Accident Logs", "Helmet Logs"])
table_name = "accident_logs" if log_type == "Accident Logs" else "helmet_logs"

# Fetch logs from PostgreSQL
@st.cache_data
def fetch_logs(table):
    try:
        conn = psycopg2.connect("postgresql://123")
        query = f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 500"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

df_logs = fetch_logs(table_name)

# Build vectorstore
@st.cache_resource
def build_vectorstore(df):
    docs = []
    for _, row in df.iterrows():
        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(page_content=content))
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = build_vectorstore(df_logs)

# Build chatbot
llm = OpenAI(openai_api_key=openai_key, temperature=0.3)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Chat interface
query = st.text_input("Ask a question about the logs:")
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
    st.markdown(f"**Answer:** {response}")