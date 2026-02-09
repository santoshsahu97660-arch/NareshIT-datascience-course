import streamlit as st
from transformers import pipeline
from langdetect import detect
import PyPDF2
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

st.set_page_config(page_title="AI Summarizer", page_icon="🧠", layout="centered")

st.title("🧠 AI Multi-Feature Text Summarizer")

# ---------------- LOAD MODEL (FAST & STABLE) ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",   # lightweight model
        device=-1  # CPU
    )

summarizer = load_model()

# ---------------- TEXT SUMMARIZATION ----------------
def summarize_long_text(text):
    max_chunk = 900   # prevents token overflow
    text = text.replace("\n", " ")

    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    summary = ""
    for chunk in chunks:
        if len(chunk.strip()) < 40:
            continue
        result = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + " "

    return summary.strip()

# ---------------- LANGUAGE ----------------
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

# ---------------- PDF READER ----------------
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# ---------------- SPEECH TO TEXT ----------------
def speech_to_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    return r.recognize_google(audio)

# ---------------- TEXT TO SPEECH ----------------
def text_to_audio(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ---------------- INPUT OPTIONS ----------------
option = st.radio("Choose Input Type", ["Text", "PDF", "Voice"])

input_text = ""

# TEXT
if option == "Text":
    input_text = st.text_area("Enter Text", height=250)

# PDF
elif option == "PDF":
    pdf = st.file_uploader("Upload PDF", type="pdf")
    if pdf:
        input_text = read_pdf(pdf)
        st.success("PDF loaded successfully")

# VOICE
elif option == "Voice":
    audio = st.file_uploader("Upload WAV audio", type=["wav"])
    if audio:
        input_text = speech_to_text(audio)
        st.info(input_text)

# ---------------- SUMMARIZE ----------------
if st.button("Generate Summary 🚀"):

    if input_text.strip() == "":
        st.warning("No text found")
    else:
        with st.spinner("AI is summarizing..."):
            lang = detect_lang(input_text)
            summary = summarize_long_text(input_text)

        st.subheader("🌍 Detected Language")
        st.write(lang)

        st.subheader("📌 Summary")
        st.success(summary)

        audio_path = text_to_audio(summary)
        st.audio(audio_path)
