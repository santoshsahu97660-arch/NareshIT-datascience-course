import inspect

# -------- PATCH for Python 3.14 (NLTK fix) --------
if not hasattr(inspect, "formatargspec"):
    def formatargspec(*args, **kwargs):
        return ""
    inspect.formatargspec = formatargspec

import streamlit as st
from nltk.chat.util import Chat, reflections
from gtts import gTTS
import random
import pandas as pd
import os
from datetime import datetime

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Ultimate AI Chatbot", page_icon="🤖", layout="wide")

# ---------- LOGIN SYSTEM ----------
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid Credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.title("⚙ Settings")

language = st.sidebar.selectbox("🌍 Language", ["English", "Hindi", "Spanish"])
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# ---------- DARK MODE ----------
if dark_mode:
    st.markdown("""
        <style>
        body {background-color: #0E1117; color: white;}
        </style>
    """, unsafe_allow_html=True)

# ---------- CHAT RULES ----------
pairs_en = [
    [r"(.*)my name is (.*)", ["Hello %2! Nice to meet you"]],
    [r"(hi|hello|hey)(.*)", ["Hello!", "Hey there!"]],
    [r"(.*)", ["Tell me more...", "Interesting!", "I understand"]],
]

pairs_hi = [
    [r"(.*)mera naam (.*)", ["Namaste %2!"]],
    [r"(namaste|hello)(.*)", ["Namaste!"]],
    [r"(.*)", ["Aur batao...", "Samajh gaya"]],
]

pairs_sp = [
    [r"(.*)mi nombre es (.*)", ["Hola %2!"]],
    [r"(hola)(.*)", ["Hola amigo!"]],
    [r"(.*)", ["Cuéntame más"]],
]

if language == "English":
    chat = Chat(pairs_en, reflections)
elif language == "Hindi":
    chat = Chat(pairs_hi, reflections)
else:
    chat = Chat(pairs_sp, reflections)

# ---------- SESSION MEMORY ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "analytics" not in st.session_state:
    st.session_state.analytics = []

# ---------- VOICE FUNCTION ----------
def speak(text, lang_code="en"):
    tts = gTTS(text=text, lang=lang_code)
    tts.save("voice.mp3")
    audio_file = open("voice.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es"
}

# ---------- UI ----------
st.title("🤖 Ultimate AI Chatbot")

col1, col2 = st.columns([3,1])

with col1:
    user_input = st.text_input("Type your message")

with col2:
    send_btn = st.button("Send")

if send_btn and user_input:
    response = chat.respond(user_input)

    if not response:
        response = random.choice(["Interesting!", "Tell me more", "I see"])

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

    # Analytics tracking
    st.session_state.analytics.append({
        "time": datetime.now(),
        "user": user_input,
        "bot": response,
        "language": language
    })

# ---------- DISPLAY CHAT ----------
for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

# ---------- VOICE BUTTON ----------
if st.button("🔊 Voice Reply"):
    if st.session_state.history:
        last_msg = st.session_state.history[-1][1]
        speak(last_msg, lang_map[language])

# ---------- CHAT MEMORY AI ----------
st.subheader("🧠 Memory Summary")
if st.session_state.history:
    last_msgs = [msg for sender, msg in st.session_state.history[-4:]]
    st.write("Recent conversation:", " | ".join(last_msgs))

# ---------- ANALYTICS DASHBOARD ----------
st.subheader("📊 Chat Analytics")

if st.session_state.analytics:
    df = pd.DataFrame(st.session_state.analytics)

    st.write("Total Messages:", len(df))
    st.write("Languages Used:", df["language"].unique())

    st.dataframe(df)

# ---------- LOGOUT ----------
if st.button("Logout"):
    st.session_state.logged_in = False
    st.experimental_rerun()
