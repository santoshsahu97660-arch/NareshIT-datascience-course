import inspect

# -------- PATCH for Python 3.14 (NLTK fix) --------
if not hasattr(inspect, "formatargspec"):
    def formatargspec(*args, **kwargs):
        return ""
    inspect.formatargspec = formatargspec

import streamlit as st
from nltk.chat.util import Chat, reflections
import random
from gtts import gTTS
import os
import base64

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Smart NLTK Chatbot", page_icon="🤖", layout="centered")

# ---------- LOGIN SYSTEM ----------
def login():
    st.title("🔐 Login to Chatbot")
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

# ---------- CHATBOT RULES ----------
pairs = [
    [r"(.*)my name is (.*)", ["Hello %2, How are you today?"]],
    [r"(hi|hello|hey)(.*)", ["Hello!", "Hey there!"]],
    [r"how are you (.*)", ["I'm doing great!", "Feeling awesome!"]],
    [r"(.*)help(.*)", ["I am here to help you!"]],
    [r"(.*) your name ?", ["I am your AI chatbot 🤖"]],
    [r"(.*)created(.*)", ["I was created using Python and NLTK"]],
    [r"(.*)sports(.*)", ["I love Cricket!"]],
    [r"quit", ["Bye! See you soon"]],
    [r"(.*)", ["Tell me more...", "Interesting!", "Can you explain more?"]],
]

chat = Chat(pairs, reflections)

# ---------- AI SMART REPLY ----------
def ai_reply(user_input):
    smart_responses = [
        "That's interesting!",
        "Can you tell me more?",
        "I understand.",
        "Sounds good!",
        "Let's talk more about it."
    ]
    return random.choice(smart_responses)

# ---------- VOICE FUNCTION ----------
def speak(text):
    tts = gTTS(text)
    tts.save("voice.mp3")
    audio_file = open("voice.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

# ---------- UI ----------
st.title("🤖 Smart NLTK Chatbot")
st.write("Chat with AI | Voice | Download Chat")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Send"):
        if user_input:
            bot_response = chat.respond(user_input)

            # AI Smart Enhancement
            if not bot_response:
                bot_response = ai_reply(user_input)

            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("Bot", bot_response))

with col2:
    if st.button("🔊 Voice Reply"):
        if st.session_state.history:
            speak(st.session_state.history[-1][1])

# ---------- SHOW CHAT ----------
for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")

# ---------- DOWNLOAD CHAT ----------
chat_text = "\n".join([f"{s}: {m}" for s, m in st.session_state.history])

b64 = base64.b64encode(chat_text.encode()).decode()
href = f'<a href="data:file/txt;base64,{b64}" download="chat.txt">📥 Download Chat</a>'
st.markdown(href, unsafe_allow_html=True)

# ---------- LOGOUT ----------
if st.button("Logout"):
    st.session_state.logged_in = False
    st.experimental_rerun()
