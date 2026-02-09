import streamlit as st
import pickle
import numpy as np
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Salary Prediction App", page_icon="💰")

# ===============================
# GET CURRENT FILE DIRECTORY
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "linear_regression_model.pkl")

# ===============================
# DEBUG (optional – can remove later)
# ===============================
# st.write("App running from:", BASE_DIR)
# st.write("Model exists:", os.path.exists(MODEL_PATH))

# ===============================
# LOAD MODEL SAFELY
# ===============================
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found in app folder")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ===============================
# UI
# ===============================
st.title("💰 Salary Prediction App")
st.write("Predict salary based on **Years of Experience**")

experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

if st.button("Predict Salary"):
    prediction = model.predict(np.array([[experience]]))
    st.success(f"💵 Predicted Salary: ₹ {int(prediction[0]):,}")

st.markdown("---")
st.caption("Built with Python • Machine Learning • Streamlit")
