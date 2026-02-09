import streamlit as st
import numpy as np
import time
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

from tensorflow.keras.applications import (
    MobileNetV2, VGG16, VGG19,
    ResNet50, ResNet50V2,
    InceptionV3, Xception,
    DenseNet121, EfficientNetV2B0
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_pre, decode_predictions as mobilenet_dec
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_pre, decode_predictions as vgg16_dec
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_pre, decode_predictions as vgg19_dec
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre, decode_predictions as resnet_dec
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_pre, decode_predictions as resnetv2_dec
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_pre, decode_predictions as inception_dec
from tensorflow.keras.applications.xception import preprocess_input as xception_pre, decode_predictions as xception_dec
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre, decode_predictions as densenet_dec
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_pre, decode_predictions as effnet_dec

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Advanced CNN Image Classifier", layout="wide")
st.title("🧠 Advanced Image Classification Dashboard")

# --------------------------------------------------
# DEVICE SELECTOR (CPU / GPU)
# --------------------------------------------------
device = st.sidebar.radio("🖥️ Select Device", ["Auto", "CPU", "GPU"])

if device == "CPU":
    tf.config.set_visible_devices([], "GPU")
elif device == "GPU":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_visible_devices(gpus[0], "GPU")

# --------------------------------------------------
# MODEL CONFIG
# --------------------------------------------------
MODELS = {
    "MobileNetV2": (MobileNetV2, mobilenet_pre, mobilenet_dec, (224, 224), 71),
    "VGG16": (VGG16, vgg16_pre, vgg16_dec, (224, 224), 71),
    "VGG19": (VGG19, vgg19_pre, vgg19_dec, (224, 224), 72),
    "ResNet50": (ResNet50, resnet_pre, resnet_dec, (224, 224), 76),
    "ResNet50V2": (ResNet50V2, resnetv2_pre, resnetv2_dec, (224, 224), 76),
    "InceptionV3": (InceptionV3, inception_pre, inception_dec, (299, 299), 78),
    "Xception": (Xception, xception_pre, xception_dec, (299, 299), 79),
    "DenseNet121": (DenseNet121, densenet_pre, densenet_dec, (224, 224), 75),
    "EfficientNetV2B0": (EfficientNetV2B0, effnet_pre, effnet_dec, (224, 224), 78),
}

# --------------------------------------------------
# SAFE CACHED MODEL LOADER
# --------------------------------------------------
@st.cache_resource
def load_imagenet_model(_model_class):
    return _model_class(weights="imagenet")

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
model_name = st.sidebar.selectbox("Choose Pretrained Model", list(MODELS.keys()))
top_k = st.sidebar.slider("Top-K Predictions", 1, 10, 5)

use_custom = st.sidebar.checkbox("🧠 Use Custom Trained Model")

# --------------------------------------------------
# CUSTOM MODEL UPLOAD
# --------------------------------------------------
custom_model = None
custom_input_size = (224, 224)

if use_custom:
    uploaded_model = st.sidebar.file_uploader(
        "Upload Custom Model (.h5 / .keras)",
        type=["h5", "keras"]
    )
    if uploaded_model:
        custom_model = load_model(uploaded_model)
        st.sidebar.success("Custom model loaded")

# --------------------------------------------------
# LOAD SELECTED MODEL
# --------------------------------------------------
if not use_custom:
    model_class, preprocess_fn, decode_fn, img_size, accuracy = MODELS[model_name]
    model = load_imagenet_model(model_class)
else:
    model = custom_model
    preprocess_fn = lambda x: x / 255.0
    decode_fn = None
    img_size = custom_input_size
    accuracy = None

# --------------------------------------------------
# IMAGE UPLOAD (BATCH)
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

speed_data = []

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
if uploaded_files and model:
    for file in uploaded_files:
        st.markdown("---")
        st.subheader(f"📷 {file.name}")

        img = Image.open(file).convert("RGB")
        st.image(img, width=300)

        img_resized = img.resize(img_size)
        img_array = keras_image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_fn(img_array)

        start = time.time()
        preds = model.predict(img_array)
        end = time.time()

        inference_ms = (end - start) * 1000

        speed_data.append({
            "Model": model_name if not use_custom else "Custom",
            "Inference Time (ms)": inference_ms,
            "Accuracy (%)": accuracy
        })

        st.info(f"⏱ Inference Time: {inference_ms:.2f} ms")

        if decode_fn:
            decoded = decode_fn(preds, top=top_k)[0]
            labels = [l for _, l, _ in decoded]
            scores = [s for _, _, s in decoded]

            df = pd.DataFrame({"Confidence": scores}, index=labels)
            st.bar_chart(df)

            for i, (_, label, score) in enumerate(decoded, 1):
                st.write(f"{i}. **{label}** — {score:.2f}")
        else:
            st.write("Custom model output:", preds)

# --------------------------------------------------
# SPEED vs ACCURACY COMPARISON
# --------------------------------------------------
if speed_data:
    st.markdown("## 📈 Accuracy vs Speed Comparison")

    comp_df = pd.DataFrame(speed_data)
    st.dataframe(comp_df)

    if not use_custom:
        st.scatter_chart(
            comp_df,
            x="Inference Time (ms)",
            y="Accuracy (%)"
        )

# --------------------------------------------------
# MODEL DETAILS
# --------------------------------------------------
if model:
    st.markdown("## 📦 Model Details")
    st.write(f"**Parameters:** {model.count_params():,}")
    st.write(f"**Depth:** {len(model.layers)}")
    st.write(f"**Running on:** {device}")
