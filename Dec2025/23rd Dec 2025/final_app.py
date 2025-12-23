import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Butterfly Image Processor",
    layout="wide"
)

st.title("🦋 Butterfly Image – Multi-Color Channel Visualizer")

# ---------------- Load Image from URL ----------------
@st.cache_data(show_spinner=False)
def load_image_from_url():
    url = "https://upload.wikimedia.org/wikipedia/commons/3/34/Monarch_butterfly_insect_danaus_plexippus.jpg"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            return Image.open(BytesIO(response.content)).convert("RGB")
    except:
        pass

    return None

# ---------------- Sidebar ----------------
st.sidebar.header("Image Source")

option = st.sidebar.radio(
    "Choose Image Input",
    ["Load from URL (Default)", "Upload Your Own Image"]
)

butterfly = None

# Option 1: URL
if option == "Load from URL (Default)":
    butterfly = load_image_from_url()

    if butterfly is None:
        st.warning("URL image failed. Using local fallback image.")
        try:
            butterfly = Image.open("butterfly.jpg").convert("RGB")
        except:
            st.error("No local image found. Please upload an image.")
            st.stop()

# Option 2: Upload
else:
    uploaded = st.sidebar.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded is not None:
        butterfly = Image.open(uploaded).convert("RGB")
    else:
        st.info("Please upload an image to continue.")
        st.stop()

# ---------------- Show Original Image ----------------
st.subheader("Original Image")
st.image(butterfly, use_container_width=True)

# ---------------- Convert to NumPy ----------------
butterfly_np = np.array(butterfly)

R = butterfly_np[:, :, 0]
G = butterfly_np[:, :, 1]
B = butterfly_np[:, :, 2]

# ---------------- RGB Channel Images ----------------
red_img = np.zeros_like(butterfly_np)
green_img = np.zeros_like(butterfly_np)
blue_img = np.zeros_like(butterfly_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

st.subheader("🎨 RGB Channel Visualization")
c1, c2, c3 = st.columns(3)

with c1:
    st.image(red_img, caption="Red Channel", use_container_width=True)

with c2:
    st.image(green_img, caption="Green Channel", use_container_width=True)

with c3:
    st.image(blue_img, caption="Blue Channel", use_container_width=True)

# ---------------- Grayscale + Colormap ----------------
st.subheader("🌈 Grayscale with Colormap")

colormap = st.selectbox(
    "Choose Colormap",
    ["gray", "viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool"]
)

gray = butterfly.convert("L")
gray_np = np.array(gray)

fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.imshow(gray_np, cmap=colormap)
ax1.axis("off")
st.pyplot(fig1)

# ---------------- RGB Histogram ----------------
st.subheader("📊 RGB Histogram")

fig2, ax2 = plt.subplots()
ax2.hist(R.flatten(), bins=256, alpha=0.5, label="Red")
ax2.hist(G.flatten(), bins=256, alpha=0.5, label="Green")
ax2.hist(B.flatten(), bins=256, alpha=0.5, label="Blue")
ax2.legend()
ax2.set_title("RGB Intensity Distribution")
st.pyplot(fig2)

# ---------------- Download Processed Image ----------------
st.subheader("📥 Download Grayscale Image")

buf = BytesIO()
gray.save(buf, format="PNG")
byte_im = buf.getvalue()

st.download_button(
    label="Download Grayscale Image",
    data=byte_im,
    file_name="butterfly_grayscale.png",
    mime="image/png"
)

st.success("✅ App running successfully!")
