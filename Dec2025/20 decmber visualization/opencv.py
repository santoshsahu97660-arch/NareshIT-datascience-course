import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# -------------------- PAGE CONFIG (Dark Mode Friendly) --------------------
st.set_page_config(
    page_title="Image Processing App",
    page_icon="🎨",
    layout="wide"
)

# -------------------- FILTER FUNCTIONS --------------------
def apply_filter(img, filter_name, intensity):
    if filter_name == "Original":
        return img

    if filter_name == "Grayscale":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if filter_name == "Blur":
        k = max(1, int(intensity) * 2 + 1)
        return cv2.GaussianBlur(img, (k, k), 0)

    if filter_name == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5 + intensity, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    if filter_name == "Edges":
        return cv2.Canny(img, 50, 50 + intensity * 5)

    if filter_name == "Negative":
        return cv2.bitwise_not(img)

    if filter_name == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)

    if filter_name == "Warm":
        return cv2.add(img, np.array([intensity*2, intensity, 0]))

    if filter_name == "Cool":
        return cv2.add(img, np.array([0, intensity, intensity*2]))

    return img


# -------------------- MAIN APP --------------------
def main():
    st.title("🎨 Image Processing with OpenCV & Streamlit")
    st.caption("Created by Santosh")

    uploaded_file = st.file_uploader(
        "📤 Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # ---------------- SIDEBAR CONTROLS ----------------
        st.sidebar.header("🎛️ Controls")

        filter_name = st.sidebar.selectbox(
            "Select Filter",
            [
                "Original",
                "Grayscale",
                "Blur",
                "Sharpen",
                "Edges",
                "Negative",
                "Sepia",
                "Warm",
                "Cool"
            ]
        )

        intensity = st.sidebar.slider(
            "Filter Intensity",
            min_value=1,
            max_value=20,
            value=5
        )

        filtered_img = apply_filter(img, filter_name, intensity)

        # ---------------- BEFORE–AFTER COMPARISON ----------------
        st.subheader("🖼️ Before vs After")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(image)

        with col2:
            st.markdown(f"**Filtered Image – {filter_name}**")
            st.image(
                filtered_img,
                channels="GRAY" if len(filtered_img.shape) == 2 else "BGR"
            )

        # ---------------- DOWNLOAD IMAGE ----------------
        st.subheader("📥 Download Filtered Image")

        result_pil = Image.fromarray(
            filtered_img if len(filtered_img.shape) == 2
            else cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
        )

        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")

        st.download_button(
            label="⬇️ Download Image",
            data=buf.getvalue(),
            file_name="filtered_image.png",
            mime="image/png"
        )


if __name__ == "__main__":
    main()
