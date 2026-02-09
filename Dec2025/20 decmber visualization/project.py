import streamlit as st
import numpy as np
import cv2
from PIL import Image

def main():
    st.title("Image Processing with OpenCV and Streamlit created by santosh")
   
    # Upload image
    uploaded_file = st.file_uploader(r"C:\Users\santo\OneDrive\Desktop\Data science\Dec2025\Workshop Gemini 3\butterfly.jpg", type=["jpg", "jpeg", "png"])
   
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert to numpy array
        img_array = np.array(image)

        # Convert to grayscale using OpenCV
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Display grayscale image
        st.image(gray_image, caption='Grayscale Image', use_column_width=True, channels="GRAY")

if __name__ == "__main__":
    main()