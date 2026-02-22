import streamlit as st
from PIL import Image

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Santosh Sahu Portfolio",
    page_icon="📊",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Santosh Sahu")
st.sidebar.write("Data Analyst | Python Developer")

st.sidebar.markdown("---")
st.sidebar.subheader("Contact")
st.sidebar.write("📍 Hyderabad, India")
st.sidebar.write("📧 santoshsk14320@gmail.com")
st.sidebar.write("📱 +91-9766064261")

st.sidebar.markdown("---")
st.sidebar.subheader("Links")
st.sidebar.write("🔗 LinkedIn")
st.sidebar.write("https://linkedin.com/in/santosh-sahu")
st.sidebar.write("💻 GitHub")
st.sidebar.write("https://github.com/santoshsahu97660-arch")

# ---------------- HEADER ----------------
st.title("👋 Hello, I'm Santosh Sahu")
st.subheader("Aspiring Data Analyst & Machine Learning Enthusiast")

st.write("""
I am a BCA graduate (2025) passionate about Data Analytics, Machine Learning and Computer Vision.
I enjoy solving real-world problems using Python, Data Analysis and Visualization.
""")

st.markdown("---")

# ---------------- SKILLS ----------------
st.header("🧠 Technical Skills")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Programming")
    st.progress(90)
    st.write("Python")
    st.progress(75)
    st.write("SQL")

with col2:
    st.subheader("Libraries")
    st.write("• Pandas")
    st.write("• NumPy")
    st.write("• Matplotlib")
    st.write("• Seaborn")
    st.write("• Scikit-learn")
    st.write("• OpenCV")

st.markdown("---")

# ---------------- PROJECTS ----------------
st.header("🚀 Projects")

# Project 1
with st.container():
    st.subheader("🚗 Real-Time Vehicle Detection & Tracking")
    st.write("""
    • Captured video using cv2.VideoCapture  
    • Converted frames to grayscale using cv2.cvtColor  
    • Applied Gaussian Blur and Thresholding  
    • Detected moving vehicles using contour detection  
    • Drew bounding boxes and tracked objects  
    **Technologies:** Python, OpenCV, NumPy
    """)
    st.button("View GitHub - Vehicle Detection", key="p1")

# Project 2
with st.container():
    st.subheader("📊 Machine Learning Classification Model")
    st.write("""
    • Data cleaning using Pandas  
    • Feature scaling using StandardScaler  
    • Used train_test_split  
    • Algorithms: Logistic Regression, KNN, Random Forest  
    • Evaluation using confusion matrix & accuracy  
    **Technologies:** Python, Scikit-learn, Pandas
    """)
    st.button("View GitHub - ML Project", key="p2")

# Project 3
with st.container():
    st.subheader("📈 Sales Data Analysis & Visualization")
    st.write("""
    • Performed Exploratory Data Analysis  
    • Used groupby and aggregation  
    • Created charts & heatmaps  
    • Extracted business insights  
    **Technologies:** Pandas, Matplotlib, Seaborn
    """)
    st.button("View GitHub - Data Analysis", key="p3")

st.markdown("---")

# ---------------- EXPERIENCE ----------------
st.header("💼 Experience")
st.write("""
**Computer Instructor – Moharsh Computer Education (2020–2022)**  
• Taught 100+ students computer fundamentals  
• Maintained reports in Excel  
• Improved communication and presentation skills
""")

st.markdown("---")

# ---------------- EDUCATION ----------------
st.header("🎓 Education")
st.write("Bachelor of Computer Applications (BCA) — Gondwana University — 2025")

st.markdown("---")

# ---------------- RESUME DOWNLOAD ----------------
st.header("📄 Download Resume")

with open("resume.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(
    label="📥 Download My Resume",
    data=PDFbyte,
    file_name="santosh sahu resume.pdf",
    mime="application/octet-stream"
)

st.markdown("---")
st.success("Thank you for visiting my portfolio!")