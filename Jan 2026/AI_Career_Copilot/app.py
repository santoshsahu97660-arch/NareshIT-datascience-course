import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Career Copilot",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (AI / DARK UI)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background: #161b22;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 0 25px rgba(0,0,0,0.35);
}
.metric {
    font-size: 30px;
    font-weight: bold;
    color: #58a6ff;
}
.sub {
    color: #8b949e;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
roles_df = pd.read_csv("sample_roles.csv")

# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------
def extract_text_from_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

def recommend_role(resume_text):
    corpus = roles_df["skills"].tolist() + [resume_text]
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    idx = similarity.argmax()
    return roles_df.iloc[idx]

def ats_score(resume_text, skills):
    matched = sum(skill in resume_text for skill in skills)
    return int((matched / len(skills)) * 100)

def skill_gap(resume_text, skills):
    have = [s for s in skills if s in resume_text]
    missing = [s for s in skills if s not in resume_text]
    return have, missing

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>🧠 AI Career Copilot</h1>
<p style='text-align:center;color:gray;'>
Resume Analyzer • ATS Score • Career Recommendation • Salary • Roadmap
</p>
<p style='text-align:center;'>By <b>Santosh Sahu</b></p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("📄 Upload Resume")
uploaded_file = st.sidebar.file_uploader(
    "Upload your resume (PDF)", type=["pdf"]
)

# -------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------
if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)

    role = recommend_role(resume_text)
    skills = [s.strip() for s in role["skills"].split(",")]
    score = ats_score(resume_text, skills)
    have, missing = skill_gap(resume_text, skills)

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="sub">ATS SCORE</div>
            <div class="metric">{score} / 100</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="sub">BEST ROLE</div>
            <div class="metric">{role['role']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="sub">MARKET DEMAND</div>
            <div class="metric">{role['demand']}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------
    # SKILL GAP
    # -------------------------------------------------
    st.markdown("<div class='card'><h3>🧠 Skill Gap Analysis</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.success("✅ Skills You Have")
        for s in have:
            st.write("✔", s)

    with c2:
        st.error("❌ Skills To Learn")
        for s in missing:
            st.write("✖", s)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # SKILL GAP CHART
    # -------------------------------------------------
    st.markdown("<div class='card'><h3>📊 Skill Distribution</h3>", unsafe_allow_html=True)

    skill_df = pd.DataFrame({
        "Skill": skills,
        "Status": ["Have" if s in have else "Missing" for s in skills]
    })

    fig, ax = plt.subplots()
    skill_df["Status"].value_counts().plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Skills Overview")
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # SALARY INSIGHT
    # -------------------------------------------------
    st.markdown(f"""
    <div class="card">
        <h3>💰 Salary Insights</h3>
        <p style="font-size:22px;">
        ₹ {role['salary_min']} LPA – ₹ {role['salary_max']} LPA
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------
    # ROADMAP
    # -------------------------------------------------
    st.markdown("<div class='card'><h3>🛣️ Career Roadmap</h3>", unsafe_allow_html=True)

    if missing:
        for i, skill in enumerate(missing[:5], 1):
            st.write(f"Step {i} → Learn **{skill.upper()}**")
    else:
        st.success("🎉 Your skills already match this role very well!")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👈 Upload your resume PDF from the sidebar to start analysis")
