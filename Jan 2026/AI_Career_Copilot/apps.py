import streamlit as st
import pandas as pd
import uuid
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="AI Career Copilot X",
    page_icon="🚀",
    layout="wide"
)

# =================================================
# PREMIUM UI CSS
# =================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

body {
    background: radial-gradient(circle at top, #020617, #000000);
}

.hero h1 {
    font-size:72px;
    font-weight:900;
    background: linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    border-radius:24px;
    padding:28px;
    margin-bottom:26px;
    box-shadow: 0 25px 45px rgba(0,0,0,0.6);
}

.big {
    font-size:46px;
    font-weight:900;
    color:#22d3ee;
}

.label {
    color:#94a3b8;
    font-size:13px;
    letter-spacing:1px;
}

.warn {
    color:#f87171;
    font-weight:600;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD DATA
# =================================================
roles_df = pd.read_csv("sample_roles.csv")

countries = {
    "🇮🇳 India": ("₹", 1),
    "🇺🇸 USA": ("$", 12),
    "🇬🇧 UK": ("£", 12),
    "🇨🇦 Canada": ("C$", 12),
    "🇩🇪 Germany": ("€", 12),
    "🇦🇪 UAE": ("AED", 12),
    "🇦🇺 Australia": ("A$", 12),
    "🇸🇬 Singapore": ("SGD", 12),
}

# =================================================
# FUNCTIONS
# =================================================
def extract_text(pdf):
    reader = PdfReader(pdf)
    return " ".join([p.extract_text() or "" for p in reader.pages]).lower()

def recommend_role(resume_text):
    corpus = roles_df["skills"].tolist() + [resume_text]
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    idx = similarity.argmax()
    return roles_df.iloc[idx], int(similarity.max() * 100)

# =================================================
# HERO SECTION
# =================================================
st.markdown("""
<div class="hero" style="text-align:center;">
    <h1>AI Career Copilot X</h1>
    <p style="color:#cbd5e1;font-size:20px;">
        AI that predicts your career, salary & future demand
    </p>
    <p>By <b>Santosh Sahu</b></p>
</div>
""", unsafe_allow_html=True)

# =================================================
# SIDEBAR
# =================================================
st.sidebar.markdown("## 🚀 Start Here")
resume = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
country = st.sidebar.selectbox("🌍 Target Country", list(countries.keys()))

# =================================================
# MAIN LOGIC
# =================================================
if resume:
    resume_text = extract_text(resume)
    role, match = recommend_role(resume_text)
    skills = [s.strip() for s in role.skills.split(",")]

    currency, factor = countries[country]
    min_sal = role.salary_min * factor
    max_sal = role.salary_max * factor

    # ---------------------------------------------
    # SHAREABLE LINK
    # ---------------------------------------------
    public_id = str(uuid.uuid4())[:8]
    st.success(f"🔗 Share your AI Result:")
    st.code(f"https://ai-career-copilot.streamlit.app/?id={public_id}")

    # ---------------------------------------------
    # METRICS
    # ---------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='card'><div class='label'>CAREER MATCH</div><div class='big'>{match}%</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='label'>BEST ROLE</div><div class='big'>{role.role}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='label'>COUNTRY</div><div class='big'>{country}</div></div>", unsafe_allow_html=True)

    # ---------------------------------------------
    # CAREER RISK INDICATOR
    # ---------------------------------------------
    if match < 50:
        risk = "🔴 HIGH RISK"
        message = "🚨 Your resume is weak for this role. Immediate upskilling required."
    elif match < 75:
        risk = "🟡 MEDIUM RISK"
        message = "⚡ Your profile is decent but needs improvement to be job-ready."
    else:
        risk = "🟢 SAFE"
        message = "🔥 Your resume is strong and market-ready."

    st.markdown(f"<div class='card warn'>Career Risk: {risk}</div>", unsafe_allow_html=True)
    st.info(message)

    # ---------------------------------------------
    # SALARY SECTION
    # ---------------------------------------------
    st.markdown(f"""
    <div class='card'>
        <div class='label'>EXPECTED SALARY</div>
        <div class='big'>{currency} {min_sal} – {max_sal}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------
    # COUNTRY COMPARISON
    # ---------------------------------------------
    st.markdown("<div class='card'><div class='label'>🌍 GLOBAL SALARY COMPARISON</div>", unsafe_allow_html=True)
    for c, (sym, f) in countries.items():
        st.write(f"{c}: {sym} {role.salary_min*f} – {role.salary_max*f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------
    # SKILL HEATMAP
    # ---------------------------------------------
    st.markdown("<div class='card'><div class='label'>SKILL HEATMAP</div>", unsafe_allow_html=True)
    for s in skills:
        if s in resume_text:
            st.success(f"✔ {s}")
        else:
            st.error(f"✖ {s}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------
    # ROADMAP
    # ---------------------------------------------
    st.markdown("<div class='card'><div class='label'>90-DAY CAREER ROADMAP</div>", unsafe_allow_html=True)
    missing = [s for s in skills if s not in resume_text]
    for i, s in enumerate(missing[:5], 1):
        st.markdown(f"Week {i*2} → Learn **{s.upper()}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------
    # DOWNLOAD REPORT
    # ---------------------------------------------
    report = f"""
AI CAREER REPORT
------------------------
Career Match: {match}%
Best Role: {role.role}
Country: {country}
Salary: {currency} {min_sal} – {max_sal}

Career Risk: {risk}

Missing Skills:
{', '.join(missing)}
"""
    st.download_button(
        "📄 Download AI Career Report",
        report,
        file_name="AI_Career_Report.txt"
    )

    # ---------------------------------------------
    # SCORE BADGE
    # ---------------------------------------------
    st.markdown(
        f"<div class='card'><h2>🏆 AI Career Score: {match}%</h2></div>",
        unsafe_allow_html=True
    )

else:
    st.info("⬅ Upload resume & select country to start")
