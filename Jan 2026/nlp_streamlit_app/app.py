import streamlit as st
import re, nltk, zipfile, io
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from wordcloud import WordCloud

# ---------------- PAGE CONFIG ----------------
st.set_page_config("NLP Insight Studio", "🧠", layout="wide")

# ---------------- CSS (Premium Look) ----------------
st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);}
.card {background: rgba(255,255,255,0.12); padding:25px; border-radius:20px;}
.metric {text-align:center; color:white;}
.metric h2 {margin:0;}
.stButton>button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color:white; border-radius:25px; border:none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN SYSTEM ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_ui():
    st.markdown("<div class='card'><h2>🔐 Login</h2></div>", unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "santosh123":
            st.session_state.logged_in = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ---------------- LOGOUT ----------------
st.sidebar.success("Logged in as Admin")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.experimental_rerun()

# ---------------- NLTK ----------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"[^a-z\s]", "", text.lower())
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("🎛 Hyperparameters")

vec_size = st.sidebar.slider("Word2Vec Vector Size", 50, 300, 100)
window = st.sidebar.slider("Word2Vec Window", 2, 10, 5)
epochs = st.sidebar.slider("Word2Vec Epochs", 10, 100, 40)
max_features = st.sidebar.slider("TF-IDF Max Features", 50, 1000, 300)

# ---------------- SESSION ----------------
for k in ["tokens","bow","tfidf","w2v","sim"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ---------------- HERO ----------------
st.markdown("""
<div class="card">
<h1>🧠 NLP Insight Studio</h1>
<p>Text → Features → Semantics → Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------
text = st.text_area("✍️ Paste your paragraph", height=170)

if st.button("🚀 Generate Insights"):
    tokens = preprocess(text)
    st.session_state.tokens = tokens

    joined = " ".join(tokens)

    # BoW
    bow = CountVectorizer()
    st.session_state.bow = pd.DataFrame(
        bow.fit_transform([joined]).toarray(),
        columns=bow.get_feature_names_out()
    )

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features)
    st.session_state.tfidf = pd.DataFrame(
        tfidf.fit_transform([joined]).toarray(),
        columns=tfidf.get_feature_names_out()
    )

    # Word2Vec
    st.session_state.w2v = Word2Vec(
        sentences=[tokens],
        vector_size=vec_size,
        window=window,
        min_count=1,
        epochs=epochs
    )

    st.success("Insights Generated 🔥")

# ---------------- METRICS ----------------
if st.session_state.tokens:
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='metric'><h2>{len(st.session_state.tokens)}</h2><p>Tokens</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><h2>{len(set(st.session_state.tokens))}</h2><p>Vocabulary</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><h2>{vec_size}</h2><p>Embedding Size</p></div>", unsafe_allow_html=True)

# ---------------- TABS ----------------
if st.session_state.tokens:
    t1,t2,t3,t4 = st.tabs(["📊 BoW","📈 TF-IDF","☁️ WordCloud","🔗 Word2Vec"])

    with t1:
        st.dataframe(st.session_state.bow)
    with t2:
        st.dataframe(st.session_state.tfidf)
    with t3:
        wc = WordCloud(width=900,height=400,background_color="black").generate(" ".join(st.session_state.tokens))
        fig,ax = plt.subplots(figsize=(10,4))
        ax.imshow(wc); ax.axis("off")
        st.pyplot(fig)
    with t4:
        word = st.text_input("Find similar word")
        if word and word in st.session_state.w2v.wv:
            st.session_state.sim = pd.DataFrame(
                st.session_state.w2v.wv.most_similar(word),
                columns=["Word","Score"]
            )
            st.dataframe(st.session_state.sim)

# ---------------- ZIP DOWNLOAD ----------------
if st.session_state.tokens:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr("tokens.csv", pd.DataFrame({"tokens":st.session_state.tokens}).to_csv(index=False))
        z.writestr("bow.csv", st.session_state.bow.to_csv(index=False))
        z.writestr("tfidf.csv", st.session_state.tfidf.to_csv(index=False))
        if st.session_state.sim is not None:
            z.writestr("word2vec_similarity.csv", st.session_state.sim.to_csv(index=False))

    st.download_button(
        "📦 Download ALL Results (ZIP)",
        zip_buffer.getvalue(),
        "nlp_outputs.zip",
        "application/zip"
    )

# ---------------- FOOTER ----------------
st.markdown("<center style='opacity:0.6'>🚀 NLP Product-style Application</center>", unsafe_allow_html=True)
