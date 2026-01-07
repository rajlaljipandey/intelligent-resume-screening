import streamlit as st
import pdfplumber
import docx
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="ResumeIQ – AI Resume Screening",
    page_icon="📊",
    layout="wide"
)

# ================= PREMIUM UI STYLES =================
st.markdown("""
<style>
/* Background */
.main {
    background: radial-gradient(circle at top, #020617, #020617 60%);
}

/* Center content */
.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
}

/* HERO */
.hero {
    text-align: center;
    padding: 3rem 1rem 2.5rem 1rem;
}
.hero h1 {
    font-size: 2.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 720px;
    margin: auto;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px;
    padding: 1.8rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.45);
}

/* Section title */
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #38bdf8);
    color: white;
    font-weight: 700;
    border-radius: 14px;
    height: 3.3rem;
    border: none;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(99,102,241,0.45);
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 1rem;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.1);
    color: #94a3b8;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ================= CACHE STOPWORDS =================
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))

STOP_WORDS = load_stopwords()

# ================= HERO SECTION =================
st.markdown("""
<div class="hero">
    <h1>ResumeIQ</h1>
    <p>
        AI-powered resume screening platform that intelligently analyzes,
        ranks, and recommends candidates using NLP & Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ================= RESUME PARSING =================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def extract_resume_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    return ""

# ================= NLP PREPROCESSING =================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)

@st.cache_data(show_spinner=False)
def get_cleaned_resume_text(file):
    return preprocess_text(extract_resume_text(file))

# ================= TF-IDF =================
def calculate_similarity(job_desc, resume_texts):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([job_desc] + resume_texts)
    return cosine_similarity(tfidf[0:1], tfidf[1:])[0]

def get_recommendation(score):
    if score >= 60:
        return "Strongly Recommended"
    elif score >= 40:
        return "Consider"
    else:
        return "Not Recommended"

# ================= INPUT CARD =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📥 Get Started</div>', unsafe_allow_html=True)

job_description = st.text_area(
    "📝 Job Description",
    height=200,
    placeholder="Paste the job description here..."
)

uploaded_files = st.file_uploader(
    "📂 Upload Resumes (PDF / DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

analyze = st.button("✨ Analyze & Rank Candidates", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ================= MAIN LOGIC =================
if analyze:
    if not job_description:
        st.warning("⚠️ Please enter a Job Description.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    else:
        with st.spinner("Analyzing candidates using AI..."):

            jd_clean = preprocess_text(job_description)

            names, texts = [], []
            for f in uploaded_files:
                names.append(f.name)
                texts.append(get_cleaned_resume_text(f))

            scores = calculate_similarity(jd_clean, texts)

            df = pd.DataFrame({
                "Candidate": names,
                "Match Percentage": (scores * 100).round(2),
            })
            df["Recommendation"] = df["Match Percentage"].apply(get_recommendation)
            df = df.sort_values("Match Percentage", ascending=False).reset_index(drop=True)
            df.insert(0, "Rank", df.index + 1)

            # ================= SUMMARY =================
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Insights Overview</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("👥 Candidates", len(df))
            c2.metric("🏆 Top Match (%)", df["Match Percentage"].max())
            st.markdown('</div>', unsafe_allow_html=True)

            # ================= BEST =================
            best = df.iloc[0]
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success(
                f"🏆 Best Candidate: **{best['Candidate']}** "
                f"({best['Match Percentage']}%) — {best['Recommendation']}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # ================= TABLE =================
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 Candidate Ranking</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ================= CHART =================
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Match Comparison</div>', unsafe_allow_html=True)
            st.bar_chart(df.set_index("Candidate")["Match Percentage"])
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("🔍 View Preprocessed Resume Text"):
                for n, t in zip(names, texts):
                    st.markdown(f"**{n}**")
                    st.write(t[:700])

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    © 2026 ResumeIQ · AI Resume Intelligence Platform<br>
    Developed by <strong>Raj Lalji Pandey</strong>
</div>
""", unsafe_allow_html=True)
