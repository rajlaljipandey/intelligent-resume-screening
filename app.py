import streamlit as st
import pandas as pd

# ===== Core Imports (NEW MODULAR STRUCTURE) =====
from core.resume_parser import extract_resume_text
from core.preprocessing import preprocess_text
from core.matching import calculate_similarity

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="ResumeIQ – AI Resume Screening",
    page_icon="📊",
    layout="wide"
)

# ================= PREMIUM UI STYLES =================
st.markdown("""
<style>
.main {
    background: radial-gradient(circle at top, #020617, #020617 60%);
}
.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
}
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
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px;
    padding: 1.8rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.45);
}
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
}
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #38bdf8);
    color: white;
    font-weight: 700;
    border-radius: 14px;
    height: 3.3rem;
    border: none;
}
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 1rem;
}
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

# ================= HERO =================
st.markdown("""
<div class="hero">
    <h1>ResumeIQ</h1>
    <p>
        AI-powered resume screening platform that intelligently analyzes,
        ranks, and recommends candidates using NLP & Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

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
            for file in uploaded_files:
                names.append(file.name)
                raw_text = extract_resume_text(file)
                texts.append(preprocess_text(raw_text))

            scores = calculate_similarity(jd_clean, texts)

            df = pd.DataFrame({
                "Candidate": names,
                "Match Percentage": (scores * 100).round(2)
            })

            df["Recommendation"] = df["Match Percentage"].apply(
                lambda x: "Strongly Recommended" if x >= 60 else
                          "Consider" if x >= 40 else
                          "Not Recommended"
            )

            df = df.sort_values("Match Percentage", ascending=False).reset_index(drop=True)
            df.insert(0, "Rank", df.index + 1)

            # ================= SUMMARY =================
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Insights Overview</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("👥 Candidates", len(df))
            c2.metric("🏆 Top Match (%)", df["Match Percentage"].max())
            st.markdown('</div>', unsafe_allow_html=True)

            # ================= BEST CANDIDATE =================
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
