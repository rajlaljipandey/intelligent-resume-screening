<p align="center"> <img src="banner.png" alt="ResumeIQ Banner" width="100%"> </p>

ResumeIQ is a premium AI-powered resume screening web application that helps recruiters and hiring teams automatically analyze, rank, and recommend candidates based on job descriptions using Natural Language Processing (NLP) and Machine Learning.

Designed with a modern SaaS-style UI, ResumeIQ works seamlessly on desktop and mobile devices.

🚀 Key Features

📄 Resume Parsing – Supports PDF & DOCX formats

🧠 NLP Preprocessing – Tokenization, stopword removal, text cleaning

📊 TF-IDF Vectorization – Intelligent text representation

🔍 Cosine Similarity Matching – Accurate JD–Resume matching

🏆 Automatic Candidate Ranking

✅ Hiring Recommendations

Strongly Recommended

Consider

Not Recommended

📈 Interactive Visualizations

🎨 Premium, Responsive UI

📱 Mobile & Desktop Friendly

🧠 How ResumeIQ Works

Enter Job Description

Upload Multiple Resumes

Text Extraction & NLP Processing

Similarity Score Calculation

Candidate Ranking & Recommendation

Insights & Visual Analytics

🖥️ Screenshots

📌 Add screenshots of your app UI in a folder (e.g., screenshots/) and update paths below.

🔹 Job Description & Resume Upload
<p align="center"> <img src="screenshots/input.png" width="90%"> </p>
🔹 Candidate Ranking Dashboard
<p align="center"> <img src="screenshots/ranking.png" width="90%"> </p>
🔹 Match Percentage Visualization
<p align="center"> <img src="screenshots/chart.png" width="90%"> </p>
🛠️ Tech Stack

Frontend & App Framework: Streamlit

Programming Language: Python

NLP: NLTK

Machine Learning: Scikit-learn

Data Handling: Pandas

Resume Parsing: pdfplumber, python-docx

📂 Project Structure
intelligent-resume-screening/
│
├── .streamlit/
│   └── config.toml
├── test_resumes/
│   ├── rahul_resume.docx
│   ├── priya_resume.docx
│   └── amit_resume.docx
├── screenshots/
│   ├── input.png
│   ├── ranking.png
│   └── chart.png
├── app.py
├── banner.png
├── requirements.txt
└── README.md

▶️ Run the Project Locally
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

🌐 Deployment

This project is deployment-ready and can be hosted on:

Streamlit Cloud

AWS / Azure (via containerization)

Any cloud VM supporting Python

👨‍💻 Developer

Raj Lalji Pandey
📧 Email: rajlaljipandey@gmail.com

🌐 GitHub: https://github.com/rajlaljipandey

⭐ Support

If you find this project useful:

⭐ Star the repository

🍴 Fork it

💬 Share feedback

🏁 ResumeIQ — Making hiring smarter with AI