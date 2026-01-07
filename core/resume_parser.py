import pdfplumber
import docx

def extract_resume_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + " "
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = " ".join(p.text for p in doc.paragraphs)
    return text
