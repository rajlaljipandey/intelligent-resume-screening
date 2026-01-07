from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(jd, resumes):
    documents = [jd] + resumes
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)
    return cosine_similarity(matrix[0:1], matrix[1:])[0]
