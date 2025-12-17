# AI-Based Resume Screening System

## Overview
This project uses Natural Language Processing (NLP) techniques to automatically match resumes with job descriptions.

## Problem Statement
Manual resume screening is time-consuming and subjective.

## Approach
- Text preprocessing
- TF-IDF vectorization
- Cosine similarity

## Technologies Used
Python, scikit-learn, NLTK

## Results
The system ranks resumes based on relevance to job descriptions.

## Future Improvements
- Use semantic embeddings
- Bias reduction techniques

## Code
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample resume data
resumes = [
    "Python developer with experience in machine learning and NLP",
    "Java developer with Spring Boot and microservices experience",
    "Data analyst skilled in SQL, Python, and data visualization"
]

# Sample job description
job_description = "Looking for a Python developer with experience in machine learning"

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
resume_vectors = vectorizer.fit_transform(resumes)
job_vector = vectorizer.transform([job_description])

# Calculate similarity scores
similarity_scores = cosine_similarity(job_vector, resume_vectors)

# Rank resumes
ranked_resumes = sorted(
    list(enumerate(similarity_scores[0])),
    key=lambda x: x[1],
    reverse=True
)

# Display results
print("Resume Ranking Based on Job Description:\n")
for index, score in ranked_resumes:
    print(f"Resume {index + 1} - Similarity Score: {score:.2f}")
