import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
import os
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="📄 Resume Screening Tool", layout="centered")
st.title("📄 Resume Screening Tool (Task 3)")
st.markdown("### Filter and rank resumes based on required skills using NLP")

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

try:
    df = pd.read_csv("resume.csv")
    st.info(f"✅ Dataset loaded successfully with {len(df)} records.")
except FileNotFoundError:
    st.error("❌ resume.csv not found. Please place it in this folder.")
    st.stop()

text_col = "Resume_str" if "Resume_str" in df.columns else "Resume"
MODEL_FILE = "resume_screening_model.pkl"
VECT_FILE = "resume_tfidf.pkl"

def train_and_save_model():
    st.warning("⚙️ Training new model... Please wait...")
    df["cleaned_resume"] = df[text_col].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["cleaned_resume"])
    y = df["Category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"🎯 Model Trained Successfully — Accuracy: {acc * 100:.2f}%")
    st.text("📊 Classification Report:\n" + classification_report(y_test, y_pred))
    joblib.dump(model, MODEL_FILE, protocol=4)
    joblib.dump(tfidf, VECT_FILE, protocol=4)
    st.success("💾 Model and vectorizer saved successfully!")
    return model, tfidf

try:
    model = joblib.load(MODEL_FILE)
    tfidf = joblib.load(VECT_FILE)
    st.success("✅ Loaded saved model and vectorizer successfully.")
except Exception as e:
    st.warning(f"⚠️ Could not load existing model: {e}")
    model, tfidf = train_and_save_model()

st.markdown("---")
st.subheader("🧠 Enter Required Skills or Keywords")
skills = st.text_input("Enter skills (comma-separated):", "Python, Machine Learning, SQL")

if st.button("🔍 Find Matching Resumes"):
    skills_clean = clean_text(skills)
    skill_vec = tfidf.transform([skills_clean])
    pred_prob = model.predict_proba(skill_vec)
    top_cat = model.classes_[pred_prob.argmax()]
    st.success(f"🏆 Most Relevant Category: **{top_cat}**")
    st.markdown("---")
    st.subheader("📋 Top Matching Resumes:")
    df["cleaned_resume"] = df[text_col].apply(clean_text)
    top_resumes = df[df["Category"] == top_cat].head(5)
    for i, row in top_resumes.iterrows():
        st.write(f"**Candidate {i+1}:** {row['Category']}")
        st.write(row[text_col][:400] + "...")
        st.markdown("---")

st.markdown("---")
st.markdown("👨‍💻 **Developed for Internship Task 3 – Resume Screening Tool (Growfinix)**")
