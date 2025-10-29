import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download('stopwords')

st.set_page_config(page_title="📰 Fake News Detection using NLP", layout="centered")

st.title("📰 Fake News Detection using NLP")
st.markdown("### Detect whether a news article is **Real or Fake** using Machine Learning and NLP")

DATASET_PATH = "Fake.csv"
REAL_PATH = "True.csv"

if not os.path.exists(DATASET_PATH) or not os.path.exists(REAL_PATH):
    st.warning("⚠️ Please download the dataset from Kaggle:\n\n"
               "👉 [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)\n\n"
               "and place the `Fake.csv` and `True.csv` files in the same directory as this script.")
    st.stop()

fake_df = pd.read_csv(DATASET_PATH)
true_df = pd.read_csv(REAL_PATH)

fake_df["label"] = 0
true_df["label"] = 1

data = pd.concat([fake_df, true_df], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

st.success(f"✅ Dataset loaded successfully! Total Records: {len(data)}")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

st.info("🧹 Cleaning and preprocessing text...")
data["clean_text"] = data["text"].apply(clean_text)

st.info("🔡 Converting text to numerical features (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data["clean_text"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.info("🤖 Training Naive Bayes classifier...")
model = MultinomialNB()
model.fit(X_train, y_train)

joblib.dump(model, "fake_news_model.pkl", protocol=4)
joblib.dump(tfidf, "tfidf_vectorizer.pkl", protocol=4)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.subheader("📊 Model Evaluation Results")
st.write(f"**Accuracy:** {accuracy*100:.2f}%")
st.write(f"**Precision:** {precision*100:.2f}%")
st.write(f"**Recall:** {recall*100:.2f}%")
st.write(f"**F1 Score:** {f1*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

st.markdown("---")
st.subheader("🧠 Try Your Own News Article")
user_input = st.text_area("Enter a news headline or article text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("✅ The news article is **REAL** 🗞️")
        else:
            st.error("🚨 The news article is **FAKE** 📰")

st.markdown("---")
st.markdown("👨‍💻 **Developed for Internship Task 2 – Fake News Detection using NLP (Growfinix)**")
