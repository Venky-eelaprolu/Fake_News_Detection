import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

with open("Fake_News_Detection.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Fake News Detection")

article = st.text_input("Enter Article")

def clean_text(text):
    text = re.sub("[^A-Za-z\s]", " ", text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

if st.button("Predict"):
    if article.strip() == "":
        st.warning("Please type an article first.")
    else:
        cleaned = clean_text(article)
        result = model.predict([cleaned])[0]

        if result == 0:
            st.success("The article is classified as real/true")
        else:
            st.error("The article is classified as fake")