import streamlit as st
import pickle
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english') + stopwords.words('russian'))
regex = re.compile(r'[A-Za-zА-Яа-яёЁ]+')

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove links
    words = regex.findall(text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Modelni yuklash
with open('en_talks_vectorizer.pkl', 'rb') as f:
    en_vectorizer = pickle.load(f)

with open('ru_talks_vectorizer.pkl', 'rb') as f:
    ru_vectorizer = pickle.load(f)

# TED Talks ma'lumotlarini o'qing
en_talks = pd.read_csv('ted_talks_en.csv')  # English data
ru_talks = pd.read_csv('ted_talks_ru.csv')  # Russian data

# Vektorizatsiya qilish
en_talks_vectors = en_vectorizer.transform(en_talks['title'] + ' ' + en_talks['description'])
ru_talks_vectors = ru_vectorizer.transform(ru_talks['title'] + ' ' + ru_talks['description'])

# Tavsiyalar funktsiyasi
def recommend_talks(query, num_results=5):
    if detect(query) == 'ru':
        query = preprocess_text(query)
        query_vec = ru_vectorizer.transform([query])
        sims_ru = cosine_similarity(query_vec, ru_talks_vectors).flatten()
        top_indices_ru = sims_ru.argsort()[::-1][:num_results]
        recs_ru = ru_talks.iloc[top_indices_ru][["title", "topics", "description"]]
        return recs_ru
    else:
        query = preprocess_text(query)
        query_vec = en_vectorizer.transform([query])
        sims_en = cosine_similarity(query_vec, en_talks_vectors).flatten()
        top_indices_en = sims_en.argsort()[::-1][:num_results]
        recs_en = en_talks.iloc[top_indices_en][["title", "topics", "description"]]
        return recs_en

# Streamlit interfeysi
st.set_page_config(page_title="Tavsiyalar oynasi ", page_icon="💬", layout="wide")

# Interfeysning dizayni
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2D74D2;
        text-align: center;
    }
    .message-container {
        background-color: #FFFFFF; /* Sof oq fon */
        border: 1px solid #D3D3D3; /* Qiya kulrang chiziq */
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Yumshoq soya */
    }
    .user-message {
        background-color: #E6F4FF; /* Qoramtir ko‘k fon */
        color: #333333; /* Qoramtir matn */
        padding: 10px;
        border-radius: 5px;
        font-size: 16px;
    }
    .chatgpt-message {
        background-color: #EBF8EF; /* Qoramtir yashil fon */
        color: #333333; /* Qoramtir matn */
        padding: 10px;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Sarlavha
st.markdown('<div class="title">Ted tavsiyalar</div>', unsafe_allow_html=True)

# Foydalanuvchidan matn olish
user_input = st.text_input("Savolingizni kiriting:", key="user_input", placeholder="Savolingizni yozing...")

# Agar foydalanuvchi matn kiritgan bo'lsa
if user_input:
    # ChatGPT javobini olish va ko'rsatish
    recommendations = recommend_talks(user_input)
    st.write("Tavsiyalar:")
    st.dataframe(recommendations)
