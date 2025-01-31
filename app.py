# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv('/content/telkomsel.csv')
    df_tweet = pd.DataFrame(df[['full_text']])
    df_tweet['full_text'] = df_tweet['full_text'].str.lower()
    df_tweet.drop_duplicates(inplace=True)
    return df_tweet

df_tweet = load_data()

# Preprocessing functions
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub('', text)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def remove_emoji(tweet):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", tweet)

def remove_mentions_hashtags(text):
    return re.sub(r'@\w+|#\w+', '', text)

def remove_angka(tweet):
    tweet = re.sub('[0-9]+', '', tweet)
    tweet = re.sub(r'\$\w', '', tweet)
    tweet = re.sub(r'^RT[\s]', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    return tweet

def remove_punc(text):
    exclude = string.punctuation
    for char in exclude:
        text = text.replace(char, '')
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def nama_admin(text):
    pattern = re.compile(r'-\w+')
    return pattern.sub('', text)

# Preprocess data
df_processed = df_tweet.copy()
df_processed['clean_tweet'] = df_processed['full_text'].str.lower()
df_processed['clean_tweet'] = df_processed['clean_tweet'].apply(remove_url)
df_processed['clean_tweet'] = df_processed['clean_tweet'].apply(remove_emoji)
df_processed['clean_tweet'] = df_processed['clean_tweet'].apply(remove_mentions_hashtags)
df_processed['clean_tweet'] = df_processed['clean_tweet'].apply(remove_stopwords)
df_processed['clean_tweet'] = df_processed['clean_tweet'].apply(nama_admin)

# Sentiment classification
def classify_sentiment(text):
    positive_phrases = ["kembali normal", "sudah lancar", "banyak nih", "cukup dengan", "menarik", "promo", "ganti telkomsel",
                        "rekomendasi", "ada sinyal", "mendingan", "mending telkomsel", "cantik", "senyum", "banyak pilihan",
                        "udah bisa", "sudah bisa", "udh bisa", "udah jadi", "murah", "akhirnya", "finally", "setia sama telkomsel",
                        "udah pakai", "diskon", "gratis", "lancar", "happy", "pilih telkomsel", "terjangkau", "worth it", "aman",
                        "mantap", "enjoy", "favorit", "setia", "tawaran", "turun", "pindah ke telkomsel", "pake telkomsel",
                        "langganan telkomsel", "sudah normal kembali", "sudah kembali normal"]

    negative_phrases = ["tidak suka", "kecewa dengan layanan", "ga bs", "kendala", "belum bisa", "reinstall", "re install",
                        "maaf", "kenapa sinyalmu", "gimana ini", "gmn ini", "belum didapat", "gak bisa", "nggak bisa", "ngga bisa",
                        "gabisa", "tidak bisa", "g bisa", "gbsa", "gak bs", "belum berhasil", "belom berhasil", "ke detect diluar",
                        "lemot", "lambat", "dibalikin indihom", "clear cache", "berat", "hilang", "ga stabil", "belum masuk",
                        "tidak dapat", "force close", "gbs kebuka", "mahal", "tdk bisa dibuka", "komplain", "uninstal",
                        "tiba-tiba", "tiba2", "tb2", "dipotong", "gak mantap", "maapin", "ribet banget", "gada promo", "minusnya",
                        "ga ada", "gaada", "benerin", " lelet", "naik terus", "nyesel", "berhentikan", "ga mau nurunin", "masalah",
                        "nihil", "tidak respons", "restart", "gak jelas", "re-install", "terganggu", "sms iklan/promo",
                        "paksa keluar", "gangguan"]

    positive_words = ["baik", "bagus", "puas", "senang", "menyala", "pengguna", "lancar", "meding", "setia", "selamat",
                      "akhirnya", "keren", "beruntung", "senyum", "cantik", "mantap", "percaya", "merakyat", "aman", "sesuai",
                      "seru", "explore", "suka", "berhasil", "stabil", "adil", "pindah ke telkomsel", "terbaik"]

    negative_words = ["buruk", "kecewa", "mengecewakan", "kurang", "diperbaiki", "nggak bisa", "dijebol", "jelek", "gak dapet",
                      "nggak dapat", "gak dapat", "ga dapat", "biar kembali stabil", "biar balik stabil", "lemot", "error",
                      "eror", "ngga", "berkurang", "benci", "mahal", "lambat", "sedih", "kesel", "scam", "pusing",
                      "ganggu", "gangguan", "sampah", "kepotong", "bug", "spam", "kacau", "nunggu", "complain", "komplain",
                      "kapan sembuh", "maap", "kendala", "susah", "kenapa", "males", "bapuk", "keluhan", "bosen", "mehong",
                      "tipu", "belum", "nipu", "lelet", "parah", "emosi", "lemah", "ngelag", "ribet", "repot", "capek", "nangis",
                      "connecting", "waduh", "ketidaksesuaian", "stop", "kesal", "dituduh", "ga di respon", "ilang",
                      "kaya gini terus", "uninstall", "pinjol", "kelolosan", "force close", "lag", "gbs kebuka", "crash",
                      "menyesal", "bubar", "re-instal", "menghentikan", "bakar", "bosok"]

    positive_count = sum(1 for phrase in positive_phrases if phrase in text) + sum(1 for word in positive_words if word in text.split())
    negative_count = sum(1 for phrase in negative_phrases if phrase in text) + sum(1 for word in negative_words if word in text.split())

    if negative_count > positive_count:
        return "negatif"
    elif positive_count > negative_count:
        return "positif"
    else:
        return "netral"

df_processed['sentiment'] = df_processed['clean_tweet'].apply(classify_sentiment)

# Train-test split
X = df_processed['clean_tweet']
y = df_processed['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words=stopwords.words('indonesian'))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Naive Bayes model
naivebayes = MultinomialNB()
naivebayes.fit(X_train_tfidf, y_train)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train_tfidf, y_train)

# Streamlit app
st.title("Analisis Sentimen Twitter Terhadap Telkomsel")

# Input text for prediction
new_text = st.text_area("Masukkan teks untuk prediksi sentimen:")

if st.button("Prediksi"):
    if new_text:
        # Preprocess input text
        new_text_processed = remove_url(new_text)
        new_text_processed = remove_emoji(new_text_processed)
        new_text_processed = remove_mentions_hashtags(new_text_processed)
        new_text_processed = remove_stopwords(new_text_processed)
        new_text_processed = nama_admin(new_text_processed)

        # Predict sentiment
        text_tfidf = tfidf.transform([new_text_processed])
        sentiment_nb = naivebayes.predict(text_tfidf)[0]
        sentiment_knn = knn.predict(text_tfidf)[0]

        st.write(f"Sentimen (Naive Bayes): {sentiment_nb}")
        st.write(f"Sentimen (KNN): {sentiment_knn}")
    else:
        st.write("Silakan masukkan teks untuk prediksi.")

# Show dataset
if st.checkbox("Tampilkan Dataset"):
    st.write(df_processed)

# Show sentiment distribution
if st.checkbox("Tampilkan Distribusi Sentimen"):
    sentiment_counts = df_processed['sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
                hue=sentiment_counts.index,
                palette={'positif': 'green', 'negatif': 'red', 'netral': 'gray'},
                legend=False)
    plt.title('Distribusi Sentimen')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    st.pyplot(plt)

# Show top words
if st.checkbox("Tampilkan Kata-Kata yang Sering Muncul"):
    stopwords_id = set(stopwords.words('indonesian'))
    custom_stopwords = {
        "telkomsel", "mytelkomsel", "dm", "min", "hp", "nomor", "@telkomsel",
        "ya", "kak", "nih", "aja", "makasih", "udah", "nih", "ga", "gak",
        "coba", "bantu", "infoin", "yg", "aja", "cek", "banget", "yaa", 'ya.',
        'kakak', ':)', 'nya', 'kalo', 'biar', 'maaf', 'kak.', 'kak', '-feri', 'mohon',
        'yuk', '-dhea', 'pake', 'silakan', 'ditunggu', '-beeru', 'via', 'makasih.',
        'clear', 'maafin', '-joan'
    }
    stopwords_combined = stopwords_id.union(custom_stopwords)
    text = " ".join(df_processed['clean_tweet'])
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords_combined]
    word_counts = Counter(filtered_tokens)
    top_words = word_counts.most_common(10)
    word, count = zip(*top_words)
    colors = plt.cm.Paired(range(len(word)))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(word, count, color=colors)
    plt.xlabel("Kata")
    plt.ylabel("Frekuensi")
    plt.title("Kata-Kata yang Sering Muncul")
    plt.xticks(rotation=45)
    for bar, num in zip(bars, count):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), fontsize=12, color='black', ha='center')
    st.pyplot(plt)