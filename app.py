import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set page config
st.set_page_config(page_title="Analisis Sentimen Twitter Telkomsel", layout="wide")

# Title
st.title("Analisis Sentimen Twitter Telkomsel")

# Sidebar
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    df_tweet = pd.DataFrame(df[['full_text']])
    
    # Text preprocessing functions
    @st.cache_data
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emoji
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
        text = emoji_pattern.sub(r'', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove numbers
        text = re.sub(r'[0-9]+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove stopwords
        stop_words = set(stopwords.words('indonesian'))
        words = text.split()
        text = ' '.join([word for word in words if word not in stop_words])
        
        return text

    # Sentiment classification function
    @st.cache_data
    def classify_sentiment(text):
        positive_phrases = [
            "kembali normal", "sudah lancar", "banyak nih", "cukup dengan", "menarik", "promo",
            "rekomendasi", "ada sinyal", "mendingan", "mending telkomsel", "cantik", "senyum",
            "udah bisa", "sudah bisa", "udh bisa", "udah jadi", "murah", "akhirnya", "finally",
            "udah pakai", "diskon", "gratis", "lancar", "happy", "pilih telkomsel", "terjangkau"
        ]

        negative_phrases = [
            "tidak suka", "kecewa dengan layanan", "ga bs", "kendala", "belum bisa", "reinstall",
            "maaf", "kenapa sinyalmu", "gimana ini", "gmn ini", "belum didapat", "gak bisa",
            "nggak bisa", "ngga bisa", "gabisa", "tidak bisa", "g bisa", "gbsa", "gak bs",
            "lemot", "lambat", "clear cache", "berat", "hilang", "ga stabil", "belum masuk"
        ]

        text = text.lower()
        positive_count = sum(1 for phrase in positive_phrases if phrase in text)
        negative_count = sum(1 for phrase in negative_phrases if phrase in text)

        if negative_count > positive_count:
            return "negatif"
        elif positive_count > negative_count:
            return "positif"
        else:
            return "netral"

    # Process data
    df_tweet['clean_text'] = df_tweet['full_text'].apply(preprocess_text)
    df_tweet['sentiment'] = df_tweet['full_text'].apply(classify_sentiment)

    # Display data analysis
    st.header("Analisis Data")
    
    # Sentiment distribution
    st.subheader("Distribusi Sentimen")
    sentiment_counts = df_tweet['sentiment'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
                hue=sentiment_counts.index,
                palette={'positif': 'green', 'negatif': 'red', 'netral': 'gray'},
                legend=False, ax=ax)
    plt.title('Distribusi Sentimen')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    st.pyplot(fig)

    # Train models
    X = df_tweet['clean_text']
    y = df_tweet['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    tfidf = TfidfVectorizer(max_features=5000, stop_words=stopwords.words('indonesian'))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Naive Bayes
    naivebayes = MultinomialNB()
    naivebayes.fit(X_train_tfidf, y_train)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=13)
    knn.fit(X_train_tfidf, y_train)

    # Prediction functions
    def predict_sentiment_naivebayes(text):
        text_tfidf = tfidf.transform([text])
        return naivebayes.predict(text_tfidf)[0]

    def predict_sentiment_knn(text):
        text_tfidf = tfidf.transform([text])
        return knn.predict(text_tfidf)[0]

    # Prediction interface
    st.header("Prediksi Sentimen")
    input_text = st.text_area("Masukkan teks untuk prediksi sentimen:")
    
    if st.button("Prediksi"):
        if input_text:
            nb_prediction = predict_sentiment_naivebayes(input_text)
            knn_prediction = predict_sentiment_knn(input_text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Prediksi Naive Bayes")
                st.write(f"Sentimen: {nb_prediction}")
            
            with col2:
                st.subheader("Prediksi KNN")
                st.write(f"Sentimen: {knn_prediction}")
        else:
            st.warning("Mohon masukkan teks untuk diprediksi")

else:
    st.info("Silakan upload file CSV untuk memulai analisis")