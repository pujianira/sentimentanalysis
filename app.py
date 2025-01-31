import streamlit as st
import pandas as pd

st.markdown("""
# **Twitter Sentiment Analysis** 

## ‼️ **Tujuan** ⁉️
Untuk memahami persepsi pelanggan terhadap perusahaan dengan melakukan analisis sentimen terhadap ulasan pelanggan dan media sosial Twitter. 
Ini bertujuan untuk mengidentifikasi isu utama yang dihadapi perusahaan.          
""")

# 📂 **1. Import & Load Dataset**
st.markdown("# 📂 **1. Import & Load Dataset** 🧩")
st.markdown("**a. Import dataset**")

# Memuat file CSV
df = pd.read_csv('telkomsel.csv')

# Menampilkan dataset
st.markdown("**b. Lima baris awal dari dataset**")
st.dataframe(df.head())

# Menampilkan hanya kolom yang berisi tweet pengguna
st.markdown("**c. Dataframe dengan kolom yang berisi tweet pengguna**")
df_tweet = pd.DataFrame(df[['full_text']])
st.dataframe(df_tweet)

# Preprocessing Data
st.markdown("**d. Setelah drop duplicates**")
df_tweet['full_text'] = df_tweet['full_text'].astype(str).str.lower().str.strip()
df_tweet.drop_duplicates(subset=['full_text'], inplace=True)
st.dataframe(df_tweet)

# 📂 **2. Klasifikasi Sentimen**
st.markdown("# 📂 **2. Klasifikasi Sentimen** 🧩")

def classify_sentiment(text):
    positive_phrases = ["kembali normal", "sudah lancar", "promo", "diskon", "gratis", "mantap", "happy"]
    negative_phrases = ["tidak suka", "kecewa", "lemot", "error", "mahal", "gak bisa", "gangguan"]

    positive_count = sum(1 for phrase in positive_phrases if phrase in text)
    negative_count = sum(1 for phrase in negative_phrases if phrase in text)

    if negative_count > positive_count:
        return "Negatif"
    elif positive_count > negative_count:
        return "Positif"
    else:
        return "Netral"

df_tweet['sentiment'] = df_tweet['full_text'].apply(classify_sentiment)

# Menampilkan hasil analisis sentimen
st.markdown("**e. Hasil Sentimen**")
st.dataframe(df_tweet)
