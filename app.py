import streamlit as st
import pandas as pd

st.markdown("""
# **Twitter Sentiment Analysis** 

# ‼️**Tujuan**⁉️
Untuk memahami persepsi pelanggan terhadap perusahaan dengan melakukan analisis sentimen terhadap ulasan pelanggan dan media sosial twitter. 
Ini bertujuan untuk mengidentifikasi isu utama yang dihadapi perusahaan.          
""")

# 📂 **1. Import & Load Dataset**
st.markdown("# 📂**1. Import & Load Dataset**🧩")
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
st.markdown("# 📂**2. Klasifikasi Sentimen**🧩")

def classify_sentiment(text):
    positive_phrases = [
        "kembali normal", "sudah lancar", "banyak nih", "cukup dengan", "menarik", "promo", "ganti telkomsel",
        "rekomendasi", "ada sinyal", "mendingan", "mending telkomsel", "cantik", "senyum", "banyak pilihan",
        "udah bisa", "sudah bisa", "udh bisa", "udah jadi", "murah", "akhirnya", "finally", "setia sama telkomsel",
        "udah pakai", "diskon", "gratis", "lancar", "happy", "pilih telkomsel", "terjangkau", "worth it", "aman",
        "mantap", "enjoy", "favorit", "setia", "tawaran", "turun", "pindah ke telkomsel", "pake telkomsel",
        "langganan telkomsel", "sudah normal kembali", "sudah kembali normal"
    ]

    negative_phrases = [
        "tidak suka", "kecewa dengan layanan", "ga bs", "kendala", "belum bisa", "reinstall", "re install",
        "maaf", "kenapa sinyalmu", "gimana ini", "gmn ini", "belum didapat", "gak bisa", "nggak bisa", "ngga bisa",
        "gabisa", "tidak bisa", "g bisa", "gbsa", "gak bs", "belum berhasil", "belom berhasil", "ke detect diluar",
        "lemot", "lambat", "dibalikin indihom", "clear cache", "berat", "hilang", "ga stabil", "belum masuk",
        "tidak dapat", "force close", "gbs kebuka", "mahal", "tdk bisa dibuka", "komplain", "uninstal",
        "tiba-tiba", "tiba2", "tb2", "dipotong", "gak mantap", "maapin", "ribet banget", "gada promo", "minusnya",
        "ga ada", "gaada", "benerin", " lelet", "naik terus", "nyesel", "berhentikan", "ga mau nurunin", "masalah",
        "nihil", "tidak respons", "restart", "gak jelas", "re-install", "terganggu", "sms iklan/promo",
        "paksa keluar", "gangguan"
    ]

    positive_words = [
        "baik", "bagus", "puas", "senang", "menyala", "pengguna", "lancar", "meding", "setia", "selamat",
        "akhirnya", "keren", "beruntung", "senyum", "cantik", "mantap", "percaya", "merakyat", "aman", "sesuai",
        "seru", "explore", "suka", "berhasil", "stabil", "adil", "pindah ke telkomsel", "terbaik"
    ]

    negative_words = [
        "buruk", "kecewa", "mengecewakan", "kurang", "diperbaiki", "nggak bisa", "dijebol", "jelek", "gak dapet",
        "nggak dapat", "gak dapat", "ga dapat", "biar kembali stabil", "biar balik stabil", "lemot", "error",
        "eror", "ngga", "berkurang", "benci", "mahal", "lambat", "sedih", "kesel", "scam", "pusing",
        "ganggu", "gangguan", "sampah", "kepotong", "bug", "spam", "kacau", "nunggu", "complain", "komplain",
        "kapan sembuh", "maap", "kendala", "susah", "kenapa", "males", "bapuk", "keluhan", "bosen", "mehong",
        "tipu", "belum", "nipu", "lelet", "parah", "emosi", "lemah", "ngelag", "ribet", "repot", "capek", "nangis",
        "connecting", "waduh", "ketidaksesuaian", "stop", "kesal", "dituduh", "ga di respon", "ilang",
        "kaya gini terus", "uninstall", "pinjol", "kelolosan", "force close", "lag", "gbs kebuka", "crash",
        "menyesal", "bubar", "re-instal", "menghentikan", "bakar", "bosok"
    ]

    positive_count = sum(1 for phrase in positive_phrases if phrase in text) + sum(1 for word in positive_words if word in text.split())
    negative_count = sum(1 for phrase in negative_phrases if phrase in text) + sum(1 for word in negative_words if word in text.split())

    if negative_count > positive_count:
        return "negatif"
    elif positive_count > negative_count:
        return "positif"
    else:
        return "netral"

df_tweet['sentiment'] = df_tweet['full_text'].apply(classify_sentiment)

# Menampilkan hasil analisis sentimen
st.markdown("**e. Hasil Sentimen**")
st.dataframe(df_tweet)