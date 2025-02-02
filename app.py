import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  
from sklearn.metrics import classification_report, accuracy_score

st.markdown("""
# **Twitter Sentiment Analysis** 

# ‼️**Tujuan**⁉️
Untuk memahami persepsi pelanggan terhadap perusahaan dengan melakukan analisis sentimen terhadap ulasan pelanggan dan media sosial twitter. 
Ini bertujuan untuk mengidentifikasi isu utama yang dihadapi perusahaan.          
""")

# 📂 **1. Import & Load Dataset**
st.markdown("# 📂**1. Import & Load Dataset**🧩")

# Memuat file CSV
df = pd.read_csv('telkomsel.csv')

# Menampilkan dataset
st.markdown("**a. Lima baris awal dari dataset**")
st.dataframe(df.head())

# Menampilkan hanya kolom yang berisi tweet pengguna
st.markdown("**b. Dataframe dengan kolom yang berisi tweet pengguna**")
df_tweet = pd.DataFrame(df[['full_text']])
st.dataframe(df_tweet)

# Preprocessing Data
# st.markdown("**c. Setelah drop duplicates**")
df_tweet['full_text'] = df_tweet['full_text'].astype(str).str.lower().str.strip()
df_tweet.drop_duplicates(subset=['full_text'], inplace=True)
# st.dataframe(df_tweet)

# df=pd.read_csv('telkomsel.csv')
# df_tweet = pd.DataFrame(df[['full_text']])
# df_tweet['full_text'] = df_tweet['full_text'].str.lower()
# df_tweet.drop_duplicates(inplace=True)

# 📂 **2. Klasifikasi Sentimen**
st.markdown("# 📂**2. Distribusi Sentimen**🧩")
# st.markdown("# 📊 Distribusi Sentimen")

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
st.markdown("**a. Hasil Sentimen**")
st.dataframe(df_tweet)

# st.markdown("# 📊 Distribusi Sentimen")

# Menghitung jumlah masing-masing sentimen
sentiment_counts = df_tweet['sentiment'].value_counts()

# Menampilkan tabel jumlah sentimen
st.write("**b. Jumlah Sentimen**")
st.dataframe(sentiment_counts)

# Membuat visualisasi distribusi sentimen
st.write("**c. Visualisasi Distribusi Sentimen**")
fig, ax = plt.subplots(figsize=(6, 4))  

sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
            hue=sentiment_counts.index,
            palette={'positif': 'green', 'negatif': 'red', 'netral': 'gray'},
            legend=False)

#plt.title('Distribusi Sentimen')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')

st.pyplot(fig)

st.markdown("# 📂**3. Pre-Processing**🧩")

# import re
# import string
# import streamlit as st
# import pandas as pd
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords

# # Download resource NLTK yang diperlukan
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)

# class TextPreprocessor:
#     def __init__(self):
#         self.lemmatizer = WordNetLemmatizer()
#         self.stopwords_id = set(stopwords.words('indonesian'))
#         self.stopwords_id.update([
#             'yg', 'dgn', 'nya', 'utk', 'ke', 'di', 'dr', 'dan', 'atau',
#             'ini', 'itu', 'juga', 'sudah', 'saya', 'anda', 'dia', 'mereka',
#             'kita', 'akan', 'bisa', 'ada', 'tidak', 'saat', 'oleh', 'setelah',
#             'pada', 'seperti', 'dll', 'dear', 'admin', 'cs', 'customer',
#             'service', 'mohon', 'tolong', 'bantu', 'bantuan', 'terima', 'kasih',
#             'maaf', 'sorry', 'pls', 'please', 'help', 'thanks', 'tq', 'thx',
#             'via', 'adalah', 'yang', 'dari', 'dalam', 'untuk', 'dengan', 'se',
#             'bagi', 'telah', 'serta', 'agar', 'udah', 'kak', 'min', 'aci',
#             'makasih', 'mytelkomselnya', 'versi', 'dibantu', 'silakan', 'maafin',
#             'kalo', 'halo', 'hai'
#         ])

#         self.noise_words = {'kak', 'ya', 'yg', 'gitu', 'nih', 'dong', 'sih', 'kan', 'aja'}
#         self.irrelevant_words = {'telkomsel', 'hp', 'maaf', 'joan', 'gb', 'indosat', 'kakak'}
#         self.all_stopwords = self.stopwords_id | self.noise_words | self.irrelevant_words
#         self.punctuation_table = str.maketrans('', '', string.punctuation)

#     def remove_url(self, text):
#         return re.sub(r'https?://\S+|www\.\S+', '', text)

#     def remove_emoji(self, text):
#         emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
#                                    u"\U0001F300-\U0001F5FF"
#                                    u"\U0001F680-\U0001F6FF"
#                                    u"\U0001F700-\U0001F77F"
#                                    u"\U0001F780-\U0001F7FF"
#                                    u"\U0001F800-\U0001F8FF"
#                                    u"\U0001F900-\U0001F9FF"
#                                    u"\U0001FA00-\U0001FA6F"
#                                    u"\U0001FA70-\U0001FAFF"
#                                    u"\U00002702-\U000027B0"
#                                    u"\U000024C2-\U0001F251"
#                                    "]+", flags=re.UNICODE)
#         return emoji_pattern.sub("", text)

#     def remove_mentions_hashtags(self, text):
#         text = re.sub(r'@telkomsel', 'telkomsel', text)  # Ganti @telkomsel jadi telkomsel
#         text = re.sub(r'@([^\s]+)', ' ', text)  # Hapus mention lain
#         text = re.sub(r'#[^\s]+', ' ', text)  # Hapus hashtag
#         return text

#     def remove_numbers(self, text):
#         text = re.sub(r'rp\s?[0-9]+([.,][0-9]+)?', '', text)  # Hapus angka rupiah
#         text = re.sub(r'[0-9]+', '', text)  # Hapus angka
#         return text

#     def remove_punctuation(self, text):
#         return text.translate(self.punctuation_table)

#     def remove_stopwords(self, text):
#         words = text.split()
#         filtered_words = [word for word in words if word not in self.all_stopwords]
#         return ' '.join(filtered_words)

#     def preprocess_text(self, text):
#         text = text.lower()
#         text = self.remove_url(text)
#         text = self.remove_emoji(text)
#         text = self.remove_mentions_hashtags(text)
#         text = self.remove_numbers(text)
#         text = self.remove_punctuation(text)
#         text = self.remove_stopwords(text)

#         tokens = word_tokenize(text)
#         tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
#         return ' '.join(tokens)

# preprocessor = TextPreprocessor()
# # Tampilkan hasil
# st.write("### 📜 Data Sebelum & Sesudah Preprocessing")
# df_clean = df_tweet.copy()
# print(df_clean.head())  # Pastikan df_clean ada
# print(df_clean.columns)  # Pastikan kolom 'full_text' ada
# df_clean['clean_text'] = df_clean['full_text'].apply(preprocessor.preprocess_text)
# st.dataframe(df_clean[['full_text', 'clean_text']].head(10))

st.markdown("# 📂**4. Analisis Topik**🧩")
st.markdown("### **Analisis Topik Tweet Negatif**")

st.markdown("#### **📝 Cluster 0**")
# st.write("Jumlah Tweet: **592**")

st.markdown("#####✏️ Contoh Tweet:")
st.write("- @febridifafzx @febridifafzxgaada masalah kak febri jika membeli paket menggunakan perangkat lain seperti melalui e-commerce. untuk paket awalnya memang kakak hanya bisa memilih paket lite saja yang gabisa tethring namun setelah kartu perdana diterima kakak bisa aktifin paket yang dapat")
st.write("- @proxymous @proxymousmaaf ya kak atas ketidaknyamanan yang dialami. mengenai kendala sinyal gak stabil saat mati lampu yuk infoin dulu nomor hp waktu tanggal kendala mulai dialami lokasi (kec kel kab/kota) dan nomor telkomsel lain berkendala sama jika ada melalui dm. biar dibantu cek")
st.write("- @aguesstybear @aguesstybear maaf kak agus untuk kendala pembelian paket infoin nomor hp capture keterangan gagal sama detail paket yang ingin diaktivasi via dm yuk untuk cek lebih lanjut. tks :) -frey")

st.markdown("##### **🔍 Top Topics: 🔍**")
st.write("1. **kendala** (Kemunculan: 101)")
st.write("2. **paket** (Kemunculan: 65)")
st.write("3. **lokasi** (Kemunculan: 42)")

st.markdown("---")

st.markdown("#### **📝 Cluster 1**")
# st.write("Jumlah Tweet: **80**")

st.markdown("#####✏️ Contoh Tweet:")
st.write("- @hg76xp @hg76xp halo kak hendri. maaf jika aplikasi mytelkomsel di hp-nya masih berkendala ya. btw sudah coba uninstall dulu kemudian install kembali aplikasi mytelkomselnya di play store dengan aplikasi mytelkomsel versi 8.6.1 kak? kalo sudah tapi masih error juga silakan infoin nomor")
st.write("- @ica_giannisa maafin banget ya kak giannisa. untuk kendala gak bisa akses aplikasi mytelkomsel silakan clear cache aplikasi uninstall aplikasi lalu install kembali dengan versi terbaru ya. jika masih berkendala infoin nomor hp tipe perangkat versi os ke dm supaya bisa dibantu cek lebih")
st.write("- @rmsa814 @rmsa814 maaf ya kak setya atas ketidaknyamanan yang dialami. mengenai kendala aplikasi mytelkomsel zyad saranin kakak coba clear cache aplikasi lalu re-install aplikasi terlebih dahulu ya. namun jika masih tetap sama yuk infoin nomor hp waktu tanggal kejadian tipe hp sama")

st.markdown("##### **🔍 Top Topics: 🔍**")
st.write("1. **aplikasi** (Kemunculan: 87)")
st.write("2. **mytelkomsel** (Kemunculan: 61)")
st.write("3. **clear** (Kemunculan: 48)")

# import streamlit as st
# import pandas as pd
# import nltk
# import string
# import re
# from collections import Counter
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# nltk.download('stopwords', quiet=True)
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)

# # Class untuk Clustering
# class NegativeTweetClustering:
#     def __init__(self, n_clusters, n_topics):
#         self.n_clusters = n_clusters
#         self.n_topics = n_topics
#         self.stopwords_id = set(stopwords.words('indonesian'))
#         self.lemmatizer = WordNetLemmatizer()
#         self.noise_words = {'telkomsel', 'kak', 'nih', 'aja', 'rp'}
#         self.stopwords_id.update(self.noise_words)

#     def preprocess_text(self, text):
#         tokens = word_tokenize(str(text).lower())
#         tokens = [
#             self.lemmatizer.lemmatize(word)
#             for word in tokens
#             if word.isalpha() and word not in self.stopwords_id
#         ]
#         return " ".join(tokens)

#     def cluster_and_analyze(self, df_tweet):
#         df_negatif = df_tweet[df_tweet['sentiment'] == 'negatif'].copy()
#         df_negatif['cleaned_text'] = df_negatif['full_text'].apply(self.preprocess_text)

#         vectorizer = TfidfVectorizer(max_features=1000)
#         X = vectorizer.fit_transform(df_negatif['cleaned_text'])

#         pca = PCA(n_components=2)
#         X_reduced = pca.fit_transform(X.toarray())

#         kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
#         df_negatif['cluster'] = kmeans.fit_predict(X_reduced)

#         silhouette_avg = silhouette_score(X_reduced, df_negatif['cluster'])
#         st.write(f"**Silhouette Score: {silhouette_avg:.4f}**")

#         cluster_results = []
#         for cluster in range(self.n_clusters):
#             cluster_data = df_negatif[df_negatif['cluster'] == cluster]
#             all_words = " ".join(cluster_data['cleaned_text']).split()
#             word_counts = Counter(all_words)
#             top_words = word_counts.most_common(self.n_topics)

#             cluster_results.append({
#                 'Cluster': cluster,
#                 'Jumlah Tweet': len(cluster_data),
#                 'Topik Utama': ', '.join([word for word, _ in top_words]),
#                 'Contoh Tweet': cluster_data['full_text'].iloc[:3].tolist()
#             })

#         return df_negatif, cluster_results

# # Clustering
# clustering = NegativeTweetClustering(n_clusters=3, n_topics=5)
# df_clustered, cluster_results = clustering.cluster_and_analyze(df_tweet)

# st.subheader("Data Setelah Clustering")
# st.dataframe(df_clustered[['full_text', 'cleaned_text', 'cluster']])

# st.subheader("Analisis Cluster")
# for result in cluster_results:
#     st.markdown(f"### Cluster {result['Cluster']}")
#     st.write(f"**Jumlah Tweet:** {result['Jumlah Tweet']}")
#     st.write(f"**Topik Utama:** {result['Topik Utama']}")
    
#     st.write("**Contoh Tweet:**")
#     for tweet in result['Contoh Tweet']:
#         st.write(f"- {tweet}")


# def _print_cluster_analysis(self, cluster_results):
#     st.subheader("🔍 Analisis Topik Tweet Negatif")
#     for cluster, info in cluster_results.items():
#         st.write(f"### Cluster {cluster}")
#         st.write(f"**Jumlah Tweet:** {info['tweet_count']}")

#         st.write("**Contoh Tweet:**")
#         for tweet in info['sample_tweets']:
#             st.write(f"- {tweet}")

#         st.write("**Top Topics:**")
#         for i, (topic, count) in enumerate(info['topics'], 1):
#             st.write(f"{i}. **{topic}** (Kemunculan: {count})")

#         st.write("---")