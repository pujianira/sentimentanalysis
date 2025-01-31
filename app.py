import streamlit as st
import pandas as pd

# Memuat file CSV
df = pd.read_csv('telkomsel.csv')

# Menampilkan dataset di Streamlit
st.write("Dataset:", df)

# Atau menampilkan dataset dalam bentuk tabel interaktif
st.dataframe(df)