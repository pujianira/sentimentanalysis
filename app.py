import streamlit as st
import pandas as pd
import os

# Path ke file dataset
file_path = 'dataset.csv'

# Memeriksa apakah file ada
if os.path.exists(file_path):
    st.write("File ditemukan!")
    df = pd.read_csv(file_path)
    st.write(df.head())  # Tampilkan lima baris pertama dari dataset
else:
    st.error(f"File {file_path} tidak ditemukan.")