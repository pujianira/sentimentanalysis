import os
import streamlit as st
import pandas as pd

file_path = 'dataset.csv'

# Debugging dengan print
print(f"Checking if file exists: {os.path.exists(file_path)}")
st.write(f"Path to file: {file_path}")

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.write(df.head())
else:
    st.error(f"File {file_path} tidak ditemukan.")