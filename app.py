import streamlit as st
import pandas as pd

# 📂**1.Import & Load Dataset**🧩

# **a. Import dataset**
# df = pd.read_csv('telkomsel.csv')

# # **b. Lima baris awal dari dataset**
# st.markdown("### Lima Baris Awal dari Dataset")
# st.write(df.head())

import os
import streamlit as st

file_path = 'telkomsel.csv'
if os.path.exists(file_path):
    st.write("File ditemukan!")
    df = pd.read_csv(file_path)
    st.write(df.head())
else:
    st.error(f"File {file_path} tidak ditemukan.")
