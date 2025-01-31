import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('telkomsel.csv')
        return df
    except FileNotFoundError:
        st.error("File 'dataset.csv' tidak ditemukan. Pastikan file ada di direktori yang benar.")
        return None

data = load_data()
if data is not None:
    st.write(data)