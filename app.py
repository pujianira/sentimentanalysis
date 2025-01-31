import streamlit as st
import pandas as pd

# 📂**1.Import & Load Dataset**🧩

# **a. Import dataset**
df = pd.read_csv('telkomsel.csv')

# **b. Lima baris awal dari dataset**
st.markdown("### Lima Baris Awal dari Dataset")
st.write(df.head())
