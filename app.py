import streamlit as st

# Judul aplikasi
st.title("Aplikasi Streamlit Pertamaku!")

# Menambahkan teks
st.write("Ini adalah aplikasi Streamlit sederhana!")

# Menambahkan input teks
name = st.text_input("Masukkan nama kamu:")
if name:
    st.write(f"Halo, {name}!")

# Menambahkan slider
age = st.slider("Pilih umur kamu:", 10, 60)
st.write(f"Umur kamu adalah: {age} tahun.")
