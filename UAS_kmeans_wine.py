import streamlit as st
import pickle
import numpy as np

# Memuat model
with open('wine-clustering.pkl', 'rb') as f: 
    model_kmeans = pickle.load(f)

# Judul Aplikasi
st.title('Prediksi Jenis anggur')
st.text('Model untuk Prediksi: Kmeans') 

# Dropdown untuk memilih model
model = model_kmeans

# Inisialisasi atau reset hasil jika model berubah
Alcohol = st.slider("Kandungan Alkohol (%):", 0, 100, 0)
Malic_Acid = st.number_input('Kandungan Asam Malat (Malic Acid):', min_value=0)
Ash = st.number_input('Kandungan Ash (Abu):', min_value=0)
Ash_Alcanity = st.number_input('Kandungan Ash Alkalinity:', min_value=0)
Magnesium = st.number_input('Kandungan Magnesium:', min_value=0)
Total_Phenols = st.number_input('Kandungan Total Fenol:', min_value=0)
Flavanoids = st.number_input('Kandungan Flavanoid:', min_value=0)
Nonflavanoid_Phenols = st.number_input('Kandungan Non-Flavanoid Fenol:', min_value=0)
Proanthocyanins = st.number_input('Kandungan Proanthocyanins:', min_value=0)
Color_Intensity = st.number_input('Intensitas Warna (Color Intensity):', min_value=0)
Hue = st.number_input('Hue (Warna):', min_value=0)
OD280 = st.number_input('OD280/OD315 Rasio (Absorbansi):', min_value=0)
Proline = st.number_input('Kandungan Proline:', min_value=0)


features = np.array([Alcohol, Malic_Acid, Ash, Ash_Alcanity, Magnesium, Total_Phenols, Flavanoids,
                     Nonflavanoid_Phenols, Proanthocyanins, Color_Intensity, Hue, OD280, Proline])
# Tombol untuk memprediksi spesies ikan
if st.button('Prediksi anggur :'):
    features = np.array([[Alcohol, Malic_Acid, Ash, Ash_Alcanity, Magnesium, Total_Phenols, Flavanoids,
                     Nonflavanoid_Phenols, Proanthocyanins, Color_Intensity, Hue, OD280, Proline]])
    
    predicted_klaster = model.predict(features.reshape(1, -1))[0]
    predicted_klaster_label = ['Type-1', 'Type-2', 'Type-3'][predicted_klaster]
    st.success(f"Data telah berhasil dikelompokkan dalam klaster: {predicted_klaster_label}")
    
    pecies = model.predict(features)[0]
    st.success(f'Spesies yang Diprediksi kelas ke: {pecies}')

