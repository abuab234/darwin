import streamlit as st
import pandas as pd
import pickle

# Konfigurasi Halaman (Modern UI)
st.set_page_config(page_title="Deteksi Dini Alzheimer", layout="centered", page_icon="🧠")

st.title("🧠 Deteksi Dini Alzheimer (DARWIN Dataset)")
st.write("Aplikasi ini menggunakan Machine Learning (*Bagging Classifier*) untuk memprediksi potensi Alzheimer berdasarkan pola pergerakan tulisan tangan.")

# Cache resource agar model tidak di-load ulang setiap kali ada interaksi UI
@st.cache_resource
def load_components():
    # Pastikan nama file sesuai dengan yang ada di folder Anda
    with open('bagging_classifier_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler (1).pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        le = pickle.load(file)
    return model, scaler, le

# Coba memuat model
try:
    model, scaler, le = load_components()
    st.success("✅ Model, Scaler, dan Encoder berhasil dimuat ke sistem!")
except Exception as e:
    st.error(f"⚠️ Gagal memuat file model. Pastikan file .pkl berada di folder yang sama dengan app.py. Error: {e}")
    st.stop()

st.markdown("---")
st.subheader("Uji Data Pasien")
st.info("💡 Karena terdapat 451 fitur parameter tulisan tangan, silakan unggah file CSV berisi metrik data pasien (tanpa kolom label/target).")

# Fitur Upload File
uploaded_file = st.file_uploader("Unggah file CSV metrik tulisan tangan", type=["csv"])

if uploaded_file is not None:
    # Membaca data CSV yang diunggah
    input_data = pd.read_csv(uploaded_file)
    
    st.write("**Preview Data Input:**")
    st.dataframe(input_data.head(5))

    # Tombol Prediksi
    if st.button("Jalankan Prediksi AI", type="primary"):
        with st.spinner("Menganalisis pola tulisan tangan..."):
            try:
                # 1. Standarisasi data input menggunakan Scaler bawaan
                scaled_data = scaler.transform(input_data)
                
                # 2. Lakukan Prediksi
                predictions = model.predict(scaled_data)
                
                # 3. Kembalikan kode prediksi ke label asli menggunakan Label Encoder
                decoded_predictions = le.inverse_transform(predictions)
                
                st.subheader("📋 Hasil Analisis:")
                
                # Tampilkan hasil untuk setiap baris data
                for i, result in enumerate(decoded_predictions):
                    # Biasanya label di DARWIN adalah 'P' (Patient) dan 'H' (Healthy)
                    if result == 'P' or result == 1: 
                        st.error(f"Data Pasien {i+1}: **Terindikasi Alzheimer** (Perlu pemeriksaan klinis lebih lanjut)")
                    else:
                        st.success(f"Data Pasien {i+1}: **Sehat** (Tidak terdeteksi anomali)")
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data. Pastikan format dan jumlah kolom CSV sesuai (451 fitur). Error teknis: {e}")