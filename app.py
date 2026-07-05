import streamlit as st
import pandas as pd
import pickle

# Konfigurasi Halaman (Modern UI)
st.set_page_config(page_title="Deteksi Dini Alzheimer", layout="centered", page_icon="🧠")

st.title("🧠 Deteksi Dini Alzheimer")
st.caption("Berbasis Machine Learning menggunakan Dataset DARWIN")

# --- BAGIAN 1: PENJELASAN APLIKASI & PENYAKIT ---
with st.expander("📖 Informasi Singkat: Penyakit Alzheimer & DARWIN Dataset", expanded=True):
    st.write("""
    **Apa itu Penyakit Alzheimer?**
    Alzheimer adalah penyakit degeneratif pada otak yang secara bertahap menghancurkan daya ingat dan kemampuan kognitif. Jauh sebelum gejala memori (pikun) terlihat parah, kerusakan saraf ini terlebih dahulu memengaruhi kemampuan motorik halus penderitanya, seperti koordinasi pergerakan tangan saat menulis.

    **Apa itu DARWIN Dataset?**
    DARWIN adalah kumpulan data hasil penelitian medis. Data ini merekam pergerakan tangan ratusan orang (sehat dan pasien Alzheimer) saat mereka menulis atau menggambar di atas *pen tablet* digital. 

    **Tujuan Pembuatan Aplikasi Ini**
    Aplikasi ini dirancang untuk melakukan *screening* atau deteksi dini Alzheimer secara cepat dan murah. Algoritma *Bagging Classifier* akan membaca 451 metrik pergerakan tangan Anda untuk mencari pola anomali yang mengindikasikan gejala awal Alzheimer, tanpa memerlukan pemindaian otak (MRI) yang mahal.
    """)

# Cache resource agar model tidak di-load ulang setiap kali ada interaksi UI
@st.cache_resource
def load_components():
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
    st.success("✅ Model Machine Learning, Scaler, dan Encoder siap digunakan!")
except Exception as e:
    st.error(f"⚠️ Gagal memuat file model. Pastikan file .pkl berada di folder yang sama dengan app.py. Error: {e}")
    st.stop()

st.markdown("---")
st.subheader("Uji Data Pasien")

# --- BAGIAN 2: PENJELASAN CSV & LINK GOOGLE DRIVE ---
st.info("""
**Kenapa harus format .CSV?**
Model Kecerdasan Buatan (AI) ini tidak memproses gambar tulisan tangan (seperti JPG/PNG), melainkan memproses data metrik fisik. File `.csv` (*Comma Separated Values*) ini berisi 451 kolom angka yang mencatat kecepatan goresan, tekanan pena, dan waktu jeda secara presisi dari sensor perangkat digital.
""")

st.warning("""
**Belum memiliki dataset?**
Jika Anda tidak memiliki data sensor pasien namun ingin mencoba fungsionalitas aplikasi ini, Anda dapat mengunduh beberapa sampel data `.csv` yang telah kami sediakan melalui tautan berikut:
👉 **[Download Contoh Dataset (Google Drive)](https://drive.google.com/drive/u/0/folders/1JAaMTNAmPYrP7PNkmvt0cwAatb5NCyAJ)**
""")

# Fitur Upload File
uploaded_file = st.file_uploader("Unggah file CSV metrik tulisan tangan pasien", type=["csv"])

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
