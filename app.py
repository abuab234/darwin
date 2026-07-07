import streamlit as st
import pandas as pd
import numpy as np
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
except Exception as e:
    st.error(f"⚠️ Gagal memuat file model. Pastikan file .pkl berada di folder yang sama dengan app.py. Error: {e}")
    st.stop()

st.markdown("---")

# --- MEMBUAT SISTEM TABS UNTUK NAVIGASI ---
tab1, tab2 = st.tabs(["📁 Upload File CSV", "🎛️ Input Parameter Manual (Simulasi)"])

# ==========================================
# TAB 1: UPLOAD FILE CSV
# ==========================================
with tab1:
    st.subheader("Uji Data Menggunakan File")
    st.info("""
    **Kenapa harus format .CSV?**
    Model Kecerdasan Buatan (AI) ini tidak memproses gambar tulisan tangan (seperti JPG/PNG), melainkan memproses data metrik fisik. File `.csv` (*Comma Separated Values*) ini berisi 451 kolom angka yang mencatat kecepatan goresan, tekanan pena, dan waktu jeda secara presisi dari sensor perangkat digital.
    """)

    st.warning("""
    **Belum memiliki dataset?**
    Jika Anda tidak memiliki data sensor pasien namun ingin mencoba fungsionalitas aplikasi ini, Anda dapat mengunduh beberapa sampel data `.csv` yang telah kami sediakan melalui tautan berikut:
    👉 **[Download Contoh Dataset (Google Drive)](https://drive.google.com/drive/u/0/folders/1JAaMTNAmPYrP7PNkmvt0cwAatb5NCyAJ)**
    """)

    uploaded_file = st.file_uploader("Unggah file CSV metrik tulisan tangan pasien", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("**Preview Data Input:**")
        st.dataframe(input_data.head(5))

        if st.button("Jalankan Prediksi via CSV", type="primary"):
            with st.spinner("Menganalisis pola tulisan tangan..."):
                try:
                    scaled_data = scaler.transform(input_data)
                    predictions = model.predict(scaled_data)
                    decoded_predictions = le.inverse_transform(predictions)
                    
                    st.subheader("📋 Hasil Analisis:")
                    for i, result in enumerate(decoded_predictions):
                        if result == 'P' or result == 1: 
                            st.error(f"Data Pasien {i+1}: **Terindikasi Alzheimer** (Perlu pemeriksaan klinis lebih lanjut)")
                        else:
                            st.success(f"Data Pasien {i+1}: **Sehat** (Tidak terdeteksi anomali)")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses data. Pastikan format dan jumlah kolom CSV sesuai (451 fitur). Error teknis: {e}")

# ==========================================
# TAB 2: INPUT PARAMETER MANUAL (INTERAKTIF)
# ==========================================
with tab2:
    st.subheader("Simulasi Parameter Langsung")
    st.write("""
    Gunakan fitur ini jika Anda ingin menguji perubahan metrik secara spesifik tanpa perlu membuat file CSV. 
    Karena model membutuhkan **451 parameter**, kami telah menyediakan tabel interaktif di bawah ini yang berisi nilai nol (0.0) sebagai *baseline*. Anda bisa mengklik sel (kotak) mana saja di dalam tabel untuk mengubah angkanya secara manual.
    """)

    # 1. Ambil 451 nama kolom dari scaler
    nama_kolom = scaler.feature_names_in_
    
    # 2. Buat DataFrame kosong dengan 1 baris berisi angka 0.0
    df_manual = pd.DataFrame(np.zeros((1, len(nama_kolom))), columns=nama_kolom)
    
    # 3. Tampilkan Data Editor interaktif
    st.caption("Klik angka 0.000 di bawah ini untuk mengubah nilainya. Anda bisa menggeser tabel ke kanan untuk melihat seluruh 451 kolom.")
    edited_df = st.data_editor(df_manual, num_rows="fixed")
    
    if st.button("Jalankan Prediksi Simulasi", type="primary"):
        with st.spinner("Menganalisis parameter manual..."):
            try:
                # Proses data dari tabel yang baru saja diedit pengguna
                scaled_manual = scaler.transform(edited_df)
                pred_manual = model.predict(scaled_manual)
                decoded_manual = le.inverse_transform(pred_manual)[0]
                
                st.subheader("📋 Hasil Analisis Simulasi:")
                if decoded_manual == 'P' or decoded_manual == 1:
                    st.error("Berdasarkan parameter yang Anda masukkan: **Terindikasi Alzheimer** (Perlu pemeriksaan klinis lebih lanjut)")
                else:
                    st.success("Berdasarkan parameter yang Anda masukkan: **Sehat** (Tidak terdeteksi anomali)")
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")
