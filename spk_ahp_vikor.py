"""
SPK Rekomendasi Kampus/Jurusan - Capstone Project (VIKOR Murni + CSV Otomatis)
=============================================================================
- Membaca file database 'data_jurusan_ptn.csv' secara otomatis.
- Lintas rumpun (MIPA, IPS, Saintek, Soshum bebas).
- Integrasi: Skor Akademik (Alif) + Skor Minat RIASEC (Ghina) + VIKOR Murni (Putri).
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SPK Rekomendasi PTN - VIKOR", layout="wide")

# ------------------------------------------------------------------
# 1. Load Data CSV Otomatis
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data_jurusan_ptn.csv")
        return df
    except Exception as e:
        return None

df_raw = load_data()

if df_raw is None:
    st.error("⚠️ File 'data_jurusan_ptn.csv' tidak ditemukan di direktori yang sama dengan script ini. Pastikan file CSV sudah di-upload!")
    st.stop()

# ------------------------------------------------------------------
# Fungsi Inti VIKOR Murni
# ------------------------------------------------------------------
def hitung_vikor(data, bobot, tipe, v=0.5):
    X = data.values.astype(float)
    n_alt, n_krit = X.shape

    f_star = np.zeros(n_krit)
    f_minus = np.zeros(n_krit)
    for j in range(n_krit):
        if tipe[j] == "benefit":
            f_star[j] = X[:, j].max()
            f_minus[j] = X[:, j].min()
        else:
            f_star[j] = X[:, j].min()
            f_minus[j] = X[:, j].max()

    S = np.zeros(n_alt)
    R = np.zeros(n_alt)
    for i in range(n_alt):
        total, maxval = 0, -np.inf
        for j in range(n_krit):
            denom = (f_star[j] - f_minus[j])
            d = 0 if denom == 0 else bobot[j] * (f_star[j] - X[i, j]) / denom
            total += d
            maxval = max(maxval, d)
        S[i], R[i] = total, maxval

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.max(), R.min() # Sesuai definisi cost/benefit VIKOR

    Q = np.zeros(n_alt)
    for i in range(n_alt):
        sd = (S_minus - S_star) if (S_minus - S_star) != 0 else 1e-9
        rd = (R_minus - R_star) if (R_minus - R_star) != 0 else 1e-9
        Q[i] = v * (S[i] - S_star) / sd + (1 - v) * (R[i] - R_star) / rd

    hasil = pd.DataFrame({"Alternatif": data.index, "S": S, "R": R, "Q": Q}) \
        .sort_values("Q").reset_index(drop=True)
    hasil.index = hasil.index + 1
    hasil.insert(0, "Rank", hasil.index)
    return hasil

# Mapping nilai akreditasi teks ke angka
def map_akreditasi(val):
    v = str(val).strip().lower()
    if "unggul" in v or "a" == v:
        return 5
    elif "sekali" in v or "b" == v:
        return 4
    elif "baik" in v or "c" == v:
        return 3
    elif "cukup" in v:
        return 2
    else:
        return 1

# ------------------------------------------------------------------
# UI - Header & Sidebar
# ------------------------------------------------------------------
st.title("🎓 SPK Rekomendasi Kampus & Jurusan (Capstone Project)")
st.caption("Sistem Pendukung Keputusan Pemilihan PTN menggunakan Metode VIKOR Murni")

with st.sidebar:
    st.header("⚙️ Pengaturan VIKOR")
    v_strategi = st.slider("Bobot strategi mayoritas (v)", 0.0, 1.0, 0.5, 0.05,
                           help="v=0.5 konsensus seimbang antara utilitas kelompok dan penyesalan individual.")
    st.divider()
    st.markdown("### 👥 Tim Capstone Project:")
    id_tim = st.selectbox("Simulasi Data Siswa:", ["Siswa A (Lintas Rumpun Bebas)"])

st.divider()

# ------------------------------------------------------------------
# TAHAP 1: Filter / Pilih Database dari CSV
# ------------------------------------------------------------------
st.subheader("1️⃣ Pilih Daftar Prodi/Kampus dari Database CSV")
st.markdown("Database memuat data resmi PTN (MIPA, Soshum, Saintek lintas rumpun). Pilih beberapa prodi untuk diuji.")

# Tambah kolom akreditasi angka
df_raw["akreditasi_angka"] = df_raw["akreditasi"].apply(map_akreditasi)

# Buat label unik gabungan PTN dan Jurusan
df_raw["Label_Prodi"] = df_raw["jurusan"] + " — " + df_raw["nama_ptn"]

# Multi-select pilihan prodi oleh user/siswa
pilihan_prodi = st.multiselect(
    "Pilih program studi yang ingin dievaluasi untuk siswa:",
    options=df_raw["Label_Prodi"].tolist(),
    default=df_raw["Label_Prodi"].tolist()[:5]  # Default ambil 5 teratas biar tidak kepanjangan
)

if not pilihan_prodi:
    st.warning("⚠️️ Silakan pilih minimal 1 program studi di atas untuk melanjutkan.")
    st.stop()

# Filter dataframe berdasarkan pilihan
df_filtered = df_raw[df_raw["Label_Prodi"].isin(pilihan_prodi)].copy()

st.divider()

# ------------------------------------------------------------------
# TAHAP 2: Input Simulasi Skor dari Alif & Ghina
# ------------------------------------------------------------------
st.subheader("2️⃣ Masukkan Skor Integrasi (Alif & Ghina)")
st.caption("Mensimulasikan output skor rapor dari Machine Learning (Alif) dan skor kecocokan minat RIASEC (Ghina) untuk setiap prodi yang dipilih.")

skor_alif_dict = {}
skor_ghina_dict = {}

cols_input = st.columns(2)
for idx, row in df_filtered.iterrows():
    label = row["Label_Prodi"]
    with cols_input[0]:
        skor_alif_dict[label] = st.slider(f"Skor Akademik Rapor (Alif) | {label}", 0, 100, 75, key=f"alif_{idx}")
    with cols_input[1]:
        skor_ghina_dict[label] = st.slider(f"Skor Minat RIASEC (Ghina) | {label}", 0, 100, 80, key=f"ghina_{idx}")
    st.markdown("---")

df_filtered["Skor Akademik (Alif)"] = df_filtered["Label_Prodi"].map(skor_alif_dict)
df_filtered["Skor Minat RIASEC (Ghina)"] = df_filtered["Label_Prodi"].map(skor_ghina_dict)

# Kriteria tetap VIKOR
KRITERIA_TETAP = [
    "Skor Akademik (Alif)", 
    "Skor Minat RIASEC (Ghina)", 
    "Akreditasi", 
    "Daya Tampung", 
    "Keketatan"
]
# Benefit: Skor Alif, Skor Ghina, Akreditasi, Daya Tampung | Cost: Keketatan (karena makin kecil makin sulit)
TIPE_TETAP = ["benefit", "benefit", "benefit", "benefit", "cost"]

# Siapkan dataframe untuk matriks keputusan
matriks_vikor_data = pd.DataFrame({
    "Skor Akademik (Alif)": df_filtered["Skor Akademik (Alif)"].values,
    "Skor Minat RIASEC (Ghina)": df_filtered["Skor Minat RIASEC (Ghina)"].values,
    "Akreditasi": df_filtered["akreditasi_angka"].values,
    "Daya Tampung": df_filtered["daya_tampung"].values,
    "Keketatan": df_filtered["c4_keketatan"].values,
}, index=df_filtered["Label_Prodi"].values)

st.write("**Matriks Keputusan Final (Data CSV + Input Tim):**")
st.dataframe(matriks_vikor_data, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# TAHAP 3: Preferensi Siswa (Bobot VIKOR Murni)
# ------------------------------------------------------------------
st.subheader("3️⃣ Atur Preferensi Bobot Kriteria oleh Siswa")
st.caption("Geser slider untuk menentukan seberapa penting masing-masing kriteria bagi siswa.")

pref_user = {}
cols_pref = st.columns(len(KRITERIA_TETAP))
for i, krit in enumerate(KRITERIA_TETAP):
    with cols_pref[i]:
        pref_user[krit] = st.slider(krit, 1, 5, 3, key=f"pref_{krit}")

pref_array = np.array([pref_user[k] for k in KRITERIA_TETAP], dtype=float)
bobot_preferensi = pref_array / pref_array.sum()

st.divider()

# ------------------------------------------------------------------
# TAHAP 4: Perhitungan VIKOR Murni & Hasil Rekomendasi
# ------------------------------------------------------------------
st.subheader("4️⃣ Hasil Rekomendasi Peringkat Prodi")

if st.button("🚀 Jalankan Perhitungan VIKOR", type="primary"):
    hasil = hitung_vikor(matriks_vikor_data, bobot_preferensi, TIPE_TETAP, v=v_strategi)

    st.write("**Tabel Peringkat Rekomendasi (Nilai Q terkecil = Pilihan Paling Ideal):**")
    st.dataframe(hasil.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    
    st.bar_chart(hasil.set_index("Alternatif")["Q"])

    top_kampus = hasil.head(3)["Alternatif"].tolist()
    ranked_text = ", ".join([f"{i+1}. {nama}" for i, nama in enumerate(top_kampus)])
    st.success(f"🏆 Rekomendasi Utama untuk Siswa: {ranked_text}")

    csv_out = hasil.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Hasil Ranking (CSV)", csv_out, "hasil_rekomendasi_cp.csv", "text/csv")
else:
    st.info("Pilih prodi, sesuaikan skor dan preferensi di atas, lalu klik **Jalankan Perhitungan VIKOR**.")
