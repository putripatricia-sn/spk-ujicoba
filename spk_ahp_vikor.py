"""
SPK Rekomendasi Kampus/Jurusan - Capstone Project (VIKOR Murni)
Clean Version (Tanpa emoji, membaca langsung data_jurusan_ptn.csv)
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SPK Rekomendasi PTN", layout="wide")

# ------------------------------------------------------------------
# Fungsi Inti Algoritma VIKOR
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
    R_star, R_minus = R.max(), R.min()

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
# Load Database Master secara Otomatis dari Folder yang Sama
# ------------------------------------------------------------------
@st.cache_data
def load_database():
    try:
        df = pd.read_csv("data_jurusan_ptn.csv")
        df["akreditasi_angka"] = df["akreditasi"].apply(map_akreditasi)
        df["Label_Prodi"] = df["jurusan"] + " — " + df["nama_ptn"]
        return df
    except Exception as e:
        return None

df_raw = load_database()

# ------------------------------------------------------------------
# Antarmuka Aplikasi (UI)
# ------------------------------------------------------------------
st.title("Sistem Pendukung Keputusan Pemilihan PTN")
st.caption("Capstone Project - Metode VIKOR Murni")

if df_raw is None:
    st.error("File database 'data_jurusan_ptn.csv' tidak ditemukan pada direktori yang sama. Pastikan file berada di dalam folder Tugas Akhir.")
    st.stop()

with st.sidebar:
    st.header("Pengaturan Sistem")
    v_strategi = st.slider("Parameter Strategi Mayoritas (v)", 0.0, 1.0, 0.5, 0.05)
    st.divider()
    st.subheader("Informasi Tim")
    st.text("Modul SPK: VIKOR Murni")
    st.text("Integrasi: Alif (ML) & Ghina (RIASEC)")

st.divider()

st.subheader("1. Pilih Program Studi Tujuan")
pilihan_prodi = st.multiselect(
    "Daftar alternatif prodi dari database master:",
    options=df_raw["Label_Prodi"].tolist(),
    default=df_raw["Label_Prodi"].tolist()[:5]
)

if not pilihan_prodi:
    st.warning("Silakan pilih minimal satu program studi untuk melanjutkan.")
    st.stop()

df_filtered = df_raw[df_raw["Label_Prodi"].isin(pilihan_prodi)].copy()

st.subheader("2. Input Skor Integrasi (Simulasi Modul Alif & Ghina)")
skor_alif_dict = {}
skor_ghina_dict = {}

col1, col2 = st.columns(2)
for idx, row in df_filtered.iterrows():
    label = row["Label_Prodi"]
    with col1:
        skor_alif_dict[label] = st.slider(f"Skor Rapor (Alif) - {label}", 0, 100, 75, key=f"alif_{idx}")
    with col2:
        skor_ghina_dict[label] = st.slider(f"Skor RIASEC (Ghina) - {label}", 0, 100, 80, key=f"ghina_{idx}")
    st.markdown("---")

df_filtered["Skor Akademik (Alif)"] = df_filtered["Label_Prodi"].map(skor_alif_dict)
df_filtered["Skor Minat RIASEC (Ghina)"] = df_filtered["Label_Prodi"].map(skor_ghina_dict)

KRITERIA_TETAP = [
    "Skor Akademik (Alif)", 
    "Skor Minat RIASEC (Ghina)", 
    "Akreditasi", 
    "Daya Tampung", 
    "Keketatan"
]
TIPE_TETAP = ["benefit", "benefit", "benefit", "benefit", "cost"]

matriks_vikor_data = pd.DataFrame({
    "Skor Akademik (Alif)": df_filtered["Skor Akademik (Alif)"].values,
    "Skor Minat RIASEC (Ghina)": df_filtered["Skor Minat RIASEC (Ghina)"].values,
    "Akreditasi": df_filtered["akreditasi_angka"].values,
    "Daya Tampung": df_filtered["daya_tampung"].values,
    "Keketatan": df_filtered["c4_keketatan"].values,
}, index=df_filtered["Label_Prodi"].values)

st.subheader("3. Preferensi Bobot Kriteria Siswa")
pref_user = {}
cols_pref = st.columns(len(KRITERIA_TETAP))
for i, krit in enumerate(KRITERIA_TETAP):
    with cols_pref[i]:
        pref_user[krit] = st.slider(krit, 1, 5, 3, key=f"pref_{krit}")

pref_array = np.array([pref_user[k] for k in KRITERIA_TETAP], dtype=float)
bobot_preferensi = pref_array / pref_array.sum()

st.divider()

st.subheader("4. Hasil Perankingan Rekomendasi")
if st.button("Jalankan Perhitungan", type="primary"):
    hasil = hitung_vikor(matriks_vikor_data, bobot_preferensi, TIPE_TETAP, v=v_strategi)
    
    st.dataframe(hasil.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    
    st.bar_chart(hasil.set_index("Alternatif")["Q"])
    
    top_prodi = hasil.head(1)["Alternatif"].values[0]
    st.success(f"Rekomendasi Utama: {top_prodi}")
    
    csv_out = hasil.to_csv(index=False).encode("utf-8")
    st.download_button("Unduh Laporan CSV", csv_out, "hasil_rekomendasi.csv", "text/csv")
else:
    st.info("Lengkapi parameter di atas, lalu klik tombol Jalankan Perhitungan.")
