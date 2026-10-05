"""
SPK Rekomendasi Kampus/Jurusan - Capstone Project (VIKOR Murni)
=============================================================
Versi update sesuai arsitektur:
- Alif: Input Skor Akademik Rapor
- Ghina: Input Skor Kecocokan Minat RIASEC
- Kamu: Modul VIKOR Murni dengan studi kasus Rumpun MIPA & APAP
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SPK Rekomendasi MIPA & APAP - VIKOR", layout="wide")

# ------------------------------------------------------------------
# Konstanta & Konfigurasi Default
# ------------------------------------------------------------------
DEFAULT_AKREDITASI_MAP = {
    "Unggul": 5, "A": 5,
    "Baik Sekali": 4, "B": 4,
    "Baik": 3, "C": 3,
    "Cukup": 2,
    "Belum Terakreditasi": 1,
}

# 5 Kriteria Utama (Mencakup input Alif & Ghina)
KRITERIA_TETAP = [
    "Skor Akademik (Alif)", 
    "Skor Minat RIASEC (Ghina)", 
    "Akreditasi", 
    "Daya Tampung", 
    "Keketatan"
]
TIPE_TETAP = ["benefit", "benefit", "benefit", "benefit", "cost"]  # keketatan cost

# Data Sampel Rumpun MIPA & APAP (Studi Kasus MA Kafila)
SAMPLE_KAMPUS = pd.DataFrame({
    "Nama Kampus": ["UI", "UGM", "UNJ", "UPI", "IPB"],
    "Jurusan": [
        "Matematika", 
        "Statistika", 
        "Pendidikan Matematika", 
        "Pendidikan Biologi", 
        "Statistika dan Data Sains"
    ],
    "Akreditasi": ["Unggul", "Unggul", "Unggul", "Baik Sekali", "Unggul"],
    "Daya Tampung": [60, 50, 80, 75, 45],
    "Keketatan": [4.2, 3.8, 8.5, 9.1, 5.0],
})

REQUIRED_COLS = ["Nama Kampus", "Jurusan", "Akreditasi", "Daya Tampung", "Keketatan"]


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
    R_star, R_minus = R.min(), R.max()

    Q = np.zeros(n_alt)
    for i in range(n_alt):
        sd = (S_minus - S_star) if (S_minus - S_star) != 0 else 1e-9
        rd = (R_minus - R_star) if (R_minus - R_star) != 0 else 1e-9
        Q[i] = v * (S[i] - S_star) / sd + (1 - v) * (R[i] - R_star) / rd

    hasil = pd.DataFrame({"Kampus": data.index, "S": S, "R": R, "Q": Q}) \
        .sort_values("Q").reset_index(drop=True)
    hasil.index = hasil.index + 1
    hasil.insert(0, "Rank", hasil.index)
    return hasil


# ------------------------------------------------------------------
# UI - Header & Sidebar
# ------------------------------------------------------------------
st.title("🎓 SPK Rekomendasi Prodi MIPA & APAP (Capstone Project)")
st.caption("Integrasi: Nilai Rapor (Alif) + Minat RIASEC (Ghina) + SPK VIKOR (Putri)")

with st.sidebar:
    st.header("⚙️ Pengaturan VIKOR")
    v_strategi = st.slider("Bobot strategi mayoritas VIKOR (v)", 0.0, 1.0, 0.5, 0.05,
                           help="v=0.5 konsensus, v>0.5 mayoritas kriteria, v<0.5 veto individu")
    st.divider()
    st.subheader("Mapping Akreditasi → Angka")
    akreditasi_map = {}
    for label, default_val in DEFAULT_AKREDITASI_MAP.items():
        akreditasi_map[label] = st.number_input(label, min_value=1, max_value=5,
                                              value=default_val, key=f"ak_{label}")

with st.expander("ℹ️ Keterangan Integrasi Tim", expanded=False):
    st.markdown("""
    - **Modul Alif:** Memberikan *Skor Akademik Rapor* per jurusan MIPA/APAP.
    - **Modul Ghina:** Memberikan *Skor Kecocokan Minat RIASEC* hasil kuesioner psikotes.
    - **Modul Kamu (Putri):** Mengolah seluruh data dengan *VIKOR Murni* dan menampilkan ranking terbaik.
    """)

st.divider()

# ------------------------------------------------------------------
# TAHAP 1: Database Kampus & Input Simulasi dari Alif & Ghina
# ------------------------------------------------------------------
st.subheader("1️⃣ Database Pilihan Prodi MIPA & APAP")
st.markdown("Berikut adalah data alternatif prodi tujuan (Studi Kasus: MA Kafila).")

df_kampus = SAMPLE_KAMPUS.copy()
df_kampus = st.data_editor(df_kampus, use_container_width=True, num_rows="dynamic", key="editor_kampus")

if "Jurusan" not in df_kampus.columns or df_kampus.empty:
    st.error("⚠️ Data prodi belum lengkap.")
    st.stop()

st.subheader("2️⃣ Input Skor dari Modul Alif & Ghina (Simulasi)")
st.caption("Slider ini mensimulasikan hasil olahan data siswa dari program Alif (nilai rapor) dan Ghina (tes minat RIASEC).")

daftar_label = [f"{row['Jurusan']} - {row['Nama Kampus']}" for _, row in df_kampus.iterrows()]
skor_alif = {}
skor_ghina = {}

st.markdown("---")
for i, label in enumerate(daftar_label):
    st.write(f"**Pilihan {i+1}: {label}**")
    c1, c2 = st.columns(2)
    with c1:
        skor_alif[label] = st.slider(f"Skor Akademik Rapor (Alif)", 0, 100, 75, key=f"alif_{i}")
    with c2:
        skor_ghina[label] = st.slider(f"Skor Minat RIASEC (Ghina)", 0, 100, 80, key=f"ghina_{i}")
    st.markdown("")

# Masukkan ke dataframe
df_kampus["Label_Unik"] = daftar_label
df_kampus["Skor Akademik (Alif)"] = df_kampus["Label_Unik"].map(skor_alif)
df_kampus["Skor Minat RIASEC (Ghina)"] = df_kampus["Label_Unik"].map(skor_ghina)
df_kampus["Akreditasi (angka)"] = df_kampus["Akreditasi"].map(akreditasi_map).fillna(0)

st.write("**Matriks Keputusan Lengkap (Siap Hitung):**")
matriks_view = df_kampus[[
    "Label_Unik", "Skor Akademik (Alif)", "Skor Minat RIASEC (Ghina)", 
    "Akreditasi (angka)", "Daya Tampung", "Keketatan"
]]
st.dataframe(matriks_view, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------
# TAHAP 2: Preferensi Siswa (Bobot VIKOR Murni)
# ------------------------------------------------------------------
st.subheader("3️⃣ Preferensi Kepentingan Kriteria oleh Siswa")
st.caption("Siswa mengatur tingkat kepentingan dari 5 kriteria di bawah ini.")

pref_user = {}
cols_pref = st.columns(len(KRITERIA_TETAP))
for i, krit in enumerate(KRITERIA_TETAP):
    with cols_pref[i]:
        pref_user[krit] = st.slider(krit, 1, 5, 3, key=f"pref_{krit}")

pref_array = np.array([pref_user[k] for k in KRITERIA_TETAP], dtype=float)
bobot_preferensi = pref_array / pref_array.sum()

st.divider()

# ------------------------------------------------------------------
# TAHAP 3: Eksekusi VIKOR Murni & Hasil
# ------------------------------------------------------------------
st.subheader("4️⃣ Hasil Rekomendasi Peringkat Prodi")

def siapkan_matriks_vikor(df):
    data_vikor = df.set_index("Label_Unik")[
        ["Skor Akademik (Alif)", "Skor Minat RIASEC (Ghina)", "Akreditasi (angka)", "Daya Tampung", "Keketatan"]
    ]
    data_vikor.columns = KRITERIA_TETAP
    return data_vikor

if st.button("🚀 Hitung Rekomendasi Peringkat", type="primary"):
    data_vikor = siapkan_matriks_vikor(df_kampus)
    hasil = hitung_vikor(data_vikor, bobot_preferensi, TIPE_TETAP, v=v_strategi)

    st.write("**Tabel Ranking Pilihan Prodi (Nilai Q terkecil = Terbaik):**")
    st.dataframe(hasil.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    st.bar_chart(hasil.set_index("Kampus")["Q"])

    top3 = hasil.head(3)["Kampus"].tolist()
    ranked_text = ", ".join([f"{i+1}. {nama}" for i, nama in enumerate(top3)])
    st.success(f"🏆 Rekomendasi Utama untuk Siswa: {ranked_text}")

    csv_out = hasil.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Hasil Ranking (CSV)", csv_out, "hasil_rekomendasi_cp.csv", "text/csv")
else:
    st.info("Atur database dan preferensi di atas, lalu klik **Hitung Rekomendasi Peringkat**.")
