"""
SPK Rekomendasi Kampus/Jurusan - Capstone Project (VIKOR Murni)
=============================================================
Versi khusus untuk Capstone Project kelompok:
- Menggunakan metode VIKOR Murni.
- Bobot kriteria diatur langsung oleh preferensi siswa (slider/ROC).
- Tanpa modul AHP di aplikasi utama (agar bersih dan siap demo).

Alur data:
  Skor ML (input Alif) -> Database Kampus -> Matriks Keputusan
  -> Bobot Preferensi -> VIKOR -> Ranking
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SPK Rekomendasi Kampus - VIKOR", layout="wide")

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

KRITERIA_TETAP = ["Skor Kecocokan (ML)", "Akreditasi", "Daya Tampung", "Keketatan"]
TIPE_TETAP = ["benefit", "benefit", "benefit", "cost"]  # keketatan tinggi = makin sulit diterima -> cost

SAMPLE_KAMPUS = pd.DataFrame({
    "Nama Kampus": ["UGM", "UI", "UB", "ITB", "UNAIR"],
    "Jurusan": ["Teknik Informatika"] * 5,
    "Akreditasi": ["Unggul", "Unggul", "Baik Sekali", "Unggul", "Baik Sekali"],
    "Daya Tampung": [120, 100, 150, 90, 110],
    "Keketatan": [8.5, 9.2, 5.1, 9.8, 6.3],
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
st.title("🎓 SPK Rekomendasi Kampus & Jurusan (Capstone Project)")
st.caption("Alur: Skor kecocokan dari model ML (Alif) → Database kampus → Preferensi Siswa → VIKOR → Ranking")

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
    - **Modul Alif (Machine Learning):** Menghasilkan skor kecocokan akademik per jurusan.
    - **Modul Ghina (Clustering RIASEC):** Menyaring/filter jurusan yang sesuai minat kepribadian siswa.
    - **Modul Kamu (VIKOR):** Mengolah data akhir, menerapkan preferensi siswa, dan meranking PTN terbaik.
    """)

st.divider()

# ------------------------------------------------------------------
# TAHAP 1: Database Kampus + Skor ML dari Alif
# ------------------------------------------------------------------
st.subheader("1️⃣ Database Kampus & Jurusan")
st.caption("Upload database kampus (.csv) atau sesuaikan data pada tabel di bawah.")

col_up, col_dl = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader("Upload database kampus (.csv)", type=["csv"])
with col_dl:
    template_csv = SAMPLE_KAMPUS.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Template CSV", template_csv, "template_database_kampus.csv", "text/csv")

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        st.warning(f"⚠️ Kolom tidak lengkap: {missing}. Petakan manual di bawah:")
        col_map = {}
        map_cols = st.columns(len(REQUIRED_COLS))
        for i, req_col in enumerate(REQUIRED_COLS):
            with map_cols[i]:
                pilihan_default = req_col if req_col in df_raw.columns else df_raw.columns[0]
                col_map[req_col] = st.selectbox(f"'{req_col}'", options=list(df_raw.columns),
                                                index=list(df_raw.columns).index(pilihan_default), key=f"map_{req_col}")
        df_kampus = pd.DataFrame({req_col: df_raw[col_map[req_col]] for req_col in REQUIRED_COLS})
    else:
        df_kampus = df_raw[REQUIRED_COLS].copy()
else:
    df_kampus = SAMPLE_KAMPUS.copy()

df_kampus = st.data_editor(df_kampus, use_container_width=True, num_rows="dynamic", key="editor_kampus")

if "Jurusan" not in df_kampus.columns or df_kampus.empty:
    st.error("⚠️️ Data kampus belum lengkap / kolom 'Jurusan' tidak ditemukan.")
    st.stop()

st.subheader("2️⃣ Skor Kecocokan Akademik (Output Simulasi ML dari Alif)")
daftar_jurusan = sorted(df_kampus["Jurusan"].dropna().unique().tolist())
skor_ml = {}
cols_ml = st.columns(min(len(daftar_jurusan), 4) or 1)
for i, jurusan in enumerate(daftar_jurusan):
    with cols_ml[i % 4]:
        skor_ml[jurusan] = st.slider(f"Skor: {jurusan}", 0, 100, 75, key=f"ml_{jurusan}")

df_kampus["Skor Kecocokan (ML)"] = df_kampus["Jurusan"].map(skor_ml)
df_kampus["Akreditasi (angka)"] = df_kampus["Akreditasi"].map(akreditasi_map).fillna(0)

st.write("**Matriks Keputusan Siap Hitung:**")
matriks_view = df_kampus[["Nama Kampus", "Jurusan", "Skor Kecocokan (ML)",
                          "Akreditasi (angka)", "Daya Tampung", "Keketatan"]]
st.dataframe(matriks_view, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------
# TAHAP 2: Preferensi Siswa (Bobot VIKOR Murni)
# ------------------------------------------------------------------
st.subheader("3️⃣ Preferensi Kepentingan Kriteria oleh Siswa")
st.caption("Siswa menggeser tingkat kepentingan tiap kriteria sesuai keinginan pribadinya.")

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
st.subheader("4️⃣ Hasil Rekomendasi Kampus")

def siapkan_matriks_vikor(df):
    df = df.copy()
    df["Label"] = df["Nama Kampus"].astype(str) + " — " + df["Jurusan"].astype(str)
    data_vikor = df.set_index("Label")[
        ["Skor Kecocokan (ML)", "Akreditasi (angka)", "Daya Tampung", "Keketatan"]
    ]
    data_vikor.columns = KRITERIA_TETAP
    return data_vikor

if st.button("🚀 Hitung Rekomendasi Kampus", type="primary"):
    data_vikor = siapkan_matriks_vikor(df_kampus)
    hasil = hitung_vikor(data_vikor, bobot_preferensi, TIPE_TETAP, v=v_strategi)

    st.write("**Tabel Ranking Pilihan Kampus (Nilai Q terkecil = Terbaik):**")
    st.dataframe(hasil.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    st.bar_chart(hasil.set_index("Kampus")["Q"])

    top3 = hasil.head(3)["Kampus"].tolist()
    ranked_text = ", ".join([f"{i+1}. {nama}" for i, nama in enumerate(top3)])
    st.success(f"🏆 Rekomendasi Kampus Utama: {ranked_text}")

    csv_out = hasil.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Hasil Ranking (CSV)", csv_out, "hasil_rekomendasi_cp.csv", "text/csv")
else:
    st.info("Atur database dan preferensi di atas, lalu klik **Hitung Rekomendasi Kampus**.")
