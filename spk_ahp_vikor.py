"""
SPK Rekomendasi Kampus/Jurusan - AHP + Preferensi User + VIKOR
Mengikuti alur: Skor ML (input) -> Database Kampus -> Matriks Keputusan
                -> Bobot (AHP pakar x Slider preferensi user) -> VIKOR -> Ranking
Jalankan dengan: streamlit run spk_ahp_vikor.py
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SPK Rekomendasi Kampus - AHP & VIKOR", layout="wide")

# ------------------------------------------------------------------
# Konstanta
# ------------------------------------------------------------------
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

SAATY_SCALE = {
    "9 : Mutlak lebih penting": 9, "7 : Sangat lebih penting": 7,
    "5 : Lebih penting": 5, "3 : Sedikit lebih penting": 3,
    "1 : Sama penting": 1,
    "1/3 : Sedikit kurang penting": 1/3, "1/5 : Kurang penting": 1/5,
    "1/7 : Sangat kurang penting": 1/7, "1/9 : Mutlak kurang penting": 1/9,
}

# Mapping akreditasi huruf/predikat -> angka (bisa diedit user di sidebar)
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
    "Keketatan": [8.5, 9.2, 5.1, 9.8, 6.3],  # contoh: rasio pendaftar/kuota, makin tinggi makin ketat
})

# ------------------------------------------------------------------
# Fungsi AHP
# ------------------------------------------------------------------
def hitung_ahp(matriks):
    n = matriks.shape[0]
    kolom_sum = matriks.sum(axis=0)
    norm = matriks / kolom_sum
    bobot = norm.mean(axis=1)
    lambda_vec = matriks.dot(bobot) / bobot
    lambda_max = lambda_vec.mean()
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI = RI_TABLE.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0
    return bobot, lambda_max, CI, CR

# ------------------------------------------------------------------
# Fungsi VIKOR
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
    return hasil

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🎓 SPK Rekomendasi Kampus & Jurusan")
st.caption("Alur: Skor kecocokan dari model ML \u2192 Database kampus \u2192 Bobot (AHP pakar \u00d7 preferensi user) \u2192 VIKOR \u2192 Ranking")

with st.sidebar:
    st.header("⚙️ Pengaturan Umum")
    v_strategi = st.slider("Bobot strategi mayoritas VIKOR (v)", 0.0, 1.0, 0.5, 0.05,
                            help="v=0.5 konsensus, v>0.5 mayoritas kriteria, v<0.5 veto individu")
    st.divider()
    st.subheader("Mapping Akreditasi \u2192 Angka")
    st.caption("Sesuaikan kalau predikat akreditasi kampusmu beda.")
    akreditasi_map = {}
    for label, default_val in DEFAULT_AKREDITASI_MAP.items():
        akreditasi_map[label] = st.number_input(label, min_value=1, max_value=5,
                                                  value=default_val, key=f"ak_{label}")

with st.expander("ℹ️ Tahap 1-2: Input User & Model ML (di luar modul ini)", expanded=False):
    st.markdown("""
    Tahap ini dikerjakan di bagian lain sistem:
    - **Tahap 1 (Frontend):** siswa input nilai rapor, kuesioner RIASEC, dan slider preferensi.
    - **Tahap 2 (Model Alif & Ghina):** nilai rapor diproses model klasifikasi ML \u2192 skor kecocokan per jurusan;
      kuesioner diproses K-Means \u2192 cluster kepribadian.

    Modul di bawah ini menerima **skor kecocokan (output ML)** sebagai input, lalu menjalankan Tahap 3-5.
    """)

st.divider()

# ------------------------------------------------------------------
# TAHAP 3: Database Kampus + Skor ML
# ------------------------------------------------------------------
st.subheader("1️⃣ Database Kampus")
st.caption("Upload CSV database kampus (kolom: Nama Kampus, Jurusan, Akreditasi, Daya Tampung, Keketatan), atau edit tabel contoh di bawah.")

col_up, col_dl = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader("Upload database kampus (.csv)", type=["csv"])
with col_dl:
    template_csv = SAMPLE_KAMPUS.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Template CSV", template_csv, "template_database_kampus.csv", "text/csv")

REQUIRED_COLS = ["Nama Kampus", "Jurusan", "Akreditasi", "Daya Tampung", "Keketatan"]

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)

    st.write("**Kolom yang terbaca di CSV kamu:**", list(df_raw.columns))

    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        st.warning(f"⚠️ Nama kolom CSV kamu tidak sama persis dengan yang dibutuhkan: {missing}. "
                   f"Petakan manual di bawah supaya sistem tahu kolom mana yang mana.")
        col_map = {}
        map_cols = st.columns(len(REQUIRED_COLS))
        for i, req_col in enumerate(REQUIRED_COLS):
            with map_cols[i]:
                pilihan_default = req_col if req_col in df_raw.columns else df_raw.columns[0]
                col_map[req_col] = st.selectbox(
                    f"Kolom untuk '{req_col}'",
                    options=list(df_raw.columns),
                    index=list(df_raw.columns).index(pilihan_default),
                    key=f"map_{req_col}"
                )
        df_kampus = pd.DataFrame({req_col: df_raw[col_map[req_col]] for req_col in REQUIRED_COLS})
    else:
        df_kampus = df_raw[REQUIRED_COLS].copy()
else:
    df_kampus = SAMPLE_KAMPUS.copy()

df_kampus = st.data_editor(df_kampus, use_container_width=True, num_rows="dynamic",
                            key="editor_kampus")

if "Jurusan" not in df_kampus.columns or df_kampus.empty:
    st.error("⚠️ Data kampus belum lengkap / kolom 'Jurusan' tidak ditemukan. Cek pemetaan kolom di atas.")
    st.stop()

st.subheader("2️⃣ Skor Kecocokan dari Model ML (per Jurusan)")
st.caption("Ini output dari model ML Alif. Input manual dulu di sini sebagai simulasi, sebelum nanti disambungkan otomatis ke model.")

daftar_jurusan = sorted(df_kampus["Jurusan"].dropna().unique().tolist())
skor_ml = {}
cols_ml = st.columns(min(len(daftar_jurusan), 4) or 1)
for i, jurusan in enumerate(daftar_jurusan):
    with cols_ml[i % 4]:
        skor_ml[jurusan] = st.slider(f"Skor: {jurusan}", 0, 100, 75, key=f"ml_{jurusan}")

# suntikkan skor ML ke matriks kampus sebagai kolom Kriteria 1
df_kampus["Skor Kecocokan (ML)"] = df_kampus["Jurusan"].map(skor_ml)

# konversi akreditasi huruf/predikat -> angka
df_kampus["Akreditasi (angka)"] = df_kampus["Akreditasi"].map(akreditasi_map)
if df_kampus["Akreditasi (angka)"].isna().any():
    st.warning("⚠️ Ada nilai Akreditasi yang tidak dikenali mapping-nya (cek sidebar) — akan dianggap 0.")
    df_kampus["Akreditasi (angka)"] = df_kampus["Akreditasi (angka)"].fillna(0)

st.write("**Matriks Keputusan (setelah skor ML & akreditasi disuntikkan):**")
matriks_view = df_kampus[["Nama Kampus", "Jurusan", "Skor Kecocokan (ML)",
                           "Akreditasi (angka)", "Daya Tampung", "Keketatan"]]
st.dataframe(matriks_view, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------
# TAHAP 4a: AHP bobot pakar
# ------------------------------------------------------------------
st.subheader("3️⃣ AHP — Bobot Kriteria dari Pakar")
st.caption("Perbandingan berpasangan antar 4 kriteria tetap: Skor Kecocokan, Akreditasi, Daya Tampung, Keketatan.")

n_k = len(KRITERIA_TETAP)
if "ahp_matrix2" not in st.session_state:
    st.session_state.ahp_matrix2 = np.ones((n_k, n_k))
matriks = st.session_state.ahp_matrix2.copy()

for i in range(n_k):
    row_cols = st.columns(n_k - i - 1) if i < n_k - 1 else []
    idx = 0
    for j in range(i + 1, n_k):
        with row_cols[idx]:
            label = f"{KRITERIA_TETAP[i]} vs {KRITERIA_TETAP[j]}"
            pilihan = st.selectbox(label, list(SAATY_SCALE.keys()), index=4, key=f"ahp2_{i}_{j}")
            nilai = SAATY_SCALE[pilihan]
            matriks[i, j] = nilai
            matriks[j, i] = 1 / nilai
        idx += 1
st.session_state.ahp_matrix2 = matriks

bobot_ahp, lambda_max, CI, CR = hitung_ahp(matriks)

col_a, col_b = st.columns(2)
with col_a:
    df_bobot = pd.DataFrame({"Kriteria": KRITERIA_TETAP, "Bobot AHP": bobot_ahp.round(4)})
    st.dataframe(df_bobot, use_container_width=True, hide_index=True)
with col_b:
    st.metric("Consistency Ratio (CR)", f"{CR:.4f}")
    if CR < 0.1:
        st.success("✅ Konsisten, bobot pakar layak dipakai.")
    else:
        st.error("⚠️ CR ≥ 0.1 — perbaiki perbandingan di atas.")

st.divider()

# ------------------------------------------------------------------
# TAHAP 4b: Preferensi siswa (Ranking/ROC ATAU Slider) dikombinasikan dengan bobot AHP
# ------------------------------------------------------------------
st.subheader("4️⃣ Preferensi Siswa — dikombinasikan dengan Bobot Pakar")

metode_pref = st.radio(
    "Cara siswa mengisi preferensi:",
    ["Ranking urutan prioritas (drag-order + ROC)", "Slider bebas 1-5 (versi lama)"],
    horizontal=True,
)

if metode_pref.startswith("Ranking"):
    st.caption("Susun urutan kriteria dari yang PALING PENTING ke PALING KURANG PENTING "
               "(klik kriteria satu per satu sesuai urutan prioritas kamu).")

    urutan = st.multiselect(
        "Urutan prioritas (klik satu-satu, dari paling penting dulu):",
        options=KRITERIA_TETAP,
        default=[],
        key="urutan_prioritas",
    )

    if len(urutan) < n_k:
        st.warning(f"⚠️ Pilih semua {n_k} kriteria secara berurutan untuk melanjutkan "
                   f"(baru {len(urutan)}/{n_k} dipilih).")
        st.stop()

    # Hitung bobot pakai ROC (Rank Order Centroid)
    # rank 1 = paling penting -> bobot terbesar
    n = n_k
    roc_weights = {}
    for rank_pos, krit in enumerate(urutan, start=1):
        w = sum(1.0 / k for k in range(rank_pos, n + 1)) / n
        roc_weights[krit] = w

    pref_array = np.array([roc_weights[k] for k in KRITERIA_TETAP], dtype=float)

    df_urutan = pd.DataFrame({
        "Urutan": list(range(1, n_k + 1)),
        "Kriteria": urutan,
        "Bobot ROC": [round(roc_weights[k], 4) for k in urutan],
    })
    st.dataframe(df_urutan, use_container_width=True, hide_index=True)
    st.caption("Bobot ROC dihitung otomatis dari urutan: makin di atas, bobotnya makin besar "
               "(rumus Rank Order Centroid), tanpa siswa perlu mengisi angka sama sekali.")

else:
    st.caption("Siswa menggeser tingkat kepentingan tiap kriteria menurut prioritas pribadinya "
               "(mis. cari aman \u2192 geser Daya Tampung tinggi).")
    pref_user = {}
    cols_pref = st.columns(n_k)
    for i, krit in enumerate(KRITERIA_TETAP):
        with cols_pref[i]:
            pref_user[krit] = st.slider(krit, 1, 5, 3, key=f"pref_{krit}")
    pref_array = np.array([pref_user[k] for k in KRITERIA_TETAP], dtype=float)

alpha = st.slider("Porsi pengaruh Preferensi Siswa vs Bobot Pakar (AHP)", 0.0, 1.0, 0.5, 0.05,
                   help="0 = 100% pakai bobot AHP pakar, 1 = 100% pakai preferensi siswa")

bobot_final = (1 - alpha) * bobot_ahp + alpha * (pref_array / pref_array.sum())
bobot_final = bobot_final / bobot_final.sum()  # normalisasi ulang

df_bobot_final = pd.DataFrame({"Kriteria": KRITERIA_TETAP,
                                "Bobot AHP": bobot_ahp.round(4),
                                "Preferensi (normalisasi)": (pref_array / pref_array.sum()).round(4),
                                "Bobot Final": bobot_final.round(4)})
st.dataframe(df_bobot_final, use_container_width=True, hide_index=True)
st.bar_chart(df_bobot_final.set_index("Kriteria")["Bobot Final"])

st.divider()

# ------------------------------------------------------------------
# TAHAP 5: VIKOR
# ------------------------------------------------------------------
st.subheader("5️⃣ Hasil Ranking VIKOR")

if st.button("🚀 Hitung Rekomendasi Kampus", type="primary"):
    if df_kampus["Nama Kampus"].duplicated().any():
        st.warning("⚠️ Ada nama kampus yang sama persis di tabel, sebaiknya dibedakan (mis. tambah nama jurusan).")

    df_kampus["Label"] = df_kampus["Nama Kampus"].astype(str) + " — " + df_kampus["Jurusan"].astype(str)
    data_vikor = df_kampus.set_index("Label")[
        ["Skor Kecocokan (ML)", "Akreditasi (angka)", "Daya Tampung", "Keketatan"]
    ]
    data_vikor.columns = KRITERIA_TETAP  # samakan nama kolom dengan bobot

    hasil = hitung_vikor(data_vikor, bobot_final, TIPE_TETAP, v=v_strategi)

    st.write("**Tabel Ranking Kampus (Q terkecil = paling direkomendasikan):**")
    st.dataframe(hasil.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                 use_container_width=True)
    st.bar_chart(hasil.set_index("Kampus")["Q"])

    top3 = hasil.head(3)["Kampus"].tolist()
    ranked_text = ", ".join([f"{i+1}. {nama}" for i, nama in enumerate(top3)])
    st.success(f"🏆 Rekomendasi: Berdasarkan nilai dan preferensi Anda, "
               f"Rekomendasi Kampus Terbaik adalah: {ranked_text}")

    csv_out = hasil.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Hasil (CSV)", csv_out, "hasil_rekomendasi_kampus.csv", "text/csv")
else:
    st.info("Isi semua bagian di atas, lalu klik **Hitung Rekomendasi Kampus**.")

st.divider()
with st.expander("ℹ️ Catatan Integrasi"):
    st.markdown("""
    - **Skor Kecocokan (ML)** saat ini diinput manual lewat slider sebagai simulasi. Untuk produksi,
      ganti bagian ini dengan pemanggilan API/model Alif yang mengembalikan skor per jurusan secara otomatis.
    - **Cluster kepribadian (Ghina/K-Means)** belum dipakai langsung sebagai kriteria numerik di VIKOR;
      bisa ditambahkan sebagai kriteria ke-5 (misalnya skor kecocokan cluster-jurusan) kalau diperlukan.
    - **Database 64 kampus**: upload file CSV asli menggantikan data contoh di atas.
    """)
