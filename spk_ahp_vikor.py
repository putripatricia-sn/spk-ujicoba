"""
SPK Rekomendasi Kampus/Jurusan - VIKOR & AHP-VIKOR
=====================================================
Mendukung dua mode penggunaan:
  1. MODE CP  -> VIKOR murni, bobot HANYA dari preferensi user (slider/ranking).
                 Cocok untuk versi production/demo yang dipakai siswa.
  2. MODE TA  -> AHP-VIKOR, bobot gabungan AHP (pakar/literatur) + preferensi user.
                 Serta mode "Bandingkan Keduanya" untuk analisis TA
                 (hasil VIKOR vs AHP-VIKOR, Spearman Rank Correlation, dsb).

Alur data:
  Skor ML (input Alif) -> Database Kampus -> Matriks Keputusan
  -> Bobot -> VIKOR -> Ranking

Jalankan dengan: streamlit run spk_vikor_ahp_vikor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

st.set_page_config(page_title="SPK Rekomendasi Kampus - VIKOR & AHP-VIKOR", layout="wide")

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
    hasil.insert(0, "Rank", hasil.index)
    return hasil


# ------------------------------------------------------------------
# UI - Header & Pemilihan Mode
# ------------------------------------------------------------------
st.title("🎓 SPK Rekomendasi Kampus & Jurusan")
st.caption("Alur: Skor kecocokan dari model ML \u2192 Database kampus \u2192 Bobot \u2192 VIKOR \u2192 Ranking")

mode = st.radio(
    "Pilih mode penggunaan:",
    [
        "🟢 Mode CP — VIKOR saja (bobot dari preferensi user)",
        "🔵 Mode TA — AHP-VIKOR (bobot gabungan AHP + preferensi user)",
        "🟣 Mode TA — Bandingkan VIKOR vs AHP-VIKOR (untuk analisis skripsi)",
    ],
    index=0,
)
st.caption(
    "Mode CP: dipakai untuk sistem production/demo, user cukup atur preferensi, tanpa perlu mengisi AHP. "
    "Mode TA: untuk kebutuhan eksperimen dan penulisan Bab IV skripsi (perbandingan performa pembobotan)."
)

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
    - **Tahap 1 (Frontend):** siswa input nilai rapor, kuesioner RIASEC, dan preferensi.
    - **Tahap 2 (Model Alif & Ghina):** nilai rapor diproses model klasifikasi/rule-based \u2192 skor
      kecocokan per jurusan; kuesioner diproses K-Means \u2192 cluster kepribadian.

    Modul di bawah ini menerima **skor kecocokan (output Alif)** sebagai input, lalu menjalankan
    tahap database, pembobotan, dan perankingan.
    """)

st.divider()

# ------------------------------------------------------------------
# TAHAP: Database Kampus + Skor ML
# ------------------------------------------------------------------
st.subheader("1️⃣ Database Kampus")
st.caption("Upload CSV database kampus (kolom: Nama Kampus, Jurusan, Akreditasi, Daya Tampung, Keketatan), "
           "atau edit tabel contoh di bawah.")

col_up, col_dl = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader("Upload database kampus (.csv)", type=["csv"])
with col_dl:
    template_csv = SAMPLE_KAMPUS.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Template CSV", template_csv, "template_database_kampus.csv", "text/csv")

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

st.subheader("2️⃣ Skor Kecocokan dari Model ML / Rule-Based (per Jurusan)")
st.caption("Ini output dari modul Alif. Input manual dulu di sini sebagai simulasi, sebelum nanti "
           "disambungkan otomatis ke modulnya.")

daftar_jurusan = sorted(df_kampus["Jurusan"].dropna().unique().tolist())
skor_ml = {}
cols_ml = st.columns(min(len(daftar_jurusan), 4) or 1)
for i, jurusan in enumerate(daftar_jurusan):
    with cols_ml[i % 4]:
        skor_ml[jurusan] = st.slider(f"Skor: {jurusan}", 0, 100, 75, key=f"ml_{jurusan}")

df_kampus["Skor Kecocokan (ML)"] = df_kampus["Jurusan"].map(skor_ml)

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
# TAHAP: Preferensi User (selalu dibutuhkan, di semua mode)
# ------------------------------------------------------------------
st.subheader("3️⃣ Preferensi Siswa")

metode_pref = st.radio(
    "Cara siswa mengisi preferensi:",
    ["Ranking urutan prioritas (ROC)", "Slider bebas 1-5"],
    horizontal=True,
)

n_k = len(KRITERIA_TETAP)

if metode_pref.startswith("Ranking"):
    st.caption("Susun urutan kriteria dari yang PALING PENTING ke PALING KURANG PENTING.")
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
    st.caption("Bobot ROC dihitung otomatis dari urutan (Rank Order Centroid).")
else:
    st.caption("Siswa menggeser tingkat kepentingan tiap kriteria menurut prioritas pribadinya.")
    pref_user = {}
    cols_pref = st.columns(n_k)
    for i, krit in enumerate(KRITERIA_TETAP):
        with cols_pref[i]:
            pref_user[krit] = st.slider(krit, 1, 5, 3, key=f"pref_{krit}")
    pref_array = np.array([pref_user[k] for k in KRITERIA_TETAP], dtype=float)

bobot_preferensi = pref_array / pref_array.sum()

st.divider()

# ------------------------------------------------------------------
# TAHAP: AHP (hanya tampil di mode TA)
# ------------------------------------------------------------------
tampilkan_ahp = mode.startswith("🔵") or mode.startswith("🟣")
bobot_ahp = None
CR = None

if tampilkan_ahp:
    st.subheader("4️⃣ AHP — Bobot Referensi dari Literatur/Pakar")
    st.caption("Perbandingan berpasangan antar 4 kriteria tetap, berdasarkan sintesis literatur "
               "(sesuai Batasan Masalah: bobot referensi diperoleh dari studi literatur, bukan "
               "wawancara pakar langsung).")

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
        st.metric("λmax", f"{lambda_max:.4f}")
        st.metric("Consistency Index (CI)", f"{CI:.4f}")
        if CR < 0.1:
            st.success("✅ CR ≤ 0.10 — matriks konsisten, bobot referensi layak dipakai.")
        else:
            st.error("⚠️ CR > 0.10 — matriks TIDAK konsisten. Perbaiki nilai perbandingan di atas "
                      "(cek kembali apakah ada kontradiksi logis antar pilihan), lalu lihat ulang CR "
                      "sampai ≤ 0.10.")

    alpha = st.slider(
        "Parameter α — porsi pengaruh Preferensi Siswa vs Bobot Referensi AHP", 0.0, 1.0, 0.5, 0.05,
        help="α = 0 → 100% pakai bobot AHP referensi. α = 1 → 100% pakai preferensi siswa."
    )
    bobot_gabungan = (1 - alpha) * bobot_ahp + alpha * bobot_preferensi
    bobot_gabungan = bobot_gabungan / bobot_gabungan.sum()

    df_bobot_final = pd.DataFrame({
        "Kriteria": KRITERIA_TETAP,
        "Bobot AHP": bobot_ahp.round(4),
        "Bobot Preferensi": bobot_preferensi.round(4),
        "Bobot Gabungan (AHP-VIKOR)": bobot_gabungan.round(4),
    })
    st.dataframe(df_bobot_final, use_container_width=True, hide_index=True)
    st.bar_chart(df_bobot_final.set_index("Kriteria")["Bobot Gabungan (AHP-VIKOR)"])

    st.divider()

# ------------------------------------------------------------------
# TAHAP: Perhitungan & Hasil
# ------------------------------------------------------------------
st.subheader("5️⃣ Hasil Perankingan")


def siapkan_matriks_vikor(df):
    if df["Nama Kampus"].duplicated().any():
        st.warning("⚠️ Ada nama kampus yang sama persis, sebaiknya dibedakan (mis. tambah nama jurusan).")
    df = df.copy()
    df["Label"] = df["Nama Kampus"].astype(str) + " — " + df["Jurusan"].astype(str)
    data_vikor = df.set_index("Label")[
        ["Skor Kecocokan (ML)", "Akreditasi (angka)", "Daya Tampung", "Keketatan"]
    ]
    data_vikor.columns = KRITERIA_TETAP
    return data_vikor


if mode.startswith("🟢"):
    # ---------------- MODE CP: VIKOR SAJA ----------------
    st.caption("Bobot kriteria yang dipakai: 100% dari preferensi siswa (tanpa AHP).")
    if st.button("🚀 Hitung Rekomendasi Kampus (VIKOR)", type="primary"):
        data_vikor = siapkan_matriks_vikor(df_kampus)
        hasil = hitung_vikor(data_vikor, bobot_preferensi, TIPE_TETAP, v=v_strategi)

        st.write("**Tabel Ranking Kampus (Q terkecil = paling direkomendasikan):**")
        st.dataframe(hasil.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        st.bar_chart(hasil.set_index("Kampus")["Q"])

        top3 = hasil.head(3)["Kampus"].tolist()
        ranked_text = ", ".join([f"{i+1}. {nama}" for i, nama in enumerate(top3)])
        st.success(f"🏆 Rekomendasi Kampus Terbaik: {ranked_text}")

        csv_out = hasil.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh Hasil (CSV)", csv_out, "hasil_vikor.csv", "text/csv")
    else:
        st.info("Isi semua bagian di atas, lalu klik **Hitung Rekomendasi Kampus (VIKOR)**.")

elif mode.startswith("🔵"):
    # ---------------- MODE TA: AHP-VIKOR SAJA ----------------
    st.caption("Bobot kriteria yang dipakai: gabungan AHP referensi + preferensi siswa (parameter α di atas).")
    if st.button("🚀 Hitung Rekomendasi Kampus (AHP-VIKOR)", type="primary"):
        data_vikor = siapkan_matriks_vikor(df_kampus)
        hasil = hitung_vikor(data_vikor, bobot_gabungan, TIPE_TETAP, v=v_strategi)

        st.write("**Tabel Ranking Kampus (Q terkecil = paling direkomendasikan):**")
        st.dataframe(hasil.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        st.bar_chart(hasil.set_index("Kampus")["Q"])

        top3 = hasil.head(3)["Kampus"].tolist()
        ranked_text = ", ".join([f"{i+1}. {nama}" for i, nama in enumerate(top3)])
        st.success(f"🏆 Rekomendasi Kampus Terbaik: {ranked_text}")

        csv_out = hasil.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh Hasil (CSV)", csv_out, "hasil_ahp_vikor.csv", "text/csv")
    else:
        st.info("Isi semua bagian di atas, lalu klik **Hitung Rekomendasi Kampus (AHP-VIKOR)**.")

else:
    # ---------------- MODE TA: BANDINGKAN KEDUANYA ----------------
    st.caption("Mode ini menjalankan VIKOR (bobot preferensi saja) dan AHP-VIKOR (bobot gabungan) "
               "secara bersamaan, dengan dataset, alternatif, dan skenario preferensi yang sama — "
               "sehingga perbedaan hasil murni berasal dari mekanisme pembobotan.")

    if st.button("🚀 Jalankan Perbandingan VIKOR vs AHP-VIKOR", type="primary"):
        data_vikor = siapkan_matriks_vikor(df_kampus)

        hasil_vikor = hitung_vikor(data_vikor, bobot_preferensi, TIPE_TETAP, v=v_strategi)
        hasil_ahp_vikor = hitung_vikor(data_vikor, bobot_gabungan, TIPE_TETAP, v=v_strategi)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Hasil VIKOR (bobot preferensi saja)**")
            st.dataframe(hasil_vikor.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
        with col2:
            st.write("**Hasil AHP-VIKOR (bobot gabungan)**")
            st.dataframe(hasil_ahp_vikor.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                         use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📊 Analisis Perbandingan")

        # Gabungkan rank kedua metode berdasarkan nama kampus yang sama
        rank_vikor = hasil_vikor.set_index("Kampus")["Rank"]
        rank_ahp_vikor = hasil_ahp_vikor.set_index("Kampus")["Rank"]
        df_compare = pd.DataFrame({
            "Rank VIKOR": rank_vikor,
            "Rank AHP-VIKOR": rank_ahp_vikor,
        })
        df_compare["Perubahan Posisi"] = df_compare["Rank VIKOR"] - df_compare["Rank AHP-VIKOR"]
        df_compare = df_compare.sort_values("Rank VIKOR")
        st.write("**Tabel Perubahan Peringkat per Kampus** "
                 "(positif = naik peringkat di AHP-VIKOR, negatif = turun):")
        st.dataframe(df_compare, use_container_width=True)

        # Spearman Rank Correlation — sesuai metodologi Bab III
        rho, p_value = spearmanr(df_compare["Rank VIKOR"], df_compare["Rank AHP-VIKOR"])
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Spearman Rank Correlation (ρ)", f"{rho:.4f}")
        with col_m2:
            st.metric("P-value", f"{p_value:.4f}")

        if rho > 0.9:
            st.success("✅ Korelasi sangat kuat — kedua metode menghasilkan urutan peringkat yang "
                       "hampir identik.")
        elif rho > 0.7:
            st.info("ℹ️ Korelasi kuat — kedua metode cukup sejalan, namun ada beberapa pergeseran "
                    "peringkat pada sejumlah alternatif.")
        else:
            st.warning("⚠️ Korelasi sedang/lemah — integrasi AHP memberikan perbedaan signifikan "
                       "terhadap hasil rekomendasi dibandingkan VIKOR dengan bobot preferensi saja.")

        csv_compare = df_compare.to_csv().encode("utf-8")
        st.download_button("⬇️ Unduh Tabel Perbandingan (CSV)", csv_compare,
                            "perbandingan_vikor_ahp_vikor.csv", "text/csv")
    else:
        st.info("Isi semua bagian di atas (termasuk AHP), lalu klik **Jalankan Perbandingan**.")

st.divider()
with st.expander("ℹ️ Catatan Integrasi"):
    st.markdown("""
    - **Skor Kecocokan (ML)** saat ini diinput manual lewat slider sebagai simulasi. Untuk produksi,
      ganti bagian ini dengan pemanggilan fungsi/API modul Alif yang mengembalikan skor per jurusan
      secara otomatis (format: `{nama_jurusan: skor}`).
    - **Cluster kepribadian (Ghina/K-Means)** belum dipakai langsung sebagai kriteria numerik di VIKOR;
      bisa ditambahkan sebagai kriteria ke-5 jika diperlukan.
    - **Database kampus**: upload file CSV asli menggantikan data contoh di atas.
    - **Mode CP** sebaiknya dipakai sebagai default saat sistem di-deploy untuk pengguna akhir (siswa),
      karena tidak membebani mereka dengan pengisian matriks AHP.
    - **Mode TA — Bandingkan Keduanya** dipakai untuk menghasilkan data analisis Bab IV skripsi
      (tabel perbandingan ranking, nilai Q, dan Spearman Rank Correlation).
    """)
