"""
SPK Pemilihan Jurusan SMA - Metode AHP (Pembobotan Kriteria) + VIKOR (Perankingan)
Jalankan dengan: streamlit run spk_ahp_vikor.py
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SPK Penjurusan SMA - AHP & VIKOR", layout="wide")

# ------------------------------------------------------------------
# Konstanta AHP
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

# ------------------------------------------------------------------
# Fungsi AHP
# ------------------------------------------------------------------
def hitung_ahp(matriks):
    n = matriks.shape[0]
    # normalisasi kolom
    kolom_sum = matriks.sum(axis=0)
    norm = matriks / kolom_sum
    bobot = norm.mean(axis=1)

    # cek konsistensi
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
    """
    data  : DataFrame (baris=alternatif, kolom=kriteria) berisi nilai numerik
    bobot : array bobot kriteria (hasil AHP), total = 1
    tipe  : list 'benefit' / 'cost' per kolom
    v     : bobot strategi mayoritas (default 0.5)
    """
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
        total = 0
        maxval = -np.inf
        for j in range(n_krit):
            denom = (f_star[j] - f_minus[j])
            if denom == 0:
                d = 0
            else:
                d = bobot[j] * (f_star[j] - X[i, j]) / denom
            total += d
            maxval = max(maxval, d)
        S[i] = total
        R[i] = maxval

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = np.zeros(n_alt)
    for i in range(n_alt):
        sd = (S_minus - S_star) if (S_minus - S_star) != 0 else 1e-9
        rd = (R_minus - R_star) if (R_minus - R_star) != 0 else 1e-9
        Q[i] = v * (S[i] - S_star) / sd + (1 - v) * (R[i] - R_star) / rd

    hasil = pd.DataFrame({
        "Alternatif": data.index,
        "S": S, "R": R, "Q": Q
    }).sort_values("Q").reset_index(drop=True)
    hasil.index = hasil.index + 1
    return hasil


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🎓 SPK Pemilihan Jurusan SMA")
st.caption("Metode AHP (pembobotan kriteria) dikombinasikan dengan VIKOR (perankingan jurusan/alternatif)")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    n_kriteria = st.number_input("Jumlah Kriteria", min_value=2, max_value=8, value=4, step=1)
    n_alternatif = st.number_input("Jumlah Alternatif (Jurusan)", min_value=2, max_value=8, value=3, step=1)
    v_strategi = st.slider("Bobot strategi mayoritas VIKOR (v)", 0.0, 1.0, 0.5, 0.05,
                            help="v=0.5 konsensus, v>0.5 mayoritas kriteria, v<0.5 veto individu")

default_kriteria = ["Nilai Matematika/IPA", "Nilai IPS/Bahasa", "Minat Siswa", "Rekomendasi Guru BK",
                     "Nilai Tes Bakat Minat", "Rata-rata Rapor", "Kehadiran", "Prestasi Non-Akademik"]
default_jurusan = ["IPA", "IPS", "Bahasa", "Jurusan D", "Jurusan E", "Jurusan F", "Jurusan G", "Jurusan H"]

st.subheader("1️⃣ Nama Kriteria & Tipe")
kriteria_list = []
tipe_list = []
cols = st.columns(min(n_kriteria, 4))
for i in range(n_kriteria):
    with cols[i % 4]:
        nama = st.text_input(f"Kriteria {i+1}", value=default_kriteria[i], key=f"krit_{i}")
        tipe = st.selectbox(f"Tipe {i+1}", ["benefit", "cost"], key=f"tipe_{i}")
        kriteria_list.append(nama)
        tipe_list.append(tipe)

st.subheader("2️⃣ Nama Alternatif (Jurusan)")
alt_list = []
cols2 = st.columns(min(n_alternatif, 4))
for i in range(n_alternatif):
    with cols2[i % 4]:
        nama_alt = st.text_input(f"Alternatif {i+1}", value=default_jurusan[i], key=f"alt_{i}")
        alt_list.append(nama_alt)

st.divider()

# ------------------------------------------------------------------
# Bagian AHP: matriks perbandingan berpasangan
# ------------------------------------------------------------------
st.subheader("3️⃣ AHP — Perbandingan Berpasangan Kriteria")
st.caption("Isi seberapa penting kriteria di BARIS dibanding kriteria di KOLOM.")

if "ahp_matrix" not in st.session_state or st.session_state.get("n_krit_prev") != n_kriteria:
    st.session_state.ahp_matrix = np.ones((n_kriteria, n_kriteria))
    st.session_state.n_krit_prev = n_kriteria

matriks = st.session_state.ahp_matrix.copy()

for i in range(n_kriteria):
    row_cols = st.columns(n_kriteria - i - 1) if i < n_kriteria - 1 else []
    idx = 0
    for j in range(i + 1, n_kriteria):
        with row_cols[idx]:
            label = f"{kriteria_list[i]}  vs  {kriteria_list[j]}"
            pilihan = st.selectbox(label, list(SAATY_SCALE.keys()),
                                    index=4, key=f"ahp_{i}_{j}")
            nilai = SAATY_SCALE[pilihan]
            matriks[i, j] = nilai
            matriks[j, i] = 1 / nilai
        idx += 1

st.session_state.ahp_matrix = matriks

with st.expander("Lihat Matriks Perbandingan Lengkap"):
    df_matrix = pd.DataFrame(matriks, index=kriteria_list, columns=kriteria_list).round(3)
    st.dataframe(df_matrix, use_container_width=True)

bobot_ahp, lambda_max, CI, CR = hitung_ahp(matriks)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.write("**Bobot Kriteria (hasil AHP):**")
    df_bobot = pd.DataFrame({"Kriteria": kriteria_list, "Bobot": bobot_ahp.round(4)})
    st.dataframe(df_bobot, use_container_width=True, hide_index=True)
    st.bar_chart(df_bobot.set_index("Kriteria"))

with col_b:
    st.write("**Uji Konsistensi:**")
    st.metric("λ maks", f"{lambda_max:.4f}")
    st.metric("Consistency Index (CI)", f"{CI:.4f}")
    st.metric("Consistency Ratio (CR)", f"{CR:.4f}")
    if CR < 0.1:
        st.success("✅ CR < 0.1 → matriks KONSISTEN, bobot layak dipakai.")
    else:
        st.error("⚠️ CR ≥ 0.1 → matriks TIDAK konsisten. Perbaiki isian perbandingan di atas.")

st.divider()

# ------------------------------------------------------------------
# Bagian nilai alternatif x kriteria
# ------------------------------------------------------------------
st.subheader("4️⃣ Nilai Setiap Alternatif per Kriteria")
st.caption("Masukkan nilai numerik (misal skala 0-100) untuk tiap jurusan pada tiap kriteria.")

if ("nilai_df" not in st.session_state
        or st.session_state.get("shape_prev") != (n_alternatif, n_kriteria)):
    st.session_state.nilai_df = pd.DataFrame(
        np.full((n_alternatif, n_kriteria), 75.0),
        index=alt_list, columns=kriteria_list
    )
    st.session_state.shape_prev = (n_alternatif, n_kriteria)

edit_df = st.session_state.nilai_df.copy()
edit_df.index = alt_list
edit_df.columns = kriteria_list

nilai_edited = st.data_editor(edit_df, use_container_width=True, num_rows="fixed")
st.session_state.nilai_df = nilai_edited

st.divider()

# ------------------------------------------------------------------
# Hitung VIKOR
# ------------------------------------------------------------------
st.subheader("5️⃣ Hasil Perankingan VIKOR")

if st.button("🚀 Hitung Ranking", type="primary"):
    if CR >= 0.1:
        st.warning("Bobot AHP belum konsisten (CR ≥ 0.1). Hasil tetap dihitung, tapi sebaiknya perbaiki dulu perbandingan kriteria di atas.")

    hasil = hitung_vikor(nilai_edited, bobot_ahp, tipe_list, v=v_strategi)

    st.write("**Tabel Ranking (semakin kecil nilai Q, semakin direkomendasikan):**")
    st.dataframe(hasil.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}),
                 use_container_width=True)

    st.write("**Grafik Nilai Q per Alternatif:**")
    st.bar_chart(hasil.set_index("Alternatif")["Q"])

    terbaik = hasil.iloc[0]
    st.success(f"🏆 Rekomendasi jurusan terbaik: **{terbaik['Alternatif']}** "
               f"(Q = {terbaik['Q']:.4f})")

    # cek syarat acceptable advantage & acceptable stability (opsional, sesuai teori VIKOR asli)
    if len(hasil) > 1:
        DQ = 1 / (len(hasil) - 1)
        selisih = hasil.iloc[1]["Q"] - hasil.iloc[0]["Q"]
        with st.expander("Cek Syarat Kelayakan Solusi Kompromi (Opsional)"):
            st.write(f"Syarat *Acceptable Advantage*: Q(A2) - Q(A1) ≥ DQ = {DQ:.4f}")
            st.write(f"Selisih aktual: {selisih:.4f}")
            if selisih >= DQ:
                st.success("✅ Solusi kompromi tunggal terpenuhi.")
            else:
                st.info("ℹ️ Selisih kecil — beberapa alternatif teratas bisa dianggap setara sebagai solusi kompromi.")

    csv = hasil.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Hasil (CSV)", csv, "hasil_ranking_vikor.csv", "text/csv")
else:
    st.info("Klik tombol **Hitung Ranking** setelah semua data terisi.")

st.divider()
with st.expander("ℹ️ Penjelasan Singkat Metode"):
    st.markdown("""
**AHP (Analytic Hierarchy Process)** digunakan untuk menentukan **bobot tiap kriteria**
berdasarkan perbandingan berpasangan (pairwise comparison) antar kriteria. Hasilnya diuji
konsistensinya lewat *Consistency Ratio* (CR); jika CR < 0.1, bobot dianggap layak dipakai.

**VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje)** digunakan untuk **meranking
alternatif** (jurusan) menggunakan bobot dari AHP. VIKOR menghitung tiga nilai:
- **S** (utility measure) — jarak rata-rata terbobot ke solusi ideal
- **R** (regret measure) — jarak maksimum terbobot ke solusi ideal
- **Q** — gabungan S dan R dengan parameter strategi *v*

Alternatif dengan **Q terkecil** adalah solusi kompromi terbaik.
    """)
