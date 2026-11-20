import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

# ===========================================================================
# KONFIGURASI
# ===========================================================================
INPUT           = "output/preprocessing/hasil-preprocessed.csv"    # hasil preprocessing + hapus_outlier.py
OUTPUT_HEATMAP  = "output/korelasi/hasil-heatmap.png"
OUTPUT_KORELASI = "output/korelasi/hasil_korelasi_target.csv"  # tabel korelasi thd harga (untuk Bab IV)

TARGET = "harga"
# kolom di CSV -> label pada heatmap (urutan ini dipakai di heatmap; target sebaiknya pertama)
VARIABEL = {
    "harga":            "Harga",
    "luas tanah":       "Luas Tanah",
    "luas bangunan":    "Luas Bangunan",
    "kamar tidur":      "Kamar Tidur",
    "kamar mandi":      "Kamar Mandi",
    "jumlah lantai":    "Jumlah Lantai",
    "lebar jalan":      "Lebar Jalan Depan",
    "daya listrik":     "Daya Listrik",
    "Tahun":            "Tahun",
    "Jumlah Penduduk":  "Jumlah Penduduk",
    "UMK":              "UMK",
    "Inflasi y-o-y":    "Inflasi",
    "Nilai Tanah Zona": "Nilai Tanah Zona",
}
# kolom kategorik nominal (kode label encoding tidak bermakna besar-kecil),
# tidak dianalisis korelasinya (tetap digunakan pada pemodelan)
TIDAK_DIANALISIS = ["sertifikat", "KPR", "kecamatan", "Wilayah"]

AMBANG_MULTIKOLINEAR = 0.9   # |rho| >= 0,9 antar-prediktor
ALPHA = 0.05                  # taraf signifikansi uji korelasi


# interpretasi kekuatan korelasi (batas bawah |rho|, label) - sesuaikan dengan tinjauan pustaka
INTERPRETASI = [(0.80, "Sangat kuat"), (0.60, "Kuat"), (0.40, "Sedang"),
                (0.20, "Rendah"), (0.00, "Sangat rendah")]   # sesuai Tabel 2.10

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# ===========================================================================
# FUNGSI BANTU
# ===========================================================================
def judul(teks):
    print("\n" + "=" * 75 + f"\n{teks}\n" + "=" * 75)


def kekuatan(r):
    if pd.isna(r):
        return "Tidak terdefinisi"
    for batas, label in INTERPRETASI:
        if abs(r) >= batas:
            return label
    return INTERPRETASI[-1][1]


def arah(r):
    if pd.isna(r):
        return "-"
    return "Positif" if r > 0 else ("Negatif" if r < 0 else "Tidak ada")


# ===========================================================================
# 1. BACA DATA LATIH
# ===========================================================================
df = pd.read_csv(INPUT)
kolom = [c for c in VARIABEL if c in df.columns]
hilang = [c for c in VARIABEL if c not in df.columns]
if TARGET not in kolom:
    raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan di {INPUT}")

data = df[kolom].rename(columns=VARIABEL)
label_target = VARIABEL[TARGET]
LABEL_KE_KOLOM = {v: k for k, v in VARIABEL.items()}

judul("1. DATA")
print(f"Jumlah data: {len(df)} (dari {INPUT})")
print("Variabel dianalisis :", list(data.columns))
print("Tidak dianalisis    :", [c for c in TIDAK_DIANALISIS if c in df.columns],
      "(kategorik nominal, tetap digunakan pada pemodelan)")
if hilang:
    print("[INFO] Kolom tidak ada di data, dilewati:", hilang)

# variabel yang nilainya hanya satu -> korelasi tidak terdefinisi
konstan = [c for c in data.columns if data[c].nunique(dropna=True) <= 1]
if konstan:
    print("\n[PERHATIAN] Variabel hanya memiliki satu nilai pada data, "
          "korelasi tidak dapat dihitung:")
    for c in konstan:
        print(f"   - {c}: {data[c].dropna().unique().tolist()}")

# ===========================================================================
# 2. MATRIKS KORELASI SPEARMAN
# ===========================================================================
judul("2. MATRIKS KORELASI SPEARMAN")
rho = data.corr(method="spearman")
print(rho.round(3).to_string())

# ===========================================================================
# 3. KORELASI SETIAP VARIABEL TERHADAP HARGA
# ===========================================================================
judul(f"3. KORELASI TERHADAP {label_target.upper()}")
baris = []
for c in data.columns.drop(label_target):
    if c in konstan:
        r, p = np.nan, np.nan
    else:
        pasangan = data[[c, label_target]].dropna()
        r, p = spearmanr(pasangan[c], pasangan[label_target])
    baris.append([LABEL_KE_KOLOM[c], c, r, p, kekuatan(r), arah(r),
                  "-" if pd.isna(p) else ("Signifikan" if p < ALPHA else "Tidak signifikan")])
thd_target = (pd.DataFrame(baris, columns=["kolom", "variabel", "rho", "p_value", "kekuatan",
                                           "arah", f"uji (alpha={ALPHA})"])
              .assign(_abs=lambda d: d["rho"].abs())
              .sort_values("_abs", ascending=False, na_position="last")
              .drop(columns="_abs"))
print(thd_target.drop(columns="kolom")
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

negatif = thd_target.loc[thd_target["arah"] == "Negatif", "variabel"].tolist()
if negatif:
    print(f"\n[INFO] Variabel berkorelasi negatif: {negatif}. Korelasi negatif berarti arah "
          "hubungan berlawanan (bukan tidak berpengaruh).")

thd_target.to_csv(OUTPUT_KORELASI, index=False, encoding="utf-8-sig")
print(f"\nTabel korelasi tersimpan: {OUTPUT_KORELASI}")
print("[INFO] Analisis korelasi hanya untuk EDA: tidak ada variabel yang dihapus, "
      "seluruh variabel prediktor tetap digunakan pada pemodelan.")

# ===========================================================================
# 4. DETEKSI KORELASI SANGAT KUAT ANTAR-PREDIKTOR
# ===========================================================================
judul(f"4. PASANGAN PREDIKTOR DENGAN |rho| >= {AMBANG_MULTIKOLINEAR}")
prediktor = [c for c in data.columns if c != label_target]
r_target = rho[label_target].abs()
pasang = []
for i, a in enumerate(prediktor):
    for b in prediktor[i + 1:]:
        r = rho.loc[a, b]
        if pd.notna(r) and abs(r) >= AMBANG_MULTIKOLINEAR:
            pasang.append([a, b, r, f"{a}: {r_target[a]:.3f} | {b}: {r_target[b]:.3f}"])
if pasang:
    print(pd.DataFrame(pasang, columns=["variabel_1", "variabel_2", "rho",
                                        f"|rho| thd {label_target}"])
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n[INFO] Pasangan di atas hanya dilaporkan (tidak dihapus), karena algoritma berbasis "
          "pohon memilih satu variabel terbaik pada setiap percabangan.")
else:
    print("Tidak ada pasangan prediktor dengan korelasi sangat kuat.")
if konstan:
    print(f"\nVariabel tanpa variasi ({', '.join(konstan)}) tidak memberi informasi pada model "
          "selama datanya hanya satu nilai.")

# ===========================================================================
# 5. HEATMAP
# ===========================================================================
# diverging: biru (negatif) - abu netral (0) - merah (positif)
cmap = LinearSegmentedColormap.from_list("spearman", ["#2a78d6", "#f0efec", "#e34948"])
cmap.set_bad("white")                        # segitiga atas dikosongkan

mask_atas = np.triu(np.ones_like(rho, dtype=bool), k=1)   # hanya segitiga bawah
n = len(rho)
fig, ax = plt.subplots(figsize=(0.75 * n + 3, 0.65 * n + 2), dpi=150)
sns.heatmap(rho, mask=mask_atas, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=2, linecolor="#fcfcfb", annot=False,
            cbar_kws={"label": "Koefisien korelasi Spearman (ρ)", "shrink": 0.75}, ax=ax)

# sel tidak terdefinisi (variabel konstan) diberi abu-abu
for i in range(n):
    for j in range(i + 1):
        if pd.isna(rho.iat[i, j]):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="#e6e5e1", edgecolor="#fcfcfb", lw=2))

# anotasi nilai: tinta gelap di sel terang, putih di sel pekat; '-' untuk tidak terdefinisi
for i in range(n):
    for j in range(i + 1):
        r = rho.iat[i, j]
        teks = "-" if pd.isna(r) else f"{r:.2f}"
        warna = "white" if pd.notna(r) and abs(r) >= 0.6 else "#2b2b29"
        ax.text(j + 0.5, i + 0.5, teks, ha="center", va="center", fontsize=8, color=warna)

ax.set_title("Heatmap Korelasi Spearman", fontsize=12, pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
ax.tick_params(length=0)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():   # tebalkan label target
    if lbl.get_text() == label_target:
        lbl.set_fontweight("bold")
if konstan:
    fig.text(0.01, -0.02, f"'-' = tidak terdefinisi (variabel hanya memiliki satu nilai: "
             f"{', '.join(konstan)})", fontsize=7, color="#5f5e5a")
fig.tight_layout()
fig.savefig(OUTPUT_HEATMAP, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nHeatmap tersimpan: {OUTPUT_HEATMAP}")
plt.show()