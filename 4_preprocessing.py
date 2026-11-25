import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================================================================
# KONFIGURASI
# ===========================================================================
INPUT = "output/hasil-merge-internal-eksternal.csv"
OUTPUT = "output/preprocessing/hasil-preprocessed.csv"
KOLOM_LINK = "link detail rumah"
POLA_ID = r"((?:hos|nps)\d+)"  # kode ID properti rumah123 di dalam link
KOLOM_DIHAPUS = [
    "link detail rumah",
    "judul",
    "lokasi",
    "overview",
    "deskripsi",
    "tanggal update",
    "tag",
    "kondisi",
]
NILAI_KOSONG = {"", "-", "nan", "none", "null", "n/a"}  # dianggap NaN (huruf kecil)
# kolom numerik yang dibersihkan dari satuan/simbol
KOLOM_NUMERIK = [
    "harga",
    "luas tanah",
    "luas bangunan",
    "kamar tidur",
    "kamar mandi",
    "daya listrik",
    "jumlah lantai",
    "Tahun",
    "Jumlah Penduduk",
    "UMK",
    "Inflasi y-o-y",
    "Nilai Tanah Zona",
]
KOLOM_HARGA = "harga"  # kolom yang memakai satuan Juta/Miliar
NILAI_NAN_KHUSUS = {"daya listrik": {"lainnya"}}  # nilai tertentu per kolom -> NaN
# a.4 batas nilai wajar: (min, max); None = tanpa batas. Nilai di luar batas -> NaN
#     lalu diisi median (luas, kamar, lantai) / modus (daya listrik) pada tahap c.
#     Harga tidak dimasukkan karena target tidak boleh diisi median.
BATAS_WAJAR = {
    "luas tanah": (10, None),  # m2; di bawah 10 m2 dianggap salah input
    "luas bangunan": (15, None),  # m2; rumah terkecil umumnya tipe 21
    "kamar tidur": (1, None),
    "kamar mandi": (1, None),
    "jumlah lantai": (1, 5),
    "daya listrik": (450, 200000),  # watt; golongan PLN rumah tangga terkecil 450 VA
}
# b. outlier
KOLOM_OUTLIER = ["harga", "luas tanah", "luas bangunan"]  # dideteksi dengan IQR
# Capping tidak diterapkan: capping menyamakan luas rumah-rumah besar sehingga model tidak
# dapat membedakan harganya (R2 turun +/- 0,10 pada uji coba). Isi daftar ini bila ingin
# mengaktifkan capping kembali, mis. ["luas tanah", "luas bangunan"].
KOLOM_CAPPING = []
IQR_K = 1.5
# [REVISI] output gambar & data untuk analisis outlier
OUTPUT_GAMBAR_OUTLIER = (
    "output/preprocessing/hasil-outlier-boxplot.png"  # boxplot + histogram per atribut
)
OUTPUT_GAMBAR_SEBARAN = "output/preprocessing/hasil-outlier-sebaran.png"  # sebaran luas vs harga
OUTPUT_DATA_OUTLIER = (
    "output/preprocessing/hasil-data-outlier.csv"  # daftar baris yang terdeteksi outlier
)
# (pembagi, satuan) agar sumbu gambar mudah dibaca
SATUAN_GAMBAR = {"harga": (1e9, "miliar Rp"), "luas tanah": (1, "m²"), "luas bangunan": (1, "m²")}
WARNA_NORMAL, WARNA_OUTLIER, WARNA_BATAS = "#2a78d6", "#e34948", "#5f5e5a"
# c. missing value
ISI_MEDIAN = ["luas tanah", "luas bangunan", "kamar tidur", "kamar mandi", "jumlah lantai"]
ISI_MODUS = ["sertifikat", "daya listrik"]
ISI_TIDAK_DIKETAHUI = {
    "lebar jalan": "Tidak Diketahui",
    "KPR": "Tidak Diketahui",
}
HAPUS_JIKA_KOSONG = ["harga", "kecamatan"]
KOLOM_BILANGAN_BULAT = ["kamar tidur", "kamar mandi", "jumlah lantai"]  # median dibulatkan
# d. label encoding (kamus pemetaan)
KECAMATAN_SURABAYA = sorted(
    [
        "Asemrowo",
        "Benowo",
        "Bubutan",
        "Bulak",
        "Dukuh Pakis",
        "Gayungan",
        "Genteng",
        "Gubeng",
        "Gunung Anyar",
        "Jambangan",
        "Karang Pilang",
        "Kenjeran",
        "Krembangan",
        "Lakarsantri",
        "Mulyorejo",
        "Pabean Cantian",
        "Pakal",
        "Rungkut",
        "Sambikerep",
        "Sawahan",
        "Semampir",
        "Simokerto",
        "Sukolilo",
        "Sukomanunggal",
        "Tambaksari",
        "Tandes",
        "Tegalsari",
        "Tenggilis Mejoyo",
        "Wiyung",
        "Wonocolo",
        "Wonokromo",
    ]
)
WILAYAH_SURABAYA = sorted(
    ["Surabaya Barat", "Surabaya Pusat", "Surabaya Selatan", "Surabaya Timur", "Surabaya Utara"]
)
LEBAR_JALAN_MAKS = 4  # "4 Mobil atau lebih" -> 4
ENCODING = {
    "sertifikat": {"Lainnya": 0, "SHM": 1, "HGB": 2, "SHP": 3, "Girik": 4},
    "KPR": {"Tidak Diketahui": 0, "Tidak": 1, "Bisa": 2},
    "kecamatan": {k: i for i, k in enumerate(KECAMATAN_SURABAYA)},
    "Wilayah": {w: i for i, w in enumerate(WILAYAH_SURABAYA)},
    # "lebar jalan" di-encode dari angka jumlah mobil (lihat encode_lebar_jalan)
}
# Nilai di luar kamus menggunakan kategori cadangan dan dilaporkan di terminal.
DEFAULT_ENCODING = {
    "sertifikat": "Lainnya",
}
# penulisan lain di data -> kategori pada kamus encoding (huruf kecil)
ALIAS_ENCODING = {
    "sertifikat": {
        "hak pakai": "SHP",
        "shp": "SHP",
        "sertifikat hak pakai": "SHP",
        "shm - sertifikat hak milik": "SHM",
        "hgb - hak guna bangunan": "HGB",
        "shm": "SHM",
        "hak milik": "SHM",
        "sertifikat hak milik": "SHM",
        "hgb": "HGB",
        "shgb": "HGB",
        "hak guna bangunan": "HGB",
        "girik": "Girik",
        "petok d": "Girik",
        "letter c": "Girik",
    },
}
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# ===========================================================================
# FUNGSI BANTU
# ===========================================================================
def judul(teks):
    print("\n" + "=" * 75 + f"\n{teks}\n" + "=" * 75)


def ada(df, kolom):
    """hanya kolom yang benar-benar ada di data"""
    return [c for c in kolom if c in df.columns]


def cari_kolom(df, nama):
    """cari kolom tanpa peduli huruf besar/kecil & spasi (mis. 'wilayah' = 'Wilayah')"""
    kunci = re.sub(r"\s+", " ", nama.strip().lower())
    for c in df.columns:
        if re.sub(r"\s+", " ", c.strip().lower()) == kunci:
            return c
    return None


SATUAN = {
    "ribu": 1e3,
    "rb": 1e3,
    "juta": 1e6,
    "jt": 1e6,
    "miliar": 1e9,
    "milyar": 1e9,
    "m": 1e9,
    "triliun": 1e12,
    "t": 1e12,
}


def ke_angka(teks):
    """'1.300 Watt'->1300, '3.62%'->3.62, 'Rp3,100,000.00'->3100000, 'Rp 5.288.796,00'->5288796,
    'Rp 885 Juta'->885000000, 'Rp 3,5 Miliar'->3500000000"""
    if teks is None or (isinstance(teks, float) and np.isnan(teks)):
        return np.nan
    s = str(teks).strip().lower()
    if s in NILAI_KOSONG:
        return np.nan
    m = re.search(r"\d[\d.,]*", s)
    if not m:
        return np.nan
    angka, sisa = m.group(0).rstrip(".,"), s[m.end() :].strip()
    satuan = re.match(r"[a-z]+", sisa)
    pengali = SATUAN.get(satuan.group(0), 1) if satuan else 1
    if satuan and satuan.group(0) in ("m", "t") and "m²" in sisa:  # 'm²' bukan Miliar
        pengali = 1
    if pengali != 1:  # format Indonesia: koma = desimal
        angka = angka.replace(".", "").replace(",", ".")
    elif "." in angka and "," in angka:  # pemisah yang terakhir = desimal
        if angka.rfind(",") > angka.rfind("."):
            angka = angka.replace(".", "").replace(",", ".")
        else:
            angka = angka.replace(",", "")
    elif "," in angka:
        angka = (
            angka.replace(",", "")
            if re.fullmatch(r"\d{1,3}(,\d{3})+", angka)
            else angka.replace(",", ".")
        )
    elif "." in angka:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", angka):  # 1.300 / 5.288.796 = ribuan
            angka = angka.replace(".", "")
    try:
        nilai = float(angka) * pengali
    except ValueError:
        return np.nan
    return round(nilai, 4)  # hilangkan galat float (2,91 x 1e9 = 2910000000.0000005)


def encode_lebar_jalan(nilai):
    """'Tidak Diketahui'->0, '1 Mobil'->1, ..., '4 Mobil' / '5 Mobil' -> 4"""
    m = re.search(r"\d+", str(nilai))
    if not m:
        return 0
    return min(int(m.group(0)), LEBAR_JALAN_MAKS)


# ===========================================================================
# BACA DATA
# ===========================================================================
os.makedirs("output/preprocessing", exist_ok=True)
df = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
# rapikan spasi dan ubah penanda kosong menjadi NaN
df = df.apply(lambda s: s.str.strip())
df = df.mask(df.apply(lambda s: s.str.lower().isin(NILAI_KOSONG)))
# samakan nama kolom konfigurasi dengan nama kolom di file (beda kapital/spasi tetap terbaca)
peta_nama = {}
for nama in set(
    KOLOM_DIHAPUS
    + KOLOM_NUMERIK
    + KOLOM_OUTLIER
    + ISI_MEDIAN
    + ISI_MODUS
    + list(ISI_TIDAK_DIKETAHUI)
    + HAPUS_JIKA_KOSONG
    + list(ENCODING)
    + [KOLOM_LINK, "lebar jalan"]
):
    asli = cari_kolom(df, nama)
    if asli and asli != nama:
        peta_nama[asli] = nama
df = df.rename(columns=peta_nama)
# Audit data terhadap deskripsi listing SUDAH dilakukan pada langkah 0 (0_audit_data.py),
# sehingga data input di sini berasal dari file *_audit.csv. Jangan memanggil audit lagi di sini:
# pemanggilan kedua akan menimpa log koreksi di output/audit dengan log yang hampir kosong.
judul("DATA AWAL")
print(f"Jumlah baris: {len(df)} | jumlah kolom: {df.shape[1]}")
print("Kolom:", list(df.columns))
# ===========================================================================
# a.1 PENGHAPUSAN DATA DUPLIKAT
# ===========================================================================
judul("a.1 PENGHAPUSAN DATA DUPLIKAT (berdasarkan ID properti)")
if KOLOM_LINK in df.columns:
    id_properti = df[KOLOM_LINK].str.extract(POLA_ID, flags=re.I)[0].str.lower()
    id_properti = id_properti.fillna(df[KOLOM_LINK])  # bila pola ID tidak ada, pakai link utuh
    duplikat = id_properti.duplicated(keep="first") & id_properti.notna()
    print(
        f"Data duplikat dihapus: {duplikat.sum()} baris "
        f"({id_properti[duplikat].nunique()} ID properti muncul lebih dari sekali)"
    )
    df = df[~duplikat].reset_index(drop=True)
else:
    print(f"Kolom '{KOLOM_LINK}' tidak ada, deteksi memakai seluruh kolom")
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Data duplikat dihapus: {n - len(df)} baris")
print(f"Sisa data: {len(df)} baris")
# ===========================================================================
# a.2 PENGHAPUSAN ATRIBUT
# ===========================================================================
judul("a.2 PENGHAPUSAN ATRIBUT")
hapus = ada(df, KOLOM_DIHAPUS)
info_asal = df[ada(df, [KOLOM_LINK, "judul"])].copy()  # [REVISI] untuk menelusuri baris outlier
df["property_id"] = df[KOLOM_LINK].str.extract(POLA_ID, flags=re.I)[0].str.lower()
df["property_id"] = df["property_id"].fillna(
    pd.Series("row_" + df.index.astype(str), index=df.index)
)
df = df.drop(columns=hapus)
print("Atribut dihapus :", hapus)
print("Atribut tersisa :", list(df.columns))
# ===========================================================================
# a.3 PENGHAPUSAN KARAKTER
# ===========================================================================
judul("a.3 PENGHAPUSAN KARAKTER")
contoh = []
for kol in ada(df, KOLOM_NUMERIK):
    sebelum = df[kol].copy()
    khusus = NILAI_NAN_KHUSUS.get(kol, set())
    if khusus:
        df.loc[df[kol].str.lower().isin(khusus), kol] = np.nan
    df[kol] = df[kol].map(ke_angka)
    gagal = sebelum.notna() & df[kol].isna() & ~sebelum.str.lower().isin(khusus)
    if gagal.any():
        print(
            f"[PERHATIAN] {kol}: {gagal.sum()} nilai tidak bisa diubah ke angka ->",
            sebelum[gagal].unique()[:5].tolist(),
        )
    # contoh sebelum -> sesudah: satu nilai yang menjadi angka & satu nilai yang menjadi NaN
    jadi_angka = sebelum.notna() & df[kol].notna()
    if jadi_angka.any():
        i = jadi_angka.idxmax()
        contoh.append((kol, sebelum[i], df.loc[i, kol]))
    jadi_nan = sebelum.notna() & df[kol].isna()
    if jadi_nan.any():
        contoh.append((kol, sebelum[jadi_nan.idxmax()], np.nan))
print(
    pd.DataFrame(contoh, columns=["kolom", "sebelum", "sesudah"]).to_string(
        index=False, float_format=lambda x: f"{x:,.2f}".rstrip("0").rstrip(".")
    )
)
print("(nilai '-' / kosong sudah diubah menjadi NaN sejak data dibaca)")
print("\nJumlah NaN per kolom setelah penghapusan karakter:")
print(df.isna().sum().to_string())
# ===========================================================================
# a.4 PENANGANAN NILAI TIDAK WAJAR
# ===========================================================================
judul("a.4 PENANGANAN NILAI TIDAK WAJAR (di luar batas -> NaN, diisi pada tahap c)")
laporan = []
for kol, (bawah, atas) in BATAS_WAJAR.items():
    if kol not in df.columns:
        continue
    tidak_wajar = pd.Series(False, index=df.index)
    if bawah is not None:
        tidak_wajar |= df[kol] < bawah
    if atas is not None:
        tidak_wajar |= df[kol] > atas
    nilai = sorted(df.loc[tidak_wajar, kol].unique().tolist())
    laporan.append(
        [
            kol,
            "-" if bawah is None else bawah,
            "-" if atas is None else atas,
            int(tidak_wajar.sum()),
            ", ".join(f"{v:g}" for v in nilai[:10]) or "-",
        ]
    )
    df.loc[tidak_wajar, kol] = np.nan
print(
    pd.DataFrame(
        laporan,
        columns=["atribut", "batas_bawah", "batas_atas", "jumlah_diubah_NaN", "nilai_tidak_wajar"],
    ).to_string(index=False)
)
# ===========================================================================
# b. DETEKSI OUTLIER (IQR)
# ===========================================================================
judul("b. DETEKSI OUTLIER (IQR, NaN tidak diikutsertakan)")
ringkas = []
batas_iqr, df_sebelum_capping = {}, df.copy()  # [REVISI] dipakai untuk gambar
for kol in ada(df, KOLOM_OUTLIER):
    data = df[kol].dropna()
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    bawah, atas = q1 - IQR_K * iqr, q3 + IQR_K * iqr
    batas_iqr[kol] = (q1, q3, bawah, atas)
    n_bawah, n_atas = int((df[kol] < bawah).sum()), int((df[kol] > atas).sum())
    tindakan = "capping" if kol in KOLOM_CAPPING else "dipertahankan"
    if kol in KOLOM_CAPPING:
        df[kol] = df[kol].clip(lower=bawah, upper=atas)  # NaN tetap NaN
    ringkas.append([kol, q1, q3, iqr, bawah, atas, n_bawah, n_atas, tindakan])
print(
    pd.DataFrame(
        ringkas,
        columns=[
            "atribut",
            "Q1",
            "Q3",
            "IQR",
            "batas_bawah",
            "batas_atas",
            "n<bawah",
            "n>atas",
            "tindakan",
        ],
    ).to_string(index=False)
)
# ---------------------------------------------------------------------------
# [REVISI] b.1 VISUALISASI OUTLIER
# ---------------------------------------------------------------------------
judul("b.1 VISUALISASI OUTLIER")
d0 = df_sebelum_capping
flag = pd.DataFrame({kol: (d0[kol] < b) | (d0[kol] > a) for kol, (_, _, b, a) in batas_iqr.items()})
outlier_mana_pun = flag.any(axis=1)


def fmt(v, kol):
    pembagi, satuan = SATUAN_GAMBAR.get(kol, (1, ""))
    desimal = 2 if pembagi > 1 else 0  # miliar Rp: 2 desimal, m²: bilangan bulat
    return (
        f"{v / pembagi:,.{desimal}f} {satuan}".replace(",", "X").replace(".", ",").replace("X", ".")
    )


# Gambar 1: boxplot (kiri) + histogram skala log (kanan) untuk setiap atribut
kolom_gambar = list(batas_iqr)
fig, axes = plt.subplots(
    len(kolom_gambar),
    2,
    figsize=(13, 3.6 * len(kolom_gambar)),
    dpi=150,
    gridspec_kw={"width_ratios": [1, 1.4]},
    squeeze=False,
)
for baris, kol in zip(axes, kolom_gambar):
    q1, q3, bawah, atas = batas_iqr[kol]
    pembagi, satuan = SATUAN_GAMBAR.get(kol, (1, ""))
    nilai = d0[kol].dropna()
    is_out = flag[kol][nilai.index]
    n_out, persen = int(is_out.sum()), is_out.mean() * 100
    # boxplot horizontal; titik outlier berwarna merah
    ax = baris[0]
    arah = (
        {"orientation": "horizontal"}
        if tuple(map(int, plt.matplotlib.__version__.split(".")[:2])) >= (3, 10)
        else {"vert": False}
    )  # parameter berbeda antar versi Matplotlib
    ax.boxplot(
        nilai / pembagi,
        **arah,
        whis=IQR_K,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#dbe8f8", edgecolor=WARNA_NORMAL),
        medianprops=dict(color=WARNA_NORMAL, linewidth=2),
        whiskerprops=dict(color=WARNA_NORMAL),
        capprops=dict(color=WARNA_NORMAL),
        flierprops=dict(
            marker="o",
            markersize=3.5,
            markerfacecolor=WARNA_OUTLIER,
            markeredgecolor="none",
            alpha=0.6,
        ),
    )
    ax.set_yticks([])
    ax.set_xlabel(satuan)
    ax.set_title(f"Boxplot {kol}", fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.25)
    # histogram skala log: batang normal biru, outlier merah, garis putus = batas IQR
    ax = baris[1]
    positif = nilai[nilai > 0]
    bins = np.logspace(np.log10(positif.min()), np.log10(positif.max()), 45)
    ax.hist(
        positif[~is_out[positif.index]], bins=bins, color=WARNA_NORMAL, alpha=0.85, label="Normal"
    )
    ax.hist(
        positif[is_out[positif.index]], bins=bins, color=WARNA_OUTLIER, alpha=0.9, label="Outlier"
    )
    ax.set_xscale("log")
    for v, teks in [(bawah, "batas bawah"), (atas, "batas atas")]:
        if v > 0:
            ax.axvline(v, color=WARNA_BATAS, linestyle="--", linewidth=1)
            ax.text(
                v,
                ax.get_ylim()[1] * 0.95,
                f" {teks}\n {fmt(v, kol)}",
                fontsize=8,
                color=WARNA_BATAS,
                va="top",
            )
    ax.set_xlabel(f"{kol} ({satuan}, skala log)" if satuan else f"{kol} (skala log)")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _p, k=kol: fmt(x, k).rsplit(" ", 1)[0])
    )
    ax.set_ylabel("Jumlah data")
    ax.set_title(
        f"{kol}: {n_out} outlier ({persen:.1f}%) | Q1 {fmt(q1, kol)} | Q3 {fmt(q3, kol)}",
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle(f"Deteksi Outlier Metode IQR (k = {IQR_K})", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_GAMBAR_OUTLIER, dpi=300, bbox_inches="tight", facecolor="white")
# Gambar 2: sebaran luas vs harga (log-log) untuk melihat apakah outlier wajar
pasangan = [
    (x, "harga")
    for x in ["luas tanah", "luas bangunan"]
    if x in d0.columns and "harga" in d0.columns
]
if pasangan:
    fig2, axes2 = plt.subplots(
        1, len(pasangan), figsize=(6.5 * len(pasangan), 5.2), dpi=150, squeeze=False
    )
    for ax, (x, y) in zip(axes2[0], pasangan):
        dd = d0[[x, y]].dropna()
        dd = dd[(dd[x] > 0) & (dd[y] > 0)]
        out = outlier_mana_pun[dd.index]
        ax.scatter(
            dd.loc[~out, x],
            dd.loc[~out, y] / 1e9,
            s=10,
            alpha=0.45,
            color=WARNA_NORMAL,
            edgecolor="none",
            label=f"Normal ({(~out).sum()})",
        )
        ax.scatter(
            dd.loc[out, x],
            dd.loc[out, y] / 1e9,
            s=14,
            alpha=0.75,
            color=WARNA_OUTLIER,
            edgecolor="none",
            label=f"Outlier pada salah satu atribut ({out.sum()})",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{x} (m², skala log)")
        ax.set_ylabel("harga (miliar Rp, skala log)")
        ax.set_title(f"{x.capitalize()} vs harga", fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.grid(alpha=0.25, which="both")
    fig2.suptitle(
        "Sebaran Luas terhadap Harga: outlier yang mengikuti pola umum menunjukkan rumah besar/mewah "
        "yang wajar",
        fontsize=11,
    )
    fig2.tight_layout()
    fig2.savefig(OUTPUT_GAMBAR_SEBARAN, dpi=300, bbox_inches="tight", facecolor="white")
# Tabel baris outlier untuk dicek manual (diurutkan dari harga tertinggi)
kol_info = ada(
    d0,
    [
        "kecamatan",
        "Tahun",
        "harga",
        "luas tanah",
        "luas bangunan",
        "kamar tidur",
        "kamar mandi",
        "jumlah lantai",
    ],
)
data_out = pd.concat(
    [info_asal.loc[outlier_mana_pun[outlier_mana_pun].index], d0.loc[outlier_mana_pun, kol_info]],
    axis=1,
)
for kol in flag.columns:
    data_out[f"outlier_{kol}"] = flag.loc[data_out.index, kol].map({True: "Ya", False: ""})
if {"harga", "luas bangunan"} <= set(data_out.columns):
    data_out["harga_per_m2_bangunan"] = (data_out["harga"] / data_out["luas bangunan"]).round(0)
if "harga" in data_out.columns:
    data_out = data_out.sort_values("harga", ascending=False)
for kol in data_out.select_dtypes("number").columns:  # 2026.0 -> 2026
    if np.allclose(data_out[kol].dropna() % 1, 0):
        data_out[kol] = data_out[kol].astype("Int64")
data_out.to_csv(OUTPUT_DATA_OUTLIER, index=False, encoding="utf-8-sig")
print(
    f"Baris yang outlier pada minimal satu atribut: {int(outlier_mana_pun.sum())} "
    f"dari {len(d0)} ({outlier_mana_pun.mean() * 100:.1f}%)"
)
print("Jumlah atribut outlier per baris:")
print(
    flag.sum(axis=1)[outlier_mana_pun]
    .value_counts()
    .sort_index()
    .rename_axis("jumlah_atribut")
    .to_string()
)
print(
    f"\nTersimpan: {OUTPUT_GAMBAR_OUTLIER}, "
    f"{OUTPUT_GAMBAR_SEBARAN if pasangan else '-'}, {OUTPUT_DATA_OUTLIER}"
)
# ===========================================================================
# c. PENANGANAN MISSING VALUE
# ===========================================================================
judul("c. PENANGANAN MISSING VALUE")
print("NaN sebelum:\n", df.isna().sum()[df.isna().sum() > 0].to_string())
# baris tanpa harga / kecamatan dihapus terlebih dahulu agar median & modus
# dihitung hanya dari data yang dipakai model
wajib = ada(df, HAPUS_JIKA_KOSONG)
n = len(df)
df = df.dropna(subset=wajib).reset_index(drop=True)
print(f"\nBaris dihapus karena {wajib} kosong: {n - len(df)} (sisa {len(df)})")
isi = []
# Imputasi numerik dipelajari dalam pipeline model, bukan seluruh dataset.
for kol in ada(df, ISI_MEDIAN + ["daya listrik"]):
    isi.append([kol, "ditunda ke pipeline", None, int(df[kol].isna().sum())])
if "sertifikat" in df:
    df["sertifikat"] = df["sertifikat"].fillna("Lainnya")
for kol, label in ISI_TIDAK_DIKETAHUI.items():
    if kol in df.columns:
        # samakan penulisan 'tidak diketahui' yang sudah ada di data
        df[kol] = df[kol].where(df[kol].str.lower() != label.lower(), label)
        isi.append([kol, "kategori", label, int(df[kol].isna().sum())])
        df[kol] = df[kol].fillna(label)
print(
    pd.DataFrame(isi, columns=["atribut", "metode", "nilai_pengisi", "jumlah_diisi"]).to_string(
        index=False
    )
)
sisa_nan = df.isna().sum()
if sisa_nan.any():
    print(
        "\n[PERHATIAN] Masih ada NaN (kolom belum diatur di konfigurasi):\n",
        sisa_nan[sisa_nan > 0].to_string(),
    )
else:
    print("\nTidak ada missing value tersisa.")
# ===========================================================================
# d. TRANSFORMASI DATA (LABEL ENCODING)
# ===========================================================================
judul("d. TRANSFORMASI DATA (LABEL ENCODING)")
if "lebar jalan" in df.columns:
    asal = df["lebar jalan"]
    df["lebar jalan"] = asal.map(encode_lebar_jalan)
    tabel = (
        pd.DataFrame({"nilai_awal": asal, "hasil": df["lebar jalan"]})
        .value_counts()
        .reset_index(name="jumlah")
        .sort_values("hasil")
    )
    print("lebar jalan:\n", tabel.to_string(index=False), "\n")
for kol, kamus in ENCODING.items():
    if kol not in df.columns:
        continue
    peta = {k.lower(): v for k, v in kamus.items()}
    asal = df[kol].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    alias = ALIAS_ENCODING.get(kol, {})
    hasil = asal.str.lower().map(lambda v: alias.get(v, v).lower()).map(peta)
    baru = asal[hasil.isna()].unique().tolist()
    if baru and kol in DEFAULT_ENCODING:
        print(f"[INFO] {kol}: kategori baru {baru} dimasukkan ke '{DEFAULT_ENCODING[kol]}'")
        hasil = hasil.fillna(kamus[DEFAULT_ENCODING[kol]])
    elif baru:
        print(f"[PERHATIAN] {kol}: kategori tidak ada di kamus {baru} -> baris dihapus")
    df[kol] = hasil
    tabel = (
        pd.DataFrame({"nilai_awal": asal, "hasil": hasil})
        .dropna()
        .value_counts()
        .reset_index(name="jumlah")
        .sort_values("hasil")
    )
    print(f"{kol}:\n", tabel.to_string(index=False), "\n")
kol_enc = ada(df, list(ENCODING))
n = len(df)
df = df.dropna(subset=kol_enc).reset_index(drop=True)
if n != len(df):
    print(f"Baris dengan kategori tak dikenal dihapus: {n - len(df)}")
for kol in kol_enc + ada(df, ["lebar jalan"]):
    df[kol] = df[kol].astype(int)
# angka bulat disimpan sebagai integer (tanpa .0)
for kol in df.select_dtypes("number").columns:
    if np.allclose(df[kol].dropna() % 1, 0):
        df[kol] = df[kol].astype("Int64")
# ===========================================================================
# e. [REVISI] PENGHAPUSAN DUPLIKAT BERDASARKAN ISI
# ===========================================================================
# Tahap a.1 hanya menghapus duplikat dengan ID properti yang sama. Rumah yang sama
# sering diunggah ulang dengan ID berbeda (atau muncul lagi di tahun scraping lain),
# sehingga isinya identik tetapi ID-nya lain. Bila tidak dihapus, satu rumah bisa
# masuk ke data training sekaligus data testing (kebocoran data) dan membuat skor
# testing terlalu optimis. Pengecekan dilakukan setelah pembersihan & encoding agar
# perbedaan penulisan (mis. "SHM" vs "Sertifikat Hak Milik") tidak lolos.
judul("e. PENGHAPUSAN DUPLIKAT BERDASARKAN ISI (seluruh atribut + harga sama persis)")
KOLOM_BUKAN_ISI = ["link arsip", "property_id"]  # kolom penanda sumber, bukan isi rumah
kolom_isi = [c for c in df.columns if c not in KOLOM_BUKAN_ISI]
duplikat_isi = df.duplicated(subset=kolom_isi, keep="first")
print(
    f"Baris duplikat isi dihapus: {int(duplikat_isi.sum())} dari {len(df)} "
    f"({duplikat_isi.mean() * 100:.1f}%)"
)
df = df[~duplikat_isi].reset_index(drop=True)
print(f"Sisa data: {len(df)} baris")
# Hapus kolom penanda sumber (tidak diperlukan oleh model)
df = df.drop(columns=[c for c in ["link arsip"] if c in df.columns])
# ===========================================================================
# SIMPAN
# ===========================================================================
judul("HASIL AKHIR")
print(f"Jumlah baris: {len(df)} | jumlah kolom: {df.shape[1]}")
print(df.dtypes.to_string())
print("\n", df.head(10).to_string(index=False))
print("\nStatistik deskriptif:\n", df.describe().T.to_string())
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nTersimpan: {OUTPUT}")
plt.show()  # [REVISI] tampilkan gambar outlier di akhir agar tidak menghentikan proses