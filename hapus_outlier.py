import sys
import os
import numpy as np
import pandas as pd

# ---------------- KONFIGURASI ----------------
KOL_HARGA = "harga"
KOL_LT = "luas tanah"
KOL_LB = "luas bangunan"
KOL_KEC = "kecamatan"
KOL_LANTAI = "jumlah lantai"

# Aturan A: batasan lingkup harga (rumah di atas batas ini tidak diteliti)
PAKAI_A = False
BATAS_HARGA = 37_000_000_000          # Rp 37 miliar (persentil 99 data kamu)

# Aturan B: luas bangunan tidak masuk akal (salah input)
PAKAI_B = False
FAKTOR_LB = 2.0                       # LB > LT x jumlah lantai x 2 -> tidak mungkin

# Aturan C: harga per m2 sangat ekstrem di kecamatannya
PAKAI_C = False
IQR_K = 3.0                           # 3.0 = hanya yang SANGAT ekstrem (bukan 1.5)
# ---------------------------------------------


def main(path):
    df = pd.read_csv(path)
    n_awal = len(df)
    alasan = pd.Series("", index=df.index)

    if PAKAI_A:
        m = df[KOL_HARGA] > BATAS_HARGA
        alasan[m] += f"A: harga > Rp {BATAS_HARGA:,.0f}; ".replace(",", ".")
        print(f"Aturan A (harga > batas lingkup)        : {int(m.sum())} baris")

    if PAKAI_B:
        lantai = df[KOL_LANTAI].clip(lower=1) if KOL_LANTAI in df.columns else 1
        m = df[KOL_LB] > df[KOL_LT] * lantai * FAKTOR_LB
        alasan[m] += f"B: LB > LT x lantai x {FAKTOR_LB}; "
        print(f"Aturan B (luas bangunan tidak masuk akal): {int(m.sum())} baris")

    if PAKAI_C:
        luas_ref = df[[KOL_LT, KOL_LB]].max(axis=1).clip(lower=1)
        log_m2 = np.log(df[KOL_HARGA].clip(lower=1) / luas_ref)

        def flag(s):
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            return (s < q1 - IQR_K * iqr) | (s > q3 + IQR_K * iqr)

        m = log_m2.groupby(df[KOL_KEC]).transform(flag)
        alasan[m] += f"C: harga/m2 di luar IQR x{IQR_K} kecamatannya; "
        print(f"Aturan C (harga per m2 sangat ekstrem)  : {int(m.sum())} baris")

    hapus = alasan != ""
    os.makedirs("output/diagnosa", exist_ok=True)
    dihapus = df[hapus].assign(alasan_dihapus=alasan[hapus])
    dihapus.to_csv("output/diagnosa/data_dihapus.csv", index=False)

    bersih = df[~hapus].reset_index(drop=True)
    out = path.replace(".csv", "-bersih.csv")
    bersih.to_csv(out, index=False)

    print(f"\nTotal dihapus : {int(hapus.sum())} dari {n_awal} baris ({hapus.mean():.1%})")
    print(f"Sisa data     : {len(bersih)} baris")
    print(f"Harga maksimum sekarang: Rp {bersih[KOL_HARGA].max():,.0f}".replace(",", "."))
    print(f"Tersimpan: {out}")
    print("Tersimpan: output/diagnosa/data_dihapus.csv (cek alasan tiap baris)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pakai: python3 hapus_outlier.py <data_sebelum_split.csv>")
        sys.exit(1)
    main(sys.argv[1])
