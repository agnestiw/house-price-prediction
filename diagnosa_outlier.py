import sys
import os
import numpy as np
import pandas as pd

# ---------------- KONFIGURASI ----------------
KOL_HARGA = "harga"
KOL_LT = "luas tanah"
KOL_LB = "luas bangunan"
KOL_KEC = "kecamatan"
KOL_LANTAI = "jumlah lantai"   # boleh tidak ada
IQR_K = 1.5                    # 1.5 = standar Tukey; 3.0 = hanya yang sangat ekstrem
OUT = "output/diagnosa"
# ---------------------------------------------


def main(path):
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(path)
    n = len(df)
    print(f"Data: {n} baris")

    # 1. Sebaran harga
    h = df[KOL_HARGA]
    q = h.quantile([0.5, 0.9, 0.95, 0.99, 1.0])
    print("\n1. SEBARAN HARGA")
    for k, v in q.items():
        print(f"   persentil {int(k*100):>3}: Rp {v:,.0f}".replace(",", "."))
    top1 = h[h >= h.quantile(0.99)]
    share = ((top1 - h.mean()) ** 2).sum() / ((h - h.mean()) ** 2).sum()
    print(f"   1% rumah termahal menyumbang {share:.0%} dari total variasi harga")
    print("   (makin besar, makin R2 skala Rupiah ditentukan oleh segelintir rumah)")

    # 2. Aturan logika
    print("\n2. PELANGGARAN ATURAN LOGIKA")
    aturan = {
        "luas tanah <= 0": df[KOL_LT] <= 0,
        "luas bangunan <= 0": df[KOL_LB] <= 0,
        "harga <= 0": df[KOL_HARGA] <= 0,
    }
    if KOL_LANTAI in df.columns:
        aturan["luas bangunan > luas tanah x jumlah lantai x 1.2"] = (
            df[KOL_LB] > df[KOL_LT] * df[KOL_LANTAI].clip(lower=1) * 1.2
        )
    aturan["duplikat persis"] = df.duplicated(keep="first")
    flag_logika = pd.Series(False, index=df.index)
    for nama, m in aturan.items():
        print(f"   {nama:<50}: {int(m.sum())} baris")
        flag_logika |= m.fillna(False)

    # 3. Harga per m2 ekstrem per kecamatan (skala log)
    print("\n3. HARGA PER M2 EKSTREM PER KECAMATAN (IQR skala log)")
    luas_ref = df[[KOL_LT, KOL_LB]].max(axis=1).clip(lower=1)
    df["_log_harga_m2"] = np.log(df[KOL_HARGA].clip(lower=1) / luas_ref)

    def flag_iqr(s):
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return (s < q1 - IQR_K * iqr) | (s > q3 + IQR_K * iqr)

    flag_m2 = df.groupby(KOL_KEC)["_log_harga_m2"].transform(flag_iqr)
    print(f"   {int(flag_m2.sum())} baris ({flag_m2.mean():.1%}) di luar batas IQR x{IQR_K}")

    df["_flag_logika"] = flag_logika
    df["_flag_harga_m2"] = flag_m2
    kandidat = df[flag_logika | flag_m2].sort_values(KOL_HARGA, ascending=False)
    kandidat.to_csv(f"{OUT}/kandidat_dihapus.csv", index=False)
    print(f"\nTotal kandidat: {len(kandidat)} baris ({len(kandidat)/n:.1%})")
    print(f"Tersimpan: {OUT}/kandidat_dihapus.csv  -> PERIKSA MANUAL sebelum menghapus")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pakai: python3 diagnosa_outlier.py <data_sebelum_split.csv>")
        sys.exit(1)
    main(sys.argv[1])
