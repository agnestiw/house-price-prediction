import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ===========================================================================
# KONFIGURASI
# ===========================================================================
INPUT_TRAIN  = "output/data_train.csv"   # hasil 5_data_split.py
INPUT_TEST   = "output/data_test.csv"
OUTPUT_TRAIN = "output/normalisasi/data_train_normalisasi.csv"
OUTPUT_TEST  = "output/normalisasi/data_test_normalisasi.csv"
OUTPUT_SCALER = "output/normalisasi/scaler_minmax.pkl"

TARGET = "harga"
KOLOM_NORMALISASI = [
    "luas tanah", "luas bangunan", "kamar tidur", "kamar mandi", "jumlah lantai",
    "daya listrik", "Tahun", "Jumlah Penduduk", "UMK", "Inflasi y-o-y", "Nilai Tanah Zona",
]
# kolom label encoding (dibiarkan apa adanya): sertifikat, KPR, kecamatan, Wilayah, lebar jalan

DESIMAL = 6   # pembulatan hasil normalisasi pada file CSV

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def judul(teks):
    print("\n" + "=" * 75 + f"\n{teks}\n" + "=" * 75)


# ===========================================================================
# 1. BACA DATA TRAINING & TESTING
# ===========================================================================
train = pd.read_csv(INPUT_TRAIN)
test = pd.read_csv(INPUT_TEST)
df = train   # acuan daftar kolom
kolom = [c for c in KOLOM_NORMALISASI if c in df.columns and c != TARGET]
hilang = [c for c in KOLOM_NORMALISASI if c not in df.columns]
tidak_dinormalisasi = [c for c in df.columns if c not in kolom]


judul("1. DATA")
print(f"Data training: {len(train)} ({INPUT_TRAIN}) | data testing: {len(test)} ({INPUT_TEST})")
print("Dinormalisasi       :", kolom)
print("Tidak dinormalisasi :", tidak_dinormalisasi)
if hilang:
    print("[INFO] Kolom tidak ada di data, dilewati:", hilang)

# ===========================================================================
# 2. FIT MIN-MAX PADA DATA TRAINING
# ===========================================================================
scaler = MinMaxScaler()
scaler.fit(train[kolom])

judul("2. NILAI MIN & MAX (dihitung dari data training)")
ringkas = pd.DataFrame({"atribut": kolom,
                        "min": scaler.data_min_,
                        "max": scaler.data_max_,
                        "max - min": scaler.data_range_})
print(ringkas.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

konstan = ringkas.loc[ringkas["max - min"] == 0, "atribut"].tolist()
if konstan:
    print(f"\n[PERHATIAN] {konstan} hanya memiliki satu nilai pada data training "
          "(max - min = 0), sehingga hasil normalisasinya bernilai 0 untuk semua baris.")

# ===========================================================================
# 3. TRANSFORM DATA TRAINING & TESTING
# ===========================================================================
asli_train = train[kolom].copy()
train[kolom] = scaler.transform(train[kolom]).round(DESIMAL)
test[kolom] = scaler.transform(test[kolom]).round(DESIMAL)

judul("3. CONTOH PERHITUNGAN (baris pertama data training)")
i = train.index[0]
contoh = []
for c, mn, rg in zip(kolom, scaler.data_min_, scaler.data_range_):
    x = asli_train.loc[i, c]
    contoh.append([c, x, f"({x:,.2f} - {mn:,.2f}) / {rg:,.2f}" if rg else "max - min = 0",
                   train.loc[i, c]])
print(pd.DataFrame(contoh, columns=["atribut", "sebelum", "perhitungan", "sesudah"])
      .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

judul("4. PERBANDINGAN SEBELUM & SESUDAH (5 data training pertama)")
print("Sebelum:\n", asli_train.head().to_string(index=False))
print("\nSesudah:\n", train[kolom].head().to_string(index=False))

judul("5. PENGECEKAN RENTANG NILAI")
print("Data training (seharusnya 0 - 1):")
print(train[kolom].agg(["min", "max"]).T.to_string(float_format=lambda x: f"{x:.4f}"))
luar = ((test[kolom] < 0) | (test[kolom] > 1)).sum()
if luar.any():
    print("\n[INFO] Sebagian data testing berada di luar rentang 0 - 1 karena nilainya lebih kecil/"
          "besar dari min/max data training. Hal ini wajar dan tidak dipotong:")
    print(luar[luar > 0].to_string())
else:
    print("\nSeluruh data testing berada pada rentang 0 - 1.")

# ===========================================================================
# 4. SIMPAN
# ===========================================================================
train.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
test.to_csv(OUTPUT_TEST, index=False, encoding="utf-8-sig")
joblib.dump({"scaler": scaler, "kolom": kolom}, OUTPUT_SCALER)
print(f"\nTersimpan: {OUTPUT_TRAIN}, {OUTPUT_TEST}, {OUTPUT_SCALER}")