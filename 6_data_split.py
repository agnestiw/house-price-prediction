import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ===========================================================================
# KONFIGURASI
# ===========================================================================
INPUT        = "output/preprocessing/hasil-preprocessed.csv"   # [REVISI] data setelah hapus_outlier.py (aturan A, B, C)
OUTPUT_TRAIN = "output/data_train.csv"
OUTPUT_TEST  = "output/data_test.csv"

TARGET       = "harga"
TEST_SIZE    = 0.2     # 20% testing, 80% training
RANDOM_STATE = 42

# kolom yang ditampilkan proporsinya untuk memastikan train & test memiliki sebaran serupa
KOLOM_CEK_PROPORSI = ["Tahun", "Wilayah"]

pd.set_option("display.width", 200)


def judul(teks):
    print("\n" + "=" * 75 + f"\n{teks}\n" + "=" * 75)


# ===========================================================================
# 1. BAGI DATA
# ===========================================================================
df = pd.read_csv(INPUT)
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
a, b = next(splitter.split(df, groups=df["property_id"]))
train, test = df.iloc[a], df.iloc[b]
assert not set(train.property_id) & set(test.property_id)
os.makedirs("output", exist_ok=True)

judul("1. PEMBAGIAN DATA")
ringkas = pd.DataFrame({
    "data": ["Training", "Testing", "Total"],
    "jumlah": [len(train), len(test), len(df)],
    "persentase": [f"{len(train) / len(df):.0%}", f"{len(test) / len(df):.0%}", "100%"],
})
print(ringkas.to_string(index=False))
print(f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE})")

# ===========================================================================
# 2. PERBANDINGAN SEBARAN TRAIN & TEST
# ===========================================================================
judul(f"2. STATISTIK {TARGET.upper()} PADA DATA TRAINING & TESTING")
if TARGET in df.columns:
    stat = pd.DataFrame({"Training": train[TARGET].describe(),
                         "Testing": test[TARGET].describe()})
    print(stat.to_string(float_format=lambda x: f"{x:,.0f}"))

for kol in [c for c in KOLOM_CEK_PROPORSI if c in df.columns]:
    print(f"\nProporsi {kol}:")
    prop = pd.DataFrame({"Training": train[kol].value_counts(normalize=True),
                         "Testing": test[kol].value_counts(normalize=True)}).sort_index()
    print(prop.to_string(float_format=lambda x: f"{x:.1%}"))

# ===========================================================================
# 3. SIMPAN
# ===========================================================================
train.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
test.to_csv(OUTPUT_TEST, index=False, encoding="utf-8-sig")
print(f"\nTersimpan: {OUTPUT_TRAIN}, {OUTPUT_TEST}")