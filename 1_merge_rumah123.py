import pandas as pd
import os
import re

# Daftar file CSV yang akan digabungkan.
files = [
    "rumah123_surabaya_2024_audit.csv",
    "rumah123_surabaya_2025_audit.csv",
    "rumah123_surabaya_2026_audit.csv"
]

dataframes = []

for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file)

        # Ekstrak tahun dari nama file menggunakan regex ('rumah123_surabaya_2024_audit.csv' -> 2024)
        match = re.search(r"_(\d{4})_audit", file)
        tahun = int(match.group(1)) if match else None
        df["tahun"] = tahun

        print(f"✅ Loaded '{file}' — {len(df)} baris | kolom 'tahun' = {tahun}")
        if "link detail rumah" not in df:
            df["link detail rumah"] = pd.NA
        if "link arsip" in df:
            df["link detail rumah"] = df["link detail rumah"].replace("-", pd.NA).fillna(df["link arsip"])
        dataframes.append(df)
    else:
        print(f"⚠️  File tidak ditemukan: '{file}' — dilewati (jalankan 0_audit_data.py terlebih dahulu)")

if not dataframes:
    print("❌ Tidak ada file yang berhasil dimuat. Jalankan 0_audit_data.py terlebih dahulu, lalu periksa nama dan lokasi file.")
else:
    combined = pd.concat(dataframes, ignore_index=True)
    output_file = "output/rumah123_surabaya_2024-2026.csv"
    os.makedirs("output", exist_ok=True)
    combined.to_csv(output_file, index=False)
    print(f"\n✅ Berhasil menggabungkan {len(dataframes)} file.")
    print(f"📄 Output: '{output_file}' — Total {len(combined)} baris")