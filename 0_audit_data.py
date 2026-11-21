"""Langkah 0 - AUDIT DATA (dijalankan paling awal, sebelum 1_merge_rumah123.py).

Setiap file hasil scraping dibandingkan dengan angka yang tertulis di deskripsi listingnya
(harga, luas tanah, luas bangunan, kamar tidur, kamar mandi, jumlah lantai, daya listrik, sertifikat).
Bila nilai kolom berbeda dari deskripsi, nilai deskripsi dipakai (aturan lengkap: audit_harga.py).

Input  : rumah123_surabaya_2024.csv, rumah123_surabaya_2025.csv, rumah123_surabaya_2026.csv
         (data mentah hasil scraping; TIDAK diubah)
Output : rumah123_surabaya_2024_audit.csv, rumah123_surabaya_2025_audit.csv, rumah123_surabaya_2026_audit.csv
         -> inilah yang dibaca 1_merge_rumah123.py
         output/audit/<tahun>/koreksi_audit.csv, audit_perlu_periksa.csv, ringkasan_audit.csv
         output/audit/koreksi_audit_semua.csv, audit_perlu_periksa_semua.csv, ringkasan_audit_semua.csv
         (gabungan ketiga tahun, kolom 'tahun' menunjukkan asal file)
"""
import os

import pandas as pd

from audit_harga import audit_dan_koreksi

TAHUN = [2024, 2025, 2026]
FILE_MENTAH = "rumah123_surabaya_{tahun}.csv"
FILE_AUDIT = "rumah123_surabaya_{tahun}_audit.csv"
FOLDER_LOG = "output/audit"


def punya_bom(path):
    """True bila file diawali BOM UTF-8 (agar file hasil audit memakai format yang sama dengan aslinya)."""
    with open(path, "rb") as f:
        return f.read(3) == b"\xef\xbb\xbf"


def baca_log(folder, nama, tahun):
    log = pd.read_csv(os.path.join(folder, nama), dtype=str, keep_default_na=False)
    log.insert(0, "tahun", tahun)
    return log


os.makedirs(FOLDER_LOG, exist_ok=True)
koreksi, periksa, ringkasan = [], [], []
diproses = 0

for tahun in TAHUN:
    mentah = FILE_MENTAH.format(tahun=tahun)
    if not os.path.exists(mentah):
        print(f"⚠️  File tidak ditemukan: '{mentah}' — dilewati")
        continue

    # dibaca sebagai teks agar nilai koreksi ('Rp 2,85 Miliar', '160 m²', ...) bisa ditulis ke kolom mana pun
    df = pd.read_csv(mentah, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    print(f"\n{'=' * 75}\nAUDIT {tahun}: '{mentah}' — {len(df)} baris\n{'=' * 75}")

    folder = os.path.join(FOLDER_LOG, str(tahun))
    hasil = audit_dan_koreksi(df, folder)

    keluar = FILE_AUDIT.format(tahun=tahun)
    hasil.to_csv(keluar, index=False, encoding="utf-8-sig" if punya_bom(mentah) else "utf-8")
    n_beda = int((hasil != df).any(axis=1).sum())
    print(f"📄 Output: '{keluar}' — {len(hasil)} baris ({n_beda} baris berubah dari file mentah)")

    koreksi.append(baca_log(folder, "koreksi_audit.csv", tahun))
    periksa.append(baca_log(folder, "audit_perlu_periksa.csv", tahun))
    ringkasan.append(baca_log(folder, "ringkasan_audit.csv", tahun))
    diproses += 1

if not diproses:
    print("❌ Tidak ada file yang diaudit. Periksa nama dan lokasi file.")
else:
    pd.concat(koreksi, ignore_index=True).to_csv(os.path.join(FOLDER_LOG, "koreksi_audit_semua.csv"), index=False, encoding="utf-8-sig")
    pd.concat(periksa, ignore_index=True).to_csv(os.path.join(FOLDER_LOG, "audit_perlu_periksa_semua.csv"), index=False, encoding="utf-8-sig")
    semua = pd.concat(ringkasan, ignore_index=True)
    angka = [c for c in semua.columns if c not in ("tahun", "kolom")]
    semua[angka] = semua[angka].apply(pd.to_numeric).fillna(0).astype(int)
    semua.to_csv(os.path.join(FOLDER_LOG, "ringkasan_audit_semua.csv"), index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 75}\nRINGKASAN AUDIT SEMUA TAHUN (jumlah baris per kolom dan hasil)\n{'=' * 75}")
    print(semua.groupby("kolom")[angka].sum().to_string())
    print(f"\n✅ {diproses} file diaudit. Lanjutkan dengan: python3 1_merge_rumah123.py")