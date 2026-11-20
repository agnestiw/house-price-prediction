import re
import pandas as pd

INPUT_RUMAH     = "output/hasil-mapping-kpr-kecamatan.csv"
INPUT_EKSTERNAL = "dataset_eksternal.csv"
OUTPUT          = "output/hasil-merge-internal-eksternal.csv"

# Kolom data eksternal yang ditambahkan ke data rumah (Wilayah ikut digabungkan sebagai variabel)
KOLOM_EKSTERNAL = ["Wilayah", "Jumlah Penduduk", "UMK", "Inflasi y-o-y", "Nilai Tanah Zona"]

# Ejaan lain nama kecamatan -> disamakan sebelum dicocokkan
# (mis. main.py memakai "Asemrowo", data eksternal memakai "Asem Rowo")
ALIAS_KECAMATAN = {"tenggilis": "tenggilismejoyo", "pabeancantikan": "pabeancantian"}


def kunci_kecamatan(nama):
    """'Asem Rowo' / 'Asemrowo' / 'asem rowo ' -> 'asemrowo' (huruf kecil, tanpa spasi & tanda baca)"""
    k = re.sub(r"[^a-z0-9]", "", str(nama).lower())
    return ALIAS_KECAMATAN.get(k, k)


# ===========================================================================
# 1. BACA DATA (semua sebagai teks, nilai asli tidak diubah)
# ===========================================================================
rumah = pd.read_csv(INPUT_RUMAH, dtype=str, keep_default_na=False)
eksternal = pd.read_csv(INPUT_EKSTERNAL, dtype=str, keep_default_na=False)

# Isi kolom Wilayah disamakan menjadi "Surabaya Barat", "Surabaya Timur", dst.
eksternal["Wilayah"] = eksternal["Wilayah"].str.strip().map(
    lambda w: w if w.lower().startswith("surabaya") else f"Surabaya {w}")

rumah["_tahun"] = pd.to_numeric(rumah["tahun"].str.strip(), errors="coerce").astype("Int64")
eksternal["_tahun"] = pd.to_numeric(eksternal["Tahun"].str.strip(), errors="coerce").astype("Int64")
rumah["_kec"] = rumah["kecamatan"].map(kunci_kecamatan)
eksternal["_kec"] = eksternal["Kecamatan"].map(kunci_kecamatan)

# Satu kecamatan hanya boleh punya satu baris per tahun di data eksternal
dobel = eksternal[eksternal.duplicated(["_kec", "_tahun"], keep=False)]
if len(dobel):
    raise SystemExit(f"[ERROR] Data eksternal punya baris ganda untuk kecamatan & tahun yang sama:\n{dobel}")

# ===========================================================================
# 2. MERGE berdasarkan kecamatan & tahun (left join: seluruh baris rumah dipertahankan)
# ===========================================================================
hasil = rumah.merge(eksternal[["_kec", "_tahun"] + KOLOM_EKSTERNAL],
                    on=["_kec", "_tahun"], how="left", validate="many_to_one", indicator=True)

cocok = hasil["_merge"] == "both"
alasan = pd.Series("cocok", index=hasil.index)
alasan[~cocok & (hasil["kecamatan"].str.strip().isin(["", "-"]))] = "kecamatan kosong ('-')"
alasan[~cocok & (alasan == "cocok") & ~hasil["_tahun"].isin(eksternal["_tahun"])] = "tahun tidak ada di data eksternal"
alasan[~cocok & (alasan == "cocok")] = "nama kecamatan tidak ditemukan di data eksternal"

# Urutan kolom: kolom data rumah, lalu kolom eksternal di belakang
out = hasil[[c for c in rumah.columns if not c.startswith("_")] + KOLOM_EKSTERNAL]
print("Jumlah baris & kolom hasil merge:", out.shape)
print("Jumlah data rumah per wilayah:\n", out["Wilayah"].fillna("(tidak cocok)").value_counts().to_string(), "\n")
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

# ===========================================================================
# 3. RINGKASAN DI TERMINAL
# ===========================================================================
pd.set_option("display.width", 200)
print("=" * 25, "HASIL MERGE", "=" * 25)
print(f"Baris data rumah        : {len(rumah)}")
print(f"Berhasil digabungkan    : {cocok.sum()}")
print(f"Tidak digabungkan       : {(~cocok).sum()}  (kolom eksternal dibiarkan kosong)")
print("\nRincian:\n", alasan.value_counts().to_string())

tak_kenal = hasil.loc[alasan == "nama kecamatan tidak ditemukan di data eksternal", "kecamatan"].unique()
if len(tak_kenal):
    print("\n[PERHATIAN] Nama kecamatan belum cocok, tambahkan ke ALIAS_KECAMATAN:", list(tak_kenal))

print("\nJumlah data rumah per tahun & kecamatan (yang berhasil digabung):")
ring = (hasil[cocok].groupby(["tahun", "kecamatan"]).size()
        .reset_index(name="jumlah_rumah").sort_values(["tahun", "jumlah_rumah"], ascending=[True, False]))
print(ring.to_string(index=False))

kosong = (eksternal.groupby("_tahun")
          .apply(lambda g: sorted(g.loc[~g["_kec"].isin(hasil.loc[cocok & (hasil["_tahun"] == g.name), "_kec"]), "Kecamatan"]), include_groups=False))
print("\nKecamatan yang belum punya data rumah:")
for th, daftar in kosong.items():
    print(f"  {th}: {len(daftar)} kecamatan -> {', '.join(daftar) if daftar else '-'}")

print(f"\nTersimpan: {OUTPUT}")