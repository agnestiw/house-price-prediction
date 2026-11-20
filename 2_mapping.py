from pathlib import Path
import re
import pandas as pd

INPUT = "output/rumah123_surabaya_2024-2026.csv"
OUTPUT = (
    "output/hasil-mapping-kpr-kecamatan.csv"  # mapping & ringkasan hanya ditampilkan di terminal
)
# ===========================================================================
# 1. REFERENSI WILAYAH ADMINISTRASI KOTA SURABAYA
# ===========================================================================
WILAYAH_KECAMATAN = {
    "Surabaya Pusat": ["Tegalsari", "Genteng", "Bubutan", "Simokerto"],
    "Surabaya Utara": ["Bulak", "Kenjeran", "Semampir", "Pabean Cantian", "Krembangan"],
    "Surabaya Timur": [
        "Gubeng",
        "Gunung Anyar",
        "Sukolilo",
        "Tambaksari",
        "Mulyorejo",
        "Rungkut",
        "Tenggilis Mejoyo",
    ],
    "Surabaya Selatan": [
        "Wonokromo",
        "Wonocolo",
        "Wiyung",
        "Karang Pilang",
        "Jambangan",
        "Gayungan",
        "Dukuh Pakis",
        "Sawahan",
    ],
    "Surabaya Barat": [
        "Benowo",
        "Pakal",
        "Asemrowo",
        "Sukomanunggal",
        "Tandes",
        "Sambikerep",
        "Lakarsantri",
    ],
}
# 31 kecamatan -> 154 kelurahan
KELURAHAN = {
    "Asemrowo": ["Asemrowo", "Genting Kalianak", "Tambak Sarioso"],
    "Benowo": ["Kandangan", "Romokalisari", "Sememi", "Tambak Osowilangun"],
    "Bubutan": ["Alun-Alun Contong", "Bubutan", "Gundih", "Jepara", "Tembok Dukuh"],
    "Bulak": ["Bulak", "Kedung Cowek", "Kenjeran", "Sukolilo Baru"],
    "Dukuh Pakis": ["Dukuh Kupang", "Dukuh Pakis", "Gunung Sari", "Pradah Kalikendal"],
    "Gayungan": ["Dukuh Menanggal", "Gayungan", "Ketintang", "Menanggal"],
    "Genteng": ["Embong Kaliasin", "Genteng", "Kapasari", "Ketabang", "Peneleh"],
    "Gubeng": ["Airlangga", "Barata Jaya", "Gubeng", "Kertajaya", "Mojo", "Pucang Sewu"],
    "Gunung Anyar": ["Gunung Anyar", "Gunung Anyar Tambak", "Rungkut Menanggal", "Rungkut Tengah"],
    "Jambangan": ["Jambangan", "Karah", "Kebonsari", "Pagesangan"],
    "Karang Pilang": ["Karang Pilang", "Kebraon", "Kedurus", "Warugunung"],
    "Kenjeran": ["Bulak Banteng", "Tambak Wedi", "Tanah Kali Kedinding", "Sidotopo Wetan"],
    "Krembangan": ["Dupak", "Kemayoran", "Krembangan Selatan", "Morokrembangan", "Perak Barat"],
    "Lakarsantri": [
        "Bangkingan",
        "Jeruk",
        "Lakarsantri",
        "Lidah Kulon",
        "Lidah Wetan",
        "Sumur Welut",
    ],
    "Mulyorejo": [
        "Dukuh Sutorejo",
        "Kalijudan",
        "Kalisari",
        "Kejawan Putih Tambak",
        "Manyar Sabrangan",
        "Mulyorejo",
    ],
    "Pabean Cantian": [
        "Bongkaran",
        "Krembangan Utara",
        "Nyamplungan",
        "Perak Timur",
        "Perak Utara",
    ],
    "Pakal": ["Babat Jerawat", "Benowo", "Pakal", "Sumberejo"],
    "Rungkut": [
        "Kali Rungkut",
        "Kedung Baruk",
        "Medokan Ayu",
        "Penjaringan Sari",
        "Rungkut Kidul",
        "Wonorejo",
    ],
    "Sambikerep": ["Bringin", "Lontar", "Made", "Sambikerep"],
    "Sawahan": ["Banyu Urip", "Kupang Krajan", "Pakis", "Petemon", "Putat Jaya", "Sawahan"],
    "Semampir": ["Ampel", "Pegirian", "Sidotopo", "Ujung", "Wonokusumo"],
    "Simokerto": ["Kapasan", "Sidodadi", "Simokerto", "Simolawang", "Tambakrejo"],
    "Sukolilo": [
        "Gebang Putih",
        "Keputih",
        "Klampis Ngasem",
        "Medokan Semampir",
        "Menur Pumpungan",
        "Nginden Jangkungan",
        "Semolowaru",
    ],
    "Sukomanunggal": [
        "Putat Gede",
        "Simomulyo",
        "Simomulyo Baru",
        "Sonokwijenan",
        "Sukomanunggal",
        "Tanjungsari",
    ],
    "Tambaksari": [
        "Dukuh Setro",
        "Gading",
        "Kapas Madya",
        "Pacar Kembang",
        "Pacar Keling",
        "Ploso",
        "Rangkah",
        "Tambaksari",
    ],
    "Tandes": [
        "Balongsari",
        "Banjar Sugihan",
        "Karang Poh",
        "Manukan Kulon",
        "Manukan Wetan",
        "Tandes",
    ],
    "Tegalsari": ["Dr. Sutomo", "Kedungdoro", "Keputran", "Tegalsari", "Wonorejo"],
    "Tenggilis Mejoyo": ["Kendangsari", "Kutisari", "Panjang Jiwo", "Tenggilis Mejoyo"],
    "Wiyung": ["Babatan", "Balas Klumprik", "Jajar Tunggal", "Wiyung"],
    "Wonocolo": ["Bendul Merisi", "Jemur Wonosari", "Margorejo", "Sidosermo", "Siwalankerto"],
    "Wonokromo": ["Darmo", "Jagir", "Ngagel", "Ngagelrejo", "Sawunggaling", "Wonokromo"],
}
# Ejaan lain / nama lama -> nama resmi
ALIAS_KECAMATAN = {
    "tenggilis": "Tenggilis Mejoyo",
    "karangpilang": "Karang Pilang",
    "pabeancantikan": "Pabean Cantian",
}  # rumah123 menulis "Pabean Cantikan"
ALIAS_KELURAHAN = {
    "patemon": "Petemon",
    "prapen": "Panjang Jiwo",
    "klampis": "Klampis Ngasem",
    "drsoetomo": "Dr. Sutomo",
    "kejawenputihtambak": "Kejawan Putih Tambak",
    "jemurwonosari": "Jemur Wonosari",
    "pradahkalikendal": "Pradah Kalikendal",
}
# Kelurahan dengan nama sama di dua kecamatan -> kecamatan default bila judul tidak memberi petunjuk
DEFAULT_KELURAHAN_GANDA = {
    "wonorejo": "Rungkut"
}  # Wonorejo Rungkut jauh lebih umum di listing rumah
# Nama kelurahan yang juga kata umum / nama tempat terkenal -> tidak dipakai saat memindai judul
KATA_UMUM = {
    "made",
    "jeruk",
    "mojo",
    "darmo",
    "gading",
    "ujung",
    "karah",
    "jepara",
    "ampel",
    "jagir",
    "dupak",
    "airlangga",
    "ploso",
    "ketabang",
    "peneleh",
    "gundih",
    "kapasan",
    "pakis",
}
# Kata yang menandai tempat di sekitar (bukan lokasi rumah), judul dipotong mulai kata ini
KATA_DEKAT = (
    r"\b(dekat|dkt|dket|deket|sebelah|samping|nempel|selangkah|akses|menit|5 mnt|belakang)\b"
)
# ===========================================================================
# 2. AREA / KAWASAN PERUMAHAN (bukan kelurahan resmi) - dirinci
#    default    : (kecamatan, kelurahan|None) bila judul tidak memberi petunjuk;
#                 None = area membentang >1 kecamatan, jangan menebak -> kecamatan "-" (wilayah diisi dari key "wilayah")
#    kelurahan  : kelurahan yang tercakup area tsb (dipindai di judul)
#    kecamatan  : kecamatan tambahan yang mungkin (dipindai di judul)
#    sub        : (kata kunci di judul, kecamatan, kelurahan|None) -> diperiksa paling awal
# ===========================================================================
AREA = {
    "citraland": dict(
        default=("Sambikerep", None),
        kelurahan=[
            "Made",
            "Lontar",
            "Bringin",
            "Lidah Kulon",
            "Lidah Wetan",
            "Jeruk",
            "Sumur Welut",
        ],
        kecamatan=["Lakarsantri"],
        catatan="CitraLand Surabaya: inti di Sambikerep (Made/Lontar/Bringin), sebagian Lakarsantri",
    ),
    "pakuwonindah": dict(
        default=("Sambikerep", None),
        kelurahan=["Lontar", "Babatan", "Pradah Kalikendal"],
        sub=[("wisata bukit mas", "Wiyung", "Babatan")],
        catatan="Pakuwon Indah (Jl. Puncak Indah Lontar); sebagian blok tercatat Babatan, Wiyung",
    ),
    "pakuwoncity": dict(
        default=("Mulyorejo", None),
        kelurahan=["Kejawan Putih Tambak", "Kalisari", "Keputih"],
        catatan="Pakuwon City: Kejawan Putih Tambak/Kalisari (Mulyorejo), sebagian Keputih (Sukolilo)",
    ),
    "pakuwon": dict(
        default=("Tandes", None),
        sub=[
            ("grand pakuwon", "Tandes", None),
            ("wisata bukit mas", "Wiyung", "Babatan"),
            ("pakuwon city", "Mulyorejo", None),
            ("pakuwon indah", "Sambikerep", None),
        ],
        catatan="Nama umum Pakuwon; dibedakan per proyek dari judul (Grand Pakuwon = Tandes)",
    ),
    "manyar": dict(
        default=("Sukolilo", None),
        kelurahan=["Manyar Sabrangan", "Menur Pumpungan", "Klampis Ngasem", "Mojo"],
        sub=[
            ("manyar kertoarjo", "Gubeng", "Mojo"),
            ("kertoarjo", "Gubeng", "Mojo"),
            ("manyar kertoadi", "Sukolilo", "Klampis Ngasem"),
            ("kertoadi", "Sukolilo", "Klampis Ngasem"),
            ("tirtoyoso", "Sukolilo", "Menur Pumpungan"),
            ("tirtoasri", "Sukolilo", "Menur Pumpungan"),
            ("tirtomoyo", "Sukolilo", "Menur Pumpungan"),
            ("manyar rejo", "Sukolilo", "Menur Pumpungan"),
            ("manyar jaya", "Sukolilo", None),
            ("manyar kartika", "Sukolilo", None),
            ("manyar sabrangan", "Mulyorejo", "Manyar Sabrangan"),
        ],
        catatan="Area Manyar membentang Gubeng/Sukolilo/Mulyorejo; mayoritas ruas Manyar Tirto*/Rejo di Menur Pumpungan",
    ),
    "dharmahusada": dict(
        default=("Mulyorejo", None),
        kelurahan=["Mojo", "Mulyorejo", "Kalijudan"],
        sub=[
            ("raya dharmahusada", "Gubeng", "Mojo"),
            ("dharmahusada indah", "Mulyorejo", None),
            ("dharmahusada mas", "Mulyorejo", None),
            ("dharmahusada regency", "Mulyorejo", None),
        ],
        catatan="Perumahan Dharmahusada Indah/Mas = Mulyorejo; ruas Jl. Raya Dharmahusada = Mojo, Gubeng",
    ),
    "mojoarum": dict(
        default=("Gubeng", "Mojo"),
        sub=[("dharmahusada regency", "Mulyorejo", None)],
        catatan="Area Mojo Arum (Kel. Mojo); listing Dharmahusada Regency diarahkan ke Mulyorejo",
    ),
    "darmopermai": dict(
        default=("Dukuh Pakis", None),
        kelurahan=["Pradah Kalikendal", "Putat Gede", "Babatan"],
        kecamatan=["Sukomanunggal", "Tandes"],
        sub=[
            ("wisata bukit mas", "Wiyung", "Babatan"),
            ("darmo indah", "Tandes", None),
            ("pradah", "Dukuh Pakis", "Pradah Kalikendal"),
        ],
        catatan="Darmo Permai: inti Pradah Kalikendal (Dukuh Pakis), tepi barat menyentuh Sukomanunggal/Tandes",
    ),
    "graha": dict(
        default=("Dukuh Pakis", "Pradah Kalikendal"),
        kelurahan=["Pradah Kalikendal", "Babatan"],
        sub=[("dian istana", "Wiyung", None)],
        catatan="Graha Famili (Pradah Kalikendal, Dukuh Pakis); Dian Istana = Wiyung",
    ),
    "wisatabukitmas": dict(default=("Wiyung", "Babatan"), catatan="Wisata Bukit Mas, Kel. Babatan"),
    "galaxy": dict(
        default=("Sukolilo", None),
        kelurahan=["Klampis Ngasem", "Semolowaru"],
        catatan="Galaxy Bumi Permai (Klampis Ngasem/Semolowaru)",
    ),
    "mulyosari": dict(
        default=("Mulyorejo", None),
        kelurahan=["Kalisari", "Kalijudan", "Mulyorejo", "Dukuh Sutorejo"],
        sub=[("galaxy bumi permai", "Sukolilo", None), ("galaxi bumi permai", "Sukolilo", None)],
        catatan="Area Mulyosari seluruhnya di Kec. Mulyorejo",
    ),
    "sutorejo": dict(
        default=("Mulyorejo", None),
        kelurahan=["Dukuh Sutorejo", "Kalisari"],
        catatan="Area Sutorejo (Dukuh Sutorejo/Kalisari), Kec. Mulyorejo",
    ),
    "jemursari": dict(
        default=("Wonocolo", None),
        kelurahan=["Jemur Wonosari", "Sidosermo", "Kendangsari", "Panjang Jiwo", "Kutisari"],
        kecamatan=["Tenggilis Mejoyo"],
        sub=[
            ("jemur andayani", "Wonocolo", None),
            ("saronojiwo", "Tenggilis Mejoyo", "Panjang Jiwo"),
        ],
        catatan="Raya Jemursari: sisi barat Wonocolo, sisi timur Tenggilis Mejoyo",
    ),
    "ngagel": dict(
        default=("Wonokromo", "Ngagel"),
        sub=[
            ("ngagel jaya", "Gubeng", None),
            ("ngagel rejo", "Wonokromo", "Ngagelrejo"),
            ("ngagelrejo", "Wonokromo", "Ngagelrejo"),
        ],
        catatan="Kel. Ngagel ada di Wonokromo, tapi Ngagel Jaya (yang sering muncul di listing) masuk Gubeng",
    ),
    "rungkutindustri": dict(
        default=("Rungkut", None),
        kelurahan=["Kali Rungkut", "Kedung Baruk", "Rungkut Menanggal", "Rungkut Tengah"],
        catatan="Kawasan SIER Rungkut; sebagian di Gunung Anyar",
    ),
    "medokan": dict(
        default=("Rungkut", None),
        kelurahan=["Medokan Ayu", "Medokan Semampir"],
        catatan="Medokan Ayu (Rungkut) / Medokan Semampir (Sukolilo)",
    ),
    "kertajayaindah": dict(default=("Mulyorejo", "Manyar Sabrangan"), catatan="Kertajaya Indah"),
    "pantaimentari": dict(default=("Bulak", "Kenjeran"), catatan="Pantai Mentari"),
    "babatanpantai": dict(default=("Mulyorejo", None), catatan="Babatan Pantai, dekat Sutorejo"),
    "dianistana": dict(default=("Wiyung", None), catatan="Perumahan Dian Istana"),
    "bukitdarmogolf": dict(default=("Dukuh Pakis", None), catatan="Bukit Darmo Golf"),
    "buntaran": dict(default=("Tandes", None), catatan="Area Buntaran, Kec. Tandes"),
    # --- area lintas kecamatan: tanpa petunjuk di judul -> kecamatan '-' (tidak menebak) ---
    "margomulyo": dict(
        default=None,
        wilayah="Surabaya Barat",
        kecamatan=["Tandes", "Asemrowo"],
        sub=[("grand pakuwon", "Tandes", None)],
        catatan="Margomulyo (pergudangan/Jl. Raya Margomulyo) berbatasan Tandes & Asemrowo; keduanya Surabaya Barat",
    ),
    "kupang": dict(
        default=None,
        wilayah="Surabaya Selatan",
        kelurahan=["Kupang Krajan", "Dukuh Kupang"],
        sub=[
            ("kupang baru", "Sawahan", "Kupang Krajan"),
            ("kupang krajan", "Sawahan", "Kupang Krajan"),
            ("dukuh kupang", "Dukuh Pakis", "Dukuh Kupang"),
        ],
        catatan="Nama 'Kupang' ada di Sawahan (Kupang Krajan/Kupang Baru) & Dukuh Pakis (Dukuh Kupang); keduanya Surabaya Selatan",
    ),
    "perak": dict(
        default=None,
        wilayah="Surabaya Utara",
        kelurahan=["Perak Barat", "Perak Timur", "Perak Utara"],
        catatan="Perak Barat = Krembangan; Perak Timur/Utara = Pabean Cantian; keduanya Surabaya Utara",
    ),
}
# ===========================================================================
# 3. JALAN BESAR (lintas kecamatan) -> kecamatan '-'; wilayah diisi bila seluruh ruas di 1 wilayah
#    Selain daftar ini, lokasi diawali "Jl"/"Jalan"/"Raya " otomatis dianggap jalan.
# ===========================================================================
JALAN = {
    "raya darmo": "-",
    "mayjen sungkono": "Surabaya Selatan",
    "ambengan": "-",
    "pemuda": "Surabaya Pusat",
    "basuki rahmat": "Surabaya Pusat",
    "tunjungan": "Surabaya Pusat",
    "embong malang": "Surabaya Pusat",
    "urip sumoharjo": "-",
    "diponegoro": "-",
    "ahmad yani": "-",
    "a yani": "-",
    "hr muhammad": "-",
    "merr": "-",
    "ir soekarno": "-",
    "kenjeran raya": "-",
    "kertajaya raya": "-",
    "panglima sudirman": "Surabaya Pusat",
    "basuki rachmat": "Surabaya Pusat",  # ejaan lain dari Basuki Rahmat
    "middle east ring road": "-",  # nama lengkap MERR
    "tidar": "-",  # Jl. Tidar membentang Sawahan (Selatan) & Bubutan (Pusat)
}
# ===========================================================================
# 4. LUAR KOTA SURABAYA (kabupaten & kecamatan di Sidoarjo / Gresik / sekitar)
# ===========================================================================
LUAR_SURABAYA = {
    "sidoarjo",
    "gresik",
    "mojokerto",
    "bangkalan",
    "pasuruan",
    "lamongan",
    # Kab. Sidoarjo
    "waru",
    "taman",
    "gedangan",
    "sedati",
    "buduran",
    "candi",
    "porong",
    "krembung",
    "tulangan",
    "tanggulangin",
    "jabon",
    "krian",
    "balongbendo",
    "wonoayu",
    "tarik",
    "prambon",
    "sukodono",
    # Kab. Gresik
    "menganti",
    "driyorejo",
    "kedamean",
    "kebomas",
    "cerme",
    "benjeng",
    "bungah",
    "sidayu",
    "wringinanom",
    "duduksampeyan",
    "balongpanggang",
    "panceng",
    "dukun",
    "ujungpangkah",
}
TIDAK_SPESIFIK = {"surabayakota", "surabaya", "kotasurabaya"}


# ===========================================================================
# 5. FUNGSI BANTU
# ===========================================================================
def norm(s):
    """huruf kecil, hanya huruf/angka (tanpa spasi) -> kunci pencocokan"""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def norm_spasi(s):
    return " " + re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip() + " "


def ada(kata, teks):
    """kata ada di teks sebagai kata utuh; untuk nama panjang toleran spasi ('dharma husada' = 'dharmahusada')"""
    if norm_spasi(kata) in teks:
        return True
    k = norm(kata)
    return len(k) >= 8 and k in teks.replace(" ", "")


KEC2WIL = {k: w for w, ks in WILAYAH_KECAMATAN.items() for k in ks}
KEC_INDEX = {norm(k): k for k in KEC2WIL}
KEC_INDEX.update(ALIAS_KECAMATAN)
KEL_INDEX = {}  # norm(nama kelurahan) -> [(kelurahan, kecamatan), ...]
for kec, kels in KELURAHAN.items():
    for kel in kels:
        KEL_INDEX.setdefault(norm(kel), []).append((kel, kec))
for alias, kel in ALIAS_KELURAHAN.items():
    KEL_INDEX.setdefault(alias, KEL_INDEX[norm(kel)])
KEL2KEC = {kel: kec for kec, kels in KELURAHAN.items() for kel in kels}
# validasi referensi (salah ketik di AREA langsung ketahuan)
assert set(KELURAHAN) == set(KEC2WIL), "Daftar kecamatan KELURAHAN & WILAYAH_KECAMATAN tidak sama"
assert sum(map(len, KELURAHAN.values())) == 154, "Jumlah kelurahan Surabaya harus 154"
for a, d in AREA.items():
    for kel in d.get("kelurahan", []):
        assert kel in KEL2KEC, f"AREA {a}: kelurahan '{kel}' tidak ada di referensi"
    if d.get("default") is None:
        assert (
            d.get("wilayah") in WILAYAH_KECAMATAN
        ), f"AREA {a}: default=None wajib punya 'wilayah' yang valid"
    for kec in (
        d.get("kecamatan", [])
        + ([d["default"][0]] if d.get("default") else [])
        + [s[1] for s in d.get("sub", [])]
    ):
        assert kec in KEC2WIL, f"AREA {a}: kecamatan '{kec}' tidak ada di referensi"
AREA_INDEX = {norm(k): v for k, v in AREA.items()}
JALAN_INDEX = {norm(k): v for k, v in JALAN.items()}


def teks_judul(judul, link):
    """(teks utama, teks penuh). Teks utama = judul sebelum kata 'dekat/sebelah/...' + bagian lokasi di link
    (mis. 'surabaya-sukolilo' atau nama proyek 'wisata-bukit-mas'; slug judul 'hos123..' diabaikan).
    """
    j = str(judul).lower()
    utama = re.split(KATA_DEKAT, j)[0]
    bagian = str(link).lower().split("?")[0].split("/")[3:]
    slug = " ".join(
        b
        for b in bagian
        if b and not re.search(r"(hos|nps)\d+$", b) and b not in ("properti", "perumahan-baru")
    )
    return norm_spasi(utama + " " + slug), norm_spasi(j)


def petunjuk_wilayah(teks_penuh):
    return [w for w in WILAYAH_KECAMATAN if ada(w, teks_penuh)]


def pilih_dari_judul(kandidat_kel, kandidat_kec, utama, penuh):
    """Cari kelurahan/kecamatan kandidat di judul. Return (kecamatan, kelurahan|None, dasar) atau None."""
    # a) nama kelurahan kandidat (kecuali kata umum & yang namanya sama dengan kecamatan)
    kel_ditemukan = [
        k
        for k in kandidat_kel
        if norm(k) not in KATA_UMUM and norm(k) not in KEC_INDEX and ada(k, utama)
    ]
    kec_dari_kel = {KEL2KEC[k] for k in kel_ditemukan}
    if len(kec_dari_kel) == 1:
        kel = kel_ditemukan[0] if len(kel_ditemukan) == 1 else None
        return kec_dari_kel.pop(), kel, "nama kelurahan di judul"
    # b) nama kecamatan kandidat
    kec_ditemukan = [k for k in kandidat_kec if ada(k, utama)]
    if len(kec_ditemukan) == 1:
        return kec_ditemukan[0], None, "nama kecamatan di judul"
    # c) petunjuk 'Surabaya Timur/Barat/...'
    wil = petunjuk_wilayah(penuh)
    cocok = [k for k in kandidat_kec if KEC2WIL[k] in wil]
    if len(wil) == 1 and len(cocok) == 1:
        return cocok[0], None, f"petunjuk '{wil[0]}' di judul"
    return None


def scan_kelurahan_dalam_kecamatan(kec, utama):
    """Detail kelurahan untuk lokasi level kecamatan / area, bila disebut di judul."""
    hits = [
        k
        for k in KELURAHAN.get(kec, [])
        if norm(k) not in KATA_UMUM and norm(k) not in KEC_INDEX and ada(k, utama)
    ]
    return hits[0] if len(hits) == 1 else None


def tentukan_lokasi(lokasi, judul, link):
    """return (tipe_lokasi, kelurahan, kecamatan, wilayah, dasar_penentuan)"""
    lok = str(lokasi).strip()
    if lok in ("", "-"):
        return "Tidak ada data", "-", "-", "Tidak ada data", "lokasi kosong"
    nama = re.sub(r",\s*surabaya\s*$", "", lok, flags=re.I).strip()
    key = norm(nama)
    utama, penuh = teks_judul(judul, link)

    def hasil(tipe, kel, kec, dasar):
        return tipe, kel or "-", kec, KEC2WIL[kec], dasar

    # 1) tidak spesifik
    if key in TIDAK_SPESIFIK:
        return "Tidak spesifik", "-", "-", "Tidak spesifik", "label umum rumah123"
    # 2) jalan besar -> '-'
    key_jalan = norm(re.sub(r"^(jl|jln|jalan)\b\\.?", "", nama.lower()))
    if key_jalan in JALAN_INDEX or re.match(r"^(jl|jln|jalan|raya)\b", nama.lower()):
        return "Jalan", "-", "-", JALAN_INDEX.get(key_jalan, "-"), "nama jalan lintas kecamatan"
    # 3) kecamatan
    if key in KEC_INDEX:
        kec = KEC_INDEX[key]
        return hasil("Kecamatan", scan_kelurahan_dalam_kecamatan(kec, utama), kec, "nama kecamatan")
    # 4) area / kawasan perumahan (rinci)
    if key in AREA_INDEX:
        a = AREA_INDEX[key]
        for kata, kec, kel in a.get("sub", []):
            if ada(kata, utama):
                return hasil("Area", kel, kec, f"sub-area '{kata}' di judul")
        kand_kel = a.get("kelurahan", [])
        kand_kec = list(
            dict.fromkeys(
                ([a["default"][0]] if a.get("default") else [])
                + [KEL2KEC[k] for k in kand_kel]
                + a.get("kecamatan", [])
                + [s[1] for s in a.get("sub", [])]
            )
        )
        r = pilih_dari_judul(kand_kel, kand_kec, utama, penuh)
        if r:
            kec, kel, dasar = r
            return hasil("Area", kel, kec, dasar)
        if not a.get(
            "default"
        ):  # area lintas kecamatan & judul tidak memberi petunjuk -> jangan menebak
            return (
                "Area",
                "-",
                "-",
                a["wilayah"],
                "area lintas kecamatan, tidak ada petunjuk di judul: " + a.get("catatan", ""),
            )
        kec, kel = a["default"]
        return hasil("Area", kel, kec, "default area: " + a.get("catatan", ""))
    # 5) kelurahan resmi
    if key in KEL_INDEX:
        pasangan = KEL_INDEX[key]
        if len(pasangan) == 1:
            kel, kec = pasangan[0]
            return hasil("Kelurahan", kel, kec, "nama kelurahan resmi")
        # nama kelurahan ganda (mis. Wonorejo: Rungkut & Tegalsari)
        kand_kec = [kec for _, kec in pasangan]
        r = pilih_dari_judul([], kand_kec, utama, penuh)
        kec = r[0] if r else DEFAULT_KELURAHAN_GANDA.get(key)
        if kec:
            kel = next(k for k, c in pasangan if c == kec)
            return hasil("Kelurahan", kel, kec, r[2] if r else "kelurahan ganda, default")
        return "Kelurahan", pasangan[0][0], "-", "-", "kelurahan ganda, tidak bisa dipastikan"
    # 6) luar Surabaya
    if key in LUAR_SURABAYA:
        return (
            "Luar Surabaya",
            "-",
            "-",
            "Luar Surabaya",
            "kecamatan/kabupaten di luar Kota Surabaya",
        )
    # 7) cocok sebagian: 'nginden' -> Nginden Jangkungan, 'manukan' -> Manukan Kulon/Wetan
    if len(key) >= 5:
        kel_cocok = [
            (kel, kec)
            for k, ps in KEL_INDEX.items()
            if key in k or k in key
            for kel, kec in ps
            if len(k) >= 5
        ]
        kec_set = {kec for _, kec in kel_cocok}
        if len(kec_set) == 1:
            kel = kel_cocok[0][0] if len({k for k, _ in kel_cocok}) == 1 else None
            return hasil("Kelurahan", kel, kec_set.pop(), "nama mirip kelurahan resmi")
        kec_cocok = [kec for k, kec in KEC_INDEX.items() if key.startswith(k)]
        if len(kec_cocok) == 1:  # mis. 'Rungkut Asri' -> Rungkut
            kec = kec_cocok[0]
            return hasil(
                "Area",
                scan_kelurahan_dalam_kecamatan(kec, utama),
                kec,
                "nama diawali nama kecamatan",
            )
    # 8) tidak dikenali
    return "Tidak dikenali", "-", "-", "Tidak dikenali", "tambahkan ke AREA/JALAN/LUAR_SURABAYA"


# ===========================================================================
# 6. KLASIFIKASI KONDISI BANGUNAN, TIPE LINGKUNGAN, DAN KPR
# ===========================================================================
# Klasifikasi berbasis petunjuk eksplisit, bukan harga/luas rumah.
# Kondisi tanpa bukti diberi "tidak diketahui", bukan otomatis dianggap lama.
def teks_fitur(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower().replace("\\n", " ")).strip()


def petunjuk_positif(pattern, text):
    for match in re.finditer(pattern, text):
        prefix = text[max(0, match.start() - 55) : match.start()]
        if not re.search(
            r"\b(?:bukan|tidak|non|dekat|dkt|sebelah|samping)\s+(?:\w+\s+){0,3}$", prefix
        ):
            return True
    return False


def tentukan_kondisi_bangunan(row):
    kondisi = teks_fitur(row.get("kondisi", ""))
    if kondisi in {"baru", "baru dibangun", "brand new"}:
        return "baru"
    if kondisi in {"lama", "bekas", "second", "butuh renovasi", "perlu renovasi"}:
        return "lama"
    text = " | ".join(teks_fitur(row.get(c, "")) for c in ["judul", "overview", "deskripsi"])
    baru = petunjuk_positif(
        r"\b(?:rumah\s+(?:brand\s+)?baru|bangunan\s+baru|baru\s+dibangun|brand\s+new|belum\s+pernah\s+dihuni)\b",
        text,
    )
    lama = petunjuk_positif(
        r"\b(?:rumah\s+(?:lama|bekas|second)|bangunan\s+(?:lama|tua)|butuh\s+renovasi|perlu\s+renovasi)\b",
        text,
    )
    # Renovasi baru tidak berarti bangunan baru; konflik perlu pemeriksaan.
    if baru == lama:
        return "tidak diketahui"
    return "baru" if baru else "lama"


def tentukan_tipe_lingkungan(row):
    text = " | ".join(
        teks_fitur(row.get(c, "")) for c in ["lokasi", "judul", "overview", "deskripsi"]
    )
    kampung = petunjuk_positif(r"\b(?:kampung|perkampungan)\b", text)
    perumahan = petunjuk_positif(
        r"\b(?:perumahan|perum|cluster|klaster|komplek\s+perumahan|kompleks\s+perumahan)\b", text
    )
    if kampung == perumahan:
        return "tidak diketahui"
    return "kampung" if kampung else "perumahan"


def tentukan_kpr(overview):
    teks = str(overview).strip()
    if teks in ("", "-"):
        return "Tidak Diketahui"
    return "Bisa" if "KPR" in teks.upper() else "Tidak"


def cetak_judul(teks):
    """Cetak judul bagian dengan format seragam."""
    print("\n" + "=" * 75)
    print(teks)
    print("=" * 75)


def cetak_distribusi(nilai):
    """Tampilkan kategori, jumlah, dan persentase tanpa metadata Series."""
    jumlah = nilai.fillna("(belum teridentifikasi)").value_counts()
    tabel = jumlah.rename_axis("Kategori").reset_index(name="Jumlah")
    tabel["Persentase"] = tabel["Jumlah"].map(
        lambda n: f"{100 * n / len(nilai):.2f}%" if len(nilai) else "0.00%"
    )
    print(tabel.to_string(index=False))


def main():
    df = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    # rapikan spasi & baris baru di lokasi (mis. 'Rungkut, Surabaya\n' -> 'Rungkut, Surabaya')
    df["lokasi"] = df["lokasi"].str.replace(r"\s+", " ", regex=True).str.strip()
    # KPR (3 kategori)
    #   Bisa            : overview ada dan menyebut "KPR"
    #   Tidak           : overview ada tetapi tidak menyebut "KPR" (mis. hanya "Cash Keras")
    #   Tidak Diketahui : overview kosong / "-" (informasi cara pembayaran tidak tersedia)
    df["KPR"] = df["overview"].apply(tentukan_kpr)
    df["kondisi bangunan"] = df.apply(tentukan_kondisi_bangunan, axis=1)
    df["tipe lingkungan"] = df.apply(tentukan_tipe_lingkungan, axis=1)

    # lokasi -> tipe, kelurahan, kecamatan, wilayah
    hasil = [
        tentukan_lokasi(l, j, u)
        for l, j, u in zip(df["lokasi"], df["judul"], df["link detail rumah"])
    ]
    df["tipe_lokasi"], df["kelurahan"], df["kecamatan"], df["wilayah"], dasar = map(
        list, zip(*hasil)
    )
    # urutan kolom: KPR setelah overview; kolom lokasi baru setelah lokasi
    # tipe_lokasi, kelurahan, wilayah hanya dipakai untuk info di terminal, tidak disimpan ke CSV
    baru = ["KPR", "tipe_lokasi", "kelurahan", "kecamatan", "wilayah"]
    cols = [c for c in df.columns if c not in baru]
    cols.insert(cols.index("overview") + 1, "KPR")
    cols.insert(cols.index("lokasi") + 1, "kecamatan")
    out = df[cols]
    # tabel mapping (untuk pengecekan) & ringkasan -> dicetak ke terminal
    map_df = (
        df.assign(dasar_penentuan=dasar)
        .groupby(["lokasi", "tipe_lokasi", "kelurahan", "kecamatan", "wilayah", "dasar_penentuan"])
        .size()
        .reset_index(name="jumlah_baris")
        .sort_values(["lokasi", "jumlah_baris"], ascending=[True, False])
    )
    valid = df[df["kecamatan"] != "-"]
    ring = (
        valid.groupby(["wilayah", "kecamatan"])
        .size()
        .reset_index(name="jumlah_baris")
        .sort_values(["wilayah", "jumlah_baris"], ascending=[True, False])
    )
    ring["persen_dari_baris_terpetakan"] = (ring["jumlah_baris"] / len(valid) * 100).round(1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    cetak_judul("1. Pemetaan lokasi")
    print(map_df.to_string(index=False))
    cetak_judul("2. Ringkasan kecamatan")
    print(ring.to_string(index=False))
    for nomor, (kolom, label) in enumerate(
        [
            ("tipe_lokasi", "Tipe lokasi"),
            ("wilayah", "Wilayah"),
            ("KPR", "KPR"),
            ("kondisi bangunan", "Kondisi bangunan"),
            ("tipe lingkungan", "Tipe lingkungan"),
        ],
        start=3,
    ):
        cetak_judul(f"{nomor}. {label}")
        cetak_distribusi(df[kolom])

    tak_kenal = df.loc[df["tipe_lokasi"] == "Tidak dikenali", "lokasi"].unique()
    if len(tak_kenal):
        print(
            "\n[PERHATIAN] Lokasi belum dikenali, tambahkan ke AREA/JALAN/LUAR_SURABAYA:",
            list(tak_kenal),
        )
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nTersimpan: {OUTPUT}")


if __name__ == "__main__":
    main()
