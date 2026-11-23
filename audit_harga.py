"""Audit data listing: kolom utama dibandingkan dengan angka yang tertulis di deskripsi.

ATURAN UTAMA
    Bila nilai kolom utama BERBEDA dari nilai di deskripsi, nilai DESKRIPSI yang dipercaya
    (agen mengetik deskripsi lebih teliti daripada kolom form; salah desimal seperti
    'Rp 285 Miliar' padahal deskripsi menulis '2,85M' paling sering terjadi di kolom form).

KOLOM YANG DIAUDIT
    harga, luas tanah, luas bangunan, kamar tidur, kamar mandi, jumlah lantai, daya listrik, sertifikat

PENGAMAN (agar parser tidak merusak data yang benar)
    1. Dua nilai dianggap sama bila selisihnya masih dalam TOLERANSI (pembulatan, nego, dimensi 6x12 vs 75 m2).
    2. Koreksi otomatis HANYA bila deskripsi menyebut tepat SATU nilai yang jelas. Bila deskripsi menyebut
       beberapa nilai berbeda (mis. harga lama dan harga baru), tidak ada yang diubah: hanya dicatat
       di 'audit_perlu_periksa.csv'.
    3. Harga yang diawali 'mulai/start/terendah/dari', berakhiran '++' atau 'an', cicilan, DP, sewa,
       harga per m2, dan angka jarak/lebar jalan ('row 8m', '200m ke mal') tidak dianggap harga.
    4. Bila deskripsi sendiri tidak konsisten (luas '20m2' tetapi dimensi '10x20'), keduanya dianggap
       kandidat sehingga tidak ada koreksi otomatis.
    5. Semua perubahan dicatat lengkap dengan bukti teks dan tautan di folder keluaran.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

# ===========================================================================
# KONFIGURASI
# ===========================================================================
# Selisih relatif yang masih dianggap "sama". Isi 0 untuk memperlakukan SETIAP selisih sebagai beda.
TOLERANSI = {
    "harga": 0.25,  # 5-25% hampir selalu harga turun/nego (arah koreksi 85% naik); isi 0 = percaya deskripsi mutlak
    "luas tanah": 0.10,
    "luas bangunan": 0.10,
    "daya listrik": 0.05,
    "kamar tidur": 0.0,
    "kamar mandi": 0.0,
    "jumlah lantai": 0.0,
}
TOLERANSI_ABS = {"kamar tidur": 1, "kamar mandi": 1}  # konvensi '4+1' (kamar ART) membuat selisih 1 wajar
ISI_JIKA_KOSONG = True  # kolom kosong ('-') atau tidak jelas ('Lainnya') diisi dari deskripsi
HARGA_MIN, HARGA_MAX = 2e8, 5e11  # rentang harga wajar (Rp)
LT_MIN, LT_MAX = 10, 100_000
LB_MIN, LB_MAX = 15, 50_000
KONSISTENSI_DIMENSI = 0.15  # luas vs panjang x lebar boleh berbeda sampai 15%
RASIO_SELISIH_BESAR = 5  # harga dengan rasio > 5x atau < 1/5 dicatat sebagai 'selisih besar'

KOREKSI_MANUAL = {
    # property_id: {kolom: (nilai_lama, nilai_baru, alasan)}  -> dipakai bila koreksi otomatis tidak cukup
    "hos42622932": {
        "harga": ("Rp 285 Miliar", "Rp 2,85 Miliar", "Deskripsi listing menyatakan 2,85M nego; kolom harga 285 Miliar.")
    }
}

KATA_HARGA = r"(?:harga|hrg|price|dijual|jual|ask|asking|nego|nett?|open|only|cash|\bhj\b)"
KATA_BUKAN_HARGA = (
    r"(?:cicil|angsur|\bdp\b|uang muka|sewa|kontrak|komisi|fee|booking|\bbf\b|njop|pbb|"
    r"per\s*bulan|/\s*bulan|/\s*bln|per\s*tahun|/\s*tahun|/\s*thn|biaya\s*(?:renovasi|bangun|pajak)|pajak\s*\w*\s*(?:sekitar|rp))"
)
KATA_BATAS_HARGA = r"(?:mulai|start|terendah|termurah|dari|minimal|min\b|kisaran|sampai|s/d|hingga|max|<|>|≥|≤)"
PENANDA_UBAH = r"(?:->|→|=>|menjadi|jadi|now|sekarang|skrg|saat ini|kini|dilepas|discount|diskon|update|terkini|revisi)"
KATA_LAMA = r"(?:asli|awal|dulu|sebelumnya|normal|beli|modal|appraisal)"
KATA_PER_METER = r"(?:/\s*m|per\s*m|permeter|per\s*meter|hitung\s*tanah|x\s*luas|/\s*meter)"
KATA_JALAN = r"(?:row|jalan|jln|jl\b\.?|lebar|rj\b|akses|ruas|tinggi|panjang|jarak|radius)"

KOLOM_AUDIT = [
    "harga", "luas tanah", "luas bangunan", "kamar tidur",
    "kamar mandi", "jumlah lantai", "daya listrik", "sertifikat",
]


# ===========================================================================
# PARSER ANGKA
# ===========================================================================
def _norm(teks):
    t = str(teks).lower().replace("\u200b", "").replace("\xa0", " ").replace("_", " ")
    return " " + re.sub(r"\s+", " ", t) + " "


def _ctx(t, a, b, lebar=45):
    return re.sub(r"\s+", " ", t[max(0, a - lebar): b + lebar]).strip()


def angka_luas(s):
    """'2.500' -> 2500 ; '90,5' -> 90.5 ; '120' -> 120"""
    s = str(s).strip().rstrip(".,")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", s):
        return float(s.replace(".", "").replace(",", "."))
    if re.fullmatch(r"\d+[.,]\d+", s):
        return float(s.replace(",", "."))
    return float(s) if re.fullmatch(r"\d+", s) else np.nan


def angka_miliar(s):
    """Untuk satuan miliar/M: pemisah apa pun = desimal ('1,550' -> 1.55 ; '9, 5' -> 9.5)."""
    s = re.sub(r"\s+", "", str(s)).rstrip(".,")
    return float(s.replace(",", ".")) if re.fullmatch(r"\d+(?:[.,]\d+)?", s) else np.nan


def angka_juta(s):
    s = re.sub(r"\s+", "", str(s)).rstrip(".,")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
        return float(s.replace(".", ""))
    return float(s.replace(",", ".")) if re.fullmatch(r"\d+(?:[.,]\d+)?", s) else np.nan


# ===========================================================================
# EKSTRAKSI DARI DESKRIPSI -> [(nilai, kuat, potongan_teks)]
# ===========================================================================
def ambil_harga(teks):
    t = _norm(teks)
    hasil = []  # (awal, akhir, nilai, kuat, ctx)

    def tambah(nilai, a, b, unit_m):
        if np.isnan(nilai) or not (HARGA_MIN <= nilai <= HARGA_MAX):
            return
        sebelum, sesudah = t[max(0, a - 30): a], t[b: b + 22]
        if re.search(KATA_BUKAN_HARGA, sebelum[-14:]) or re.search(KATA_BUKAN_HARGA, sesudah[:12]):
            return
        if re.search(r"(?:turun|potong|diskon|discount|nego)\s*(?:harga)?\s*(?:sebesar|rp\.?)?\s*$", sebelum):
            return  # nominal penurunan, bukan harga
        pra = sebelum.rstrip(" :=*~-")
        if re.search(r"(?:" + KATA_BATAS_HARGA.replace("|<|>|≥|≤", "") + r"|(?<![-=])[<>≥≤])\s*(?:rp\.?)?\s*$", pra) or re.match(
            r"\s*(?:\+|an\b|ke atas)", sesudah
        ):
            return  # 'mulai 2m', '< 2m', '1m++', '2m an'
        if re.match(r"\s*" + KATA_PER_METER, sesudah) or re.search(
            KATA_PER_METER + r"\w*\s*[:=]?\s*(?:rp\.?)?\s*$", sebelum
        ):
            return  # harga per m2
        if unit_m and (
            re.search(KATA_JALAN + r"\s*(?:\w+\s*){0,1}[:=+/±\-\s]*$", sebelum)
            or re.match(r"\s*(?:ke|dari|dr|dengan|menuju|jarak|arah)\b", sesudah)
        ):
            return  # 'row jln 8 m', '100m ke oerr'
        ada_kata = bool(re.search(KATA_HARGA, sebelum)) or bool(re.search(PENANDA_UBAH, sebelum[-14:]))
        ada_kata = ada_kata or bool(re.match(r"\s*(?:nego|nett?|net|fix|cash|nb)\b", sesudah))
        if unit_m and not (ada_kata or re.search(r"rp\.?\s*$", sebelum)):
            return  # 'm' sendirian bisa 'meter'; wajib ada kata harga
        kuat = ada_kata and not re.search(KATA_LAMA, sebelum[-14:])  # 'harga asli 6m' = harga lama
        if re.search(r"(?:cuma|hanya|only)\s*$", pra) or re.search(r"sebelah|tetangga|unit lain|type lain|tipe lain", sebelum):
            kuat = False  # harga teaser / harga properti lain
        hasil.append((a, b, nilai, kuat, _ctx(t, a, b)))

    for m in re.finditer(r"(?<![\w.,/])(?:rp\.?\s*)?(\d+(?:[.,]\s?\d+)?)\s*(miliar|milyar|miliyar|m)\b(?!\s*[x×])", t):
        tambah(angka_miliar(m.group(1)) * 1e9, m.start(), m.end(), unit_m=(m.group(2) == "m"))
    for m in re.finditer(r"(?<![\w.,/])(?:rp\.?\s*)?(\d{1,3}(?:\.\d{3})+|\d+(?:[.,]\d+)?)\s*(juta|jt)\b", t):
        tambah(angka_juta(m.group(1)) * 1e6, m.start(), m.end(), unit_m=False)  # tanpa kata harga -> lemah
    for m in re.finditer(r"rp\.?\s*(\d{1,3}(?:[.,]\d{3}){2,})(?!\d)", t):
        tambah(float(re.sub(r"[.,]", "", m.group(1))), m.start(), m.end(), unit_m=False)
    for m in re.finditer(r"(?<![\w.,/])(\d+[.,]\d{1,3})\s*(?:nego|nett?|net|nb)\b", t):
        v = angka_miliar(m.group(1))
        if 0.2 <= v <= 300:
            tambah(v * 1e9, m.start(), m.end(), unit_m=False)
    # harga berubah ('15M -> 12.5M', 'menjadi 2.6m', 'harga 3m turun ...'):
    # (1) hanya harga SETELAH penanda terakhir yang berlaku; (2) harga yang diikuti penanda
    # perubahan <45 karakter sesudahnya dianggap lemah karena harga barunya mungkin tidak terbaca.
    penanda = [m.start() for m in re.finditer(PENANDA_UBAH + r"|\bturun\b", t)]
    if penanda and hasil:
        terakhir = max(penanda)
        akhir = []
        for p, e, v, k, c in hasil:
            if any(h[0] > terakhir for h in hasil) and p < terakhir:
                k = False
            if any(e <= x <= e + 45 for x in penanda) and not any(h[0] > e and h[0] > x for h in hasil for x in penanda if e <= x <= e + 45):
                k = False
            akhir.append((p, e, v, k, c))
        hasil = akhir
    return [(v, k, c) for _, _, v, k, c in sorted(hasil)]


_AREA = r"(\d[\d.,]*)\s*(?:m2|m²|m\^2|mtr|meter|m)?"
_DIM = r"(\d+(?:[.,]\d+)?)\s*(?:m|mtr)?\s*[x×*]\s*(\d+(?:[.,]\d+)?)\s*(?:m\b|mtr|meter)?"
_APPROX = r"(?:-/\+|\+/-|±|sekitar|kurang lebih|kl\.?|ca\.?|\+)?\s*"


def _hitung_luas(area, d1, d2, a, b, t, minimum, maksimum):
    """-> [(nilai, kuat, ctx)] ; area eksplisit dipakai, dimensi hanya untuk uji konsistensi."""
    luas = angka_luas(area) if area else np.nan
    prod = np.nan
    if d1 and d2:
        x, y = angka_luas(d1), angka_luas(d2)
        prod = x * y if not (np.isnan(x) or np.isnan(y)) else np.nan
    cand = []
    if not np.isnan(luas) and not np.isnan(prod) and abs(luas - prod) / prod > KONSISTENSI_DIMENSI:
        cand = [luas, prod]  # deskripsi tidak konsisten -> dua kandidat (ambigu)
    elif not np.isnan(luas):
        cand = [luas]
    elif not np.isnan(prod):
        cand = [prod]
    return [(v, True, _ctx(t, a, b)) for v in cand if minimum <= v <= maksimum]


def ambil_luas(teks):
    """-> (daftar_lt, daftar_lb)"""
    t = _norm(teks)
    lt, lb = [], []
    # a) 'LT/LB : 90m2 (6x15) / 130m2'
    blok = r"(?:" + _APPROX + _AREA + r"(?:\s*\(?\s*" + _DIM + r"\s*\)?)?)"
    pola_gab = re.compile(r"\blt\s*[/|]\s*lb\s*[:=]?\s*(\d[\d.,]*)\s*(?:m2|m²|m\^2|mtr|meter|m)?(?:\s*\(?\s*(\d+(?:[.,]\d+)?)\s*(?:m)?\s*[x×*]\s*(\d+(?:[.,]\d+)?)\s*(?:m\b)?\s*\)?)?\s*/\s*" + _APPROX + r"(\d[\d.,]*)")
    for m in pola_gab.finditer(t):
        lt += _hitung_luas(m.group(1), m.group(2), m.group(3), m.start(), m.end(), t, LT_MIN, LT_MAX)
        lb += _hitung_luas(m.group(4), None, None, m.start(), m.end(), t, LB_MIN, LB_MAX)
    # 'LB/LT : 115/81(6x13.5)' (urutan terbalik)
    pola_balik = re.compile(r"\blb\s*[/|]\s*lt\s*[:=]?\s*" + _APPROX + r"(\d[\d.,]*)\s*(?:m2|m²|m\^2|mtr|meter|m)?\s*[/|]\s*(\d[\d.,]*)\s*(?:m2|m²|m\^2|mtr|meter|m)?(?:\s*\(?\s*(\d+(?:[.,]\d+)?)\s*(?:m)?\s*[x×*]\s*(\d+(?:[.,]\d+)?)\s*(?:m\b)?\s*\)?)?")
    for m in pola_balik.finditer(t):
        lb += _hitung_luas(m.group(1), None, None, m.start(), m.end(), t, LB_MIN, LB_MAX)
        lt += _hitung_luas(m.group(2), m.group(3), m.group(4), m.start(), m.end(), t, LT_MIN, LT_MAX)
    t_sisa = pola_balik.sub(" ", pola_gab.sub(" ", t))
    t_sisa = re.sub(r"\b(?:lt\s*[/|]\s*lb|lb\s*[/|]\s*lt)\b", " ", t_sisa)  # pasangan tak terbaca -> abaikan
    # b) 'LT 72 (6x12)', 'LT 8x20', 'LT 5x20=100', 'luas tanah: 990m2 (20 x 49,5)'
    for kata, daftar, mn, mx in (
        (r"luas\s*tanah|\blt\b", lt, LT_MIN, LT_MAX),
        (r"luas\s*bangunan|\blb\b", lb, LB_MIN, LB_MAX),
    ):
        pola = re.compile(
            r"(?:" + kata + r")\.?\s*[:=]?\s*" + _APPROX
            + r"(?:" + _DIM + r"(?:\s*[=(]\s*(\d[\d.,]*)\s*(?:m2|m²|m)?)?"
            + r"|" + _AREA + r"(?:\s*\(?\s*" + _DIM.replace("(", "(", 1) + r"\s*\)?)?)"
        )
        for m in pola.finditer(t_sisa):
            sesudah = t_sisa[m.end(): m.end() + 14]
            if re.match(r"\s*\+\s*\d", sesudah) or re.match(r"\s*(?:/|per)\s*(?:lantai|lt\b)", sesudah):
                continue  # 'LB 80 + 60' atau 'LB 70/lantai' bukan total
            g = m.groups()
            if g[0] and g[1]:  # dimensi dulu: d1 d2 [area]
                daftar += _hitung_luas(g[2], g[0], g[1], m.start(), m.end(), t_sisa, mn, mx)
            else:  # area dulu: area [(d1 x d2)]
                daftar += _hitung_luas(g[3], g[4], g[5], m.start(), m.end(), t_sisa, mn, mx)
    return lt, lb


def ambil_kamar(teks, jenis):
    """KT / KM. '5+1' (utama+tambahan) menghasilkan 5 (kuat) dan 6 (lemah). Deskripsi per lantai
    ('lantai 1: 2 kt ... lantai 2: 2 kt') dianggap lemah karena bukan total.
    Langkah: (1) 'kt/km : 5+1/4+1'  (2) '4+1 kamar tidur 3+1 kamar mandi'  (3) 'kt 5+1'  (4) '3 kamar tidur'."""
    t = _norm(teks)
    t0 = t
    hasil = []
    ID = 0 if jenis == "kt" else 1
    LANTAI = r"(?<!\d)(?<!\d )(?:lantai|lt\.?)\s*(?:1|2|3|satu|dua|tiga|atas|bawah|dasar)\b|\b(?:atas|bawah)\b"

    rumit = bool(re.search(
        r"pembantu|\bart\b|security|satpam|paviliun|\bkost\b|kos-kosan|asrama|rumah induk"
        r"|\d\s*(?:kt|km)\s*\+\s*\d\s*(?:kt|km|kamar)|\d\s*(?:km\s*)?dalam.{0,8}\d\s*(?:km\s*)?luar", t0))  # total ambigu

    def angka(a, b, m):
        per_lantai = rumit or bool(re.search(LANTAI, t0[max(0, m.start() - 40): m.end() + 16]))
        a = int(a)
        hasil.append((a, not per_lantai, _ctx(t0, m.start(), m.end())))
        if b:
            hasil.append((a + int(b), False, _ctx(t0, m.start(), m.end())))

    def tutup(pola):
        nonlocal t
        ketemu = list(pola.finditer(t))
        t = pola.sub(lambda m: " " * len(m.group(0)), t)
        return ketemu

    p = r"(\d{1,2})(?:\s*\+\s*(\d{1,2}))?"
    # jarak ('2 km dari tol', 'jarak 3 km') bukan kamar mandi
    tutup(re.compile(r"(?<![a-z])\d+(?:[.,]\d+)?\s*km\b(?=\s*(?:dari|dr\b|ke\b|menuju|dengan|jauh|jarak|menit|an\b))"))
    tutup(re.compile(r"(?:jarak|sekitar|±|kurang lebih)\s*\d+(?:[.,]\d+)?\s*km\b"))
    for m in tutup(re.compile(r"\bkt\s*[/|]\s*km\s*[:=]?\s*" + p + r"\s*[/|]\s*" + p)):
        angka(*(m.group(1), m.group(2)) if ID == 0 else (m.group(3), m.group(4)), m)
    km_ok = r"(?:kamar\s*mandi|km\b(?!\s*(?:dari|dr\b|ke\b|menuju|jauh|jarak|menit)))"
    pasangan = re.compile(r"(?<![\w.,×/])(?<!x )(?<!× )" + p + r"\s*(?:kamar\s*tidur|\bkt\b)\s*[,/&]?\s*" + p + r"\s*" + km_ok)
    sah = []
    for m in pasangan.finditer(t):
        sebelum = t[: m.start()]
        if int(m.group(1)) > 15 or re.search(
            r"(?:luas\s*(?:tanah|bangunan)|\blb\b|\blt\b|dimensi|ukuran|jumlah)\s*[:=]?\s*$", sebelum
        ) or re.search(r"(?<!\d)(?<!\d )(?:lantai|tingkat)\s*[:=]?\s*$", sebelum):
            continue  # angka pertama milik 'lantai 1' / 'luas bangunan 70', bukan jumlah kamar
        sah.append(m)
    for m in sah:
        angka(*(m.group(1), m.group(2)) if ID == 0 else (m.group(3), m.group(4)), m)
    for m in sorted(sah, key=lambda x: -x.start()):
        t = t[: m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    for kata, idx in ((r"kamar\s*tidur|\bkt\b|\bk\.t\b", 0), (r"kamar\s*mandi|\bkm\b|\bk\.m\b", 1)):
        pola = re.compile(r"(?:" + kata + r")\.?\s*[:=]?\s*" + p + r"(?!\d)(?!\.\s+\d)(?!\s*(?:pembantu|art\b|asisten|supir|sopir|staf|gudang|dapur|ruang|carport|garasi|mobil|lantai|lt\b|unit|ac\b|taman|kolam|balkon|teras|musholla|paviliun|ktu))")
        for m in tutup(pola):
            if idx != ID:
                continue
            if re.search(r"(?:bisa|dapat|tambah|jadi|dibuat)\s*$", t0[max(0, m.start() - 12): m.start()]):
                continue
            if re.match(r"\s*(?:m\b|m2|m²|meter|menit|mnt)", t0[m.end(): m.end() + 8]):
                continue
            angka(m.group(1), m.group(2), m)
    rev = r"kamar\s*tidur|\bkt\b" if ID == 0 else r"kamar\s*mandi"
    pola_rev = r"(?<![\w.,×/])(?<!x )(?<!× )" + p + r"\s*(?:" + rev + r")(?:\s*\+\s*(\d{1,2})\s*(?:kamar|kt|room|art|pembantu|asisten))?(?!\s*(?:pembantu|art\b|asisten|supir|sopir|staf))"
    for m in re.finditer(pola_rev, t):
        angka(m.group(1), m.group(2) or m.group(3), m)
    return [h for h in hasil if 1 <= h[0] <= 30]


def ambil_lantai(teks):
    t = _norm(teks)
    hasil = []
    for m in re.finditer(r"(?<![\d.,x×/])(\d)\s*(?:lantai|lt\b|tingkat)(?!\s*[:=]?\s*\d)", t):
        hasil.append((int(m.group(1)), True, _ctx(t, m.start(), m.end())))
    return [h for h in hasil if 1 <= h[0] <= 5]


def ambil_daya(teks):
    t = _norm(teks)
    hasil = []
    for m in re.finditer(r"(?<![\w.,])(\d{1,3}(?:[.,]\d{3})+|\d{3,6})\s*(?:watt|w\b|va\b)", t):
        v = float(re.sub(r"[.,]", "", m.group(1)))
        daftar = bool(re.search(r"\d\s*(?:&|dan|/|atau|,)\s*$", t[max(0, m.start() - 8): m.start()]))
        if 450 <= v <= 33_000:
            hasil.append((v, not daftar, _ctx(t, m.start(), m.end())))
    return hasil


def ambil_sertifikat(teks):
    t = _norm(teks)
    pola = {
        "SHM": r"\bshm\b|sertifikat hak milik",
        "HGB": r"\bhgb\b|\bshgb\b|hak guna bangunan",
        "SHP": r"\bshp\b|hak pakai",
        "Girik": r"\bgirik\b|petok d|letter c",
    }
    hasil = []
    for kelas, p in pola.items():
        for m in re.finditer(p, t):
            if re.search(r"lainnya\s*\(|dll|ppjb\s*,", t[max(0, m.start() - 24): m.end() + 12]):
                continue  # teks template 'lainnya (ppjb, girik, adat, dll)'
            if re.search(r"(?:bukan|non|tidak|belum|bisa|dapat|upgrade|naik|ke|jadi|proses|atau)\s*(?:\w+\s*){0,2}$", t[max(0, m.start() - 28): m.start()]):
                continue
            hasil.append((kelas, True, _ctx(t, m.start(), m.end())))
    return hasil


# ===========================================================================
# NILAI KOLOM UTAMA -> angka (NaN bila kosong / tidak jelas)
# ===========================================================================
def kolom_harga(s):
    s = str(s).lower()
    m = re.search(r"(\d[\d.,]*)\s*(miliar|milyar|juta|jt|m)?\b", s)
    if not m:
        return np.nan
    if m.group(2) in ("miliar", "milyar", "m"):
        return angka_miliar(m.group(1)) * 1e9
    if m.group(2) in ("juta", "jt"):
        return angka_juta(m.group(1)) * 1e6
    return float(re.sub(r"[.,]", "", m.group(1)))


def kolom_luas(s):
    m = re.search(r"\d[\d.,]*", str(s))
    return angka_luas(m.group(0)) if m else np.nan


def kolom_int(s):
    m = re.search(r"\d+", str(s))
    return float(m.group(0)) if m else np.nan


def kolom_daya(s):
    m = re.search(r"\d[\d.,]*", str(s))
    return float(re.sub(r"[.,]", "", m.group(0))) if m else np.nan


def kolom_sertifikat(s):
    u = str(s).upper()
    if "SHM" in u or "HAK MILIK" in u:
        return "SHM"
    if "HGB" in u or "GUNA BANGUNAN" in u:
        return "HGB"
    if "SHP" in u or "HAK PAKAI" in u:
        return "SHP"
    if re.search(r"GIRIK|PETOK|LETTER C", u):
        return "Girik"
    return None  # '-', 'Lainnya', 'PPJB' -> tidak jelas


# ===========================================================================
# NILAI -> TEKS (format sama dengan data asli agar 4_preprocessing.py tetap bisa membaca)
# ===========================================================================
def teks_harga(v):
    if v >= 1e9:
        return "Rp " + f"{v / 1e9:.3f}".rstrip("0").rstrip(".").replace(".", ",") + " Miliar"
    return "Rp " + f"{v / 1e6:.3f}".rstrip("0").rstrip(".").replace(".", ",") + " Juta"


def teks_luas(v):
    return f"{v:g}".replace(".", ",") + " m²"


def teks_int(v):
    return str(int(v))


def teks_daya(v):
    return f"{int(v):,}".replace(",", ".") + " Watt"


def teks_sert(v):
    return str(v)


KONFIG = {
    # kolom: (ekstraktor, parser kolom, pembuat teks)
    "harga": (ambil_harga, kolom_harga, teks_harga),
    "kamar tidur": (lambda d: ambil_kamar(d, "kt"), kolom_int, teks_int),
    "kamar mandi": (lambda d: ambil_kamar(d, "km"), kolom_int, teks_int),
    "jumlah lantai": (ambil_lantai, kolom_int, teks_int),
    "daya listrik": (ambil_daya, kolom_daya, teks_daya),
    "sertifikat": (ambil_sertifikat, kolom_sertifikat, teks_sert),
}


# ===========================================================================
# KEPUTUSAN
# ===========================================================================
def _unik(nilai, rel=0.01):
    """gabungkan nilai yang selisihnya < 1% (angka) ; kategori dibiarkan"""
    hasil = []
    for v in sorted(nilai, key=lambda x: (isinstance(x, str), x)):
        if hasil and not isinstance(v, str) and not isinstance(hasil[-1], str) and abs(v - hasil[-1]) <= rel * abs(hasil[-1]):
            continue
        hasil.append(v)
    return hasil


def _sama(a, b, tol, tol_abs=0):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(a - b) <= max(tol * abs(b), tol_abs, 1e-9)


def putuskan(nilai_kolom, kandidat, tol, kolom):
    tol_abs = TOLERANSI_ABS.get(kolom, 0)
    """-> (aksi, nilai_baru, daftar_nilai)
    aksi: 'konsisten' | 'koreksi' | 'isi' | 'ambigu' | 'tidak_ada'"""
    if not kandidat:
        return "tidak_ada", None, []
    semua = _unik([v for v, _, _ in kandidat])
    kuat = _unik([v for v, k, _ in kandidat if k])
    kolom_ada = not (nilai_kolom is None or (not isinstance(nilai_kolom, str) and np.isnan(nilai_kolom)))
    if kolom_ada and any(_sama(v, nilai_kolom, tol, tol_abs) for v in semua):
        return "konsisten", None, semua
    if len(kuat) != 1:
        return "ambigu", None, semua
    if not kolom_ada:
        return ("isi" if ISI_JIKA_KOSONG else "ambigu"), kuat[0], semua
    return "koreksi", kuat[0], semua


# ===========================================================================
# PROSES UTAMA
# ===========================================================================
def audit_dan_koreksi(df, folder):
    df = df.copy()
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    url = df.get("link detail rumah", pd.Series("", index=df.index)).fillna("")
    if "link arsip" in df:
        url = url.where(url.str.strip().ne(""), df["link arsip"].fillna(""))
    ids = url.str.extract(r"((?:hos|nps)\d+)", flags=re.I, expand=False).str.lower()
    deskripsi = df["deskripsi"].fillna("").astype(str) if "deskripsi" in df else pd.Series("", index=df.index)

    diterapkan, periksa = [], []
    hitung = {}  # (kolom, aksi) -> jumlah

    def catat(kolom, aksi):
        hitung[(kolom, aksi)] = hitung.get((kolom, aksi), 0) + 1

    # --- 1. koreksi manual (bukti dicek ulang agar baris yang berubah tidak lolos diam-diam) ---
    for pid, perkolom in KOREKSI_MANUAL.items():
        for kolom, (lama, baru, alasan) in perkolom.items():
            for i in df.index[ids.eq(pid)]:
                sekarang = str(df.at[i, kolom]).strip()
                if sekarang == baru:
                    continue
                if sekarang != lama:
                    raise ValueError(f"{pid}: {kolom} {sekarang!r} berbeda dari bukti koreksi {lama!r}; periksa manual.")
                diterapkan.append(dict(baris_sumber=i, property_id=pid, kolom=kolom, nilai_awal=sekarang,
                                       nilai_deskripsi=baru, nilai_akhir=baru, aksi="manual", bukti=alasan, url=url.at[i]))
                df.at[i, kolom] = baru
                catat(kolom, "manual")

    # --- 2. audit otomatis ---
    for i in df.index:
        d = deskripsi.at[i]
        if len(d.strip()) < 5 or d.strip() == "-":
            continue
        kerja = {}
        lt, lb = ambil_luas(d)
        kerja["luas tanah"] = (lt, kolom_luas, teks_luas)
        kerja["luas bangunan"] = (lb, kolom_luas, teks_luas)
        for kolom, (ekstrak, parser, pembuat) in KONFIG.items():
            kerja[kolom] = (ekstrak(d), parser, pembuat)

        for kolom in KOLOM_AUDIT:
            if kolom not in df.columns:
                continue
            kandidat, parser, pembuat = kerja[kolom]
            if not kandidat:
                continue
            mentah = df.at[i, kolom]
            nilai_kolom = parser(mentah) if pd.notna(mentah) else np.nan
            aksi, baru, semua = putuskan(nilai_kolom, kandidat, TOLERANSI.get(kolom, 0.0), kolom)
            catat(kolom, aksi)
            bukti = kandidat[0][2]
            if aksi in ("koreksi", "isi"):
                rasio = None
                if kolom == "harga" and not np.isnan(nilai_kolom):
                    rasio = round(baru / nilai_kolom, 3)
                diterapkan.append(dict(baris_sumber=i, property_id=ids.at[i], kolom=kolom, nilai_awal=mentah,
                                       nilai_deskripsi=pembuat(baru), nilai_akhir=pembuat(baru), aksi=aksi,
                                       rasio_harga=rasio, bukti=bukti, url=url.at[i]))
                df.at[i, kolom] = pembuat(baru)
            elif aksi == "ambigu":
                periksa.append(dict(baris_sumber=i, property_id=ids.at[i], kolom=kolom, nilai_kolom=mentah,
                                    nilai_di_deskripsi=" | ".join(str(round(v, 3)) if not isinstance(v, str) else v for v in semua),
                                    alasan="deskripsi menyebut >1 nilai berbeda, atau kolom kosong dan isi otomatis dimatikan",
                                    bukti=bukti, url=url.at[i]))

    kolom_log = ["baris_sumber", "property_id", "kolom", "nilai_awal", "nilai_deskripsi", "nilai_akhir",
                 "aksi", "rasio_harga", "bukti", "url"]
    pd.DataFrame(diterapkan, columns=kolom_log).to_csv(out / "koreksi_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(periksa, columns=["baris_sumber", "property_id", "kolom", "nilai_kolom", "nilai_di_deskripsi",
                                   "alasan", "bukti", "url"]).drop_duplicates(
        subset=["property_id", "kolom", "nilai_di_deskripsi"]).to_csv(
        out / "audit_perlu_periksa.csv", index=False, encoding="utf-8-sig")
    ringkas = (pd.Series(hitung).rename_axis(["kolom", "aksi"]).rename("jumlah").reset_index()
               .pivot(index="kolom", columns="aksi", values="jumlah").fillna(0).astype(int))
    ringkas.to_csv(out / "ringkasan_audit.csv", encoding="utf-8-sig")

    n_ubah = sum(v for (k, a), v in hitung.items() if a in ("koreksi", "isi", "manual"))
    print(f"[AUDIT] {n_ubah} nilai diubah dari deskripsi; {len(periksa)} kasus ambigu (periksa manual)")
    print(ringkas.to_string())
    print(f"[AUDIT] laporan: {out}/koreksi_audit.csv, audit_perlu_periksa.csv, ringkasan_audit.csv")
    return df