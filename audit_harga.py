"""Audit indikatif; kandidat dari deskripsi tidak otomatis menjadi koreksi."""
from pathlib import Path
import re
import pandas as pd

KOREKSI = {
    'hos42622932': ('Rp 285 Miliar', 'Rp 2,85 Miliar',
                   'Deskripsi listing menyatakan 2,85M nego; kolom harga 285 Miliar.')
}

def audit_dan_koreksi(df, folder):
    df = df.copy()
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    url = df.get('link detail rumah', pd.Series('', index=df.index)).fillna('')
    if 'link arsip' in df:
        url = url.where(url.str.strip().ne(''), df['link arsip'].fillna(''))
    ids = url.str.extract(r'((?:hos|nps)\d+)', flags=re.I, expand=False).str.lower()
    changes = []
    for pid, (old, new, reason) in KOREKSI.items():
        mask = ids.eq(pid)
        for i in df.index[mask]:
            current = str(df.at[i,'harga']).strip()
            if current == new:
                continue
            if current != old:
                raise ValueError(f'{pid}: harga {current!r} berbeda dari bukti koreksi {old!r}; periksa manual.')
            description = str(df.at[i,'deskripsi']) if 'deskripsi' in df else ''
            if not re.search(r'2[,.]85\s*m\b', description, re.I):
                raise ValueError(f'{pid}: bukti 2,85M tidak ditemukan dalam deskripsi.')
            changes.append({'baris_sumber':i,'property_id':pid,'harga_awal':current,
                            'harga_koreksi':new,'alasan':reason,'url':url.at[i],
                            'deskripsi':description})
            df.at[i,'harga'] = new
    pd.DataFrame(changes, columns=['baris_sumber','property_id','harga_awal','harga_koreksi',
        'alasan','url','deskripsi']).to_csv(out/'koreksi_harga.csv',index=False)
    candidates=[]
    # Semua angka bersatuan harga adalah kandidat audit saja: bisa cicilan/harga per meter.
    pattern=r'(?<![\w])\d+(?:[.,]\d+)*\s*(?:miliar|milyar|juta|jt|m)\b'
    for i,row in df.iterrows():
        desc=str(row.get('deskripsi',''))
        mentions=re.findall(pattern,desc,re.I)
        if mentions:
            candidates.append({'property_id':ids.at[i],'harga_kolom':row.get('harga'),
                'angka_dalam_deskripsi':' | '.join(dict.fromkeys(mentions)),
                'judul':row.get('judul'),'url':url.at[i],'deskripsi':desc,
                'status':'Periksa manual; kandidat bisa harga/m2 atau angka nonharga'})
    pd.DataFrame(candidates).drop_duplicates().to_csv(out/'audit_harga_deskripsi.csv',index=False)
    print(f'[AUDIT] {len(changes)} baris sumber dikoreksi; laporan: {out}')
    return df
