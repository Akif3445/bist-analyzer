"""KAP olay çalışması — hangi bildirim türü gerçekten fiyatı oynatıyor?

`kap_sozluk.json`'daki yön etiketleri HİPOTEZDİ. Bu araç onları ölçüme
çevirir: her bildirim türü için, bildirimden sonraki 1/7/30/90 günde
PİYASAYA GÖRELİ (anormal) getiri ne olmuş?

YÖNTEM ve neden böyle
---------------------
Giriş fiyatı = bildirimden SONRAKİ işlem gününün kapanışı. Aynı günü
kullanmak cazip ama bildirim seans içinde de seans sonrasında da gelebilir
(ölçülen örneklerde 09:28'den 19:21'e kadar), ve hangisinin hangisi olduğunu
ayırmak yerine her zaman ulaşılabilir olan fiyatı almak daha savunulabilir.
Bu muhafazakâr yön: ilk tepkinin bir kısmını kaçırırız, ama look-ahead
tartışması kalmaz.

Anormal getiri = hissenin getirisi eksi AYNI PENCEREDE panelin kesitsel
MEDYAN getirisi. Ham getiriyle bakmak, boğa döneminde yoğunlaşan bir
bildirim türünü sahte güçlü gösterirdi — projede bu hata daha önce yapıldı.

TÜR ÖZETİ DE MEDYANLA ALINIR ve bu tercih değil zorunluluk. İlk sürüm
ortalama kullandı, tablonun TAMAMI pozitif çıktı — salt kunye evrakı olan
"Şirket Genel Bilgi Formu" bile +%2.07, t=+3.0. Sebep ölçüldü; panelin tüm
hisse-gün anormal 30g getirilerinde:

    ortalama %+4.01 | medyan %0.00 | çarpıklık 6.4 | %99 dilim %+89.5

Dağılım öyle sağa çarpık ki RASTGELE bir olay kümesi ortalamayla bakıldığında
+%4 "anormal getiri" gösterir. Tablo sinyal değil, bu yapay tabanı tekrar
ediyordu — üstelik +%2.07 tabanın ALTINDA, yani gerçekte negatif. Medyanla
taban tanım gereği sıfır olur ve türün medyanı doğrudan fazlalığı verir.

VERİ KASASI: fiyat paneli kesim tarihinde budanır, dolayısıyla kesimden
sonraya taşan ileri getiriler kendiliğinden NaN olur ve o satırlar düşer.
Rezerv dönem bu çalışmaya hiç girmez.

Kullanım:
    python tools/kap_olay.py                # tüm türler
    python tools/kap_olay.py --min-ornek 30
"""

import io
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

VERI_DIZIN = "kap_veri"
PANEL = "backtest_panel_8y.pkl"
SOZLUK = "kap_sozluk.json"
UFUKLAR = [1, 7, 30, 90]


def _kasa_kesim():
    try:
        with open("veri_kasasi.json", encoding="utf-8") as f:
            return pd.Timestamp(json.load(f)["kesim"])
    except Exception:
        return None


def _bildirimler() -> pd.DataFrame:
    """Günlük JSON'ları tek tabloya; çok hisseli bildirim satırlara açılır."""
    kayit = []
    for ad in sorted(os.listdir(VERI_DIZIN)):
        if not ad.endswith(".json"):
            continue
        with open(os.path.join(VERI_DIZIN, ad), encoding="utf-8") as f:
            for k in json.load(f):
                kodlar = (k.get("stockCodes") or "").split(",")
                konu = (k.get("subject") or "?").strip()
                tarih = (k.get("publishDate") or "")[:10]
                for kod in kodlar:
                    kod = kod.strip().upper()
                    if kod and tarih:
                        kayit.append((kod, tarih, konu))
    df = pd.DataFrame(kayit, columns=["hisse", "tarih", "konu"])
    df["tarih"] = pd.to_datetime(df["tarih"], format="%d.%m.%Y", errors="coerce")
    return df.dropna(subset=["tarih"])


def calistir(min_ornek: int = 20) -> None:
    if not os.path.isdir(VERI_DIZIN):
        raise SystemExit(f"{VERI_DIZIN}/ yok — önce tools/kap_gecmis.py cek")

    bil = _bildirimler()
    print(f"Bildirim: {len(bil)} satır (hisse×olay) | "
          f"{bil['tarih'].min():%Y-%m-%d} .. {bil['tarih'].max():%Y-%m-%d} | "
          f"{bil['konu'].nunique()} tür")

    with open(PANEL, "rb") as f:
        closes = pickle.load(f)["closes"]
    fiyat = pd.DataFrame(closes).sort_index()

    kes = _kasa_kesim()
    if kes is not None:
        fiyat = fiyat[fiyat.index <= kes]
        print(f"[KASA] fiyat paneli {kes:%Y-%m-%d} tarihinde budandı — kesimden "
              f"sonraya taşan ileri getiriler NaN olur ve düşer")
    print(f"Panel: {fiyat.shape[1]} hisse | {fiyat.index[0]:%Y-%m-%d} .. "
          f"{fiyat.index[-1]:%Y-%m-%d}")

    # Giriş günü: bildirim tarihinden SONRAKİ işlem günü
    idx = fiyat.index
    poz = idx.searchsorted(bil["tarih"].values, side="right")
    gecerli = (poz < len(idx))
    bil = bil[gecerli].copy()
    bil["giris_i"] = poz[gecerli]

    # Ufuk başına anormal getiri matrisi (giriş gününe hizalı)
    anormal = {}
    for n in UFUKLAR:
        g = fiyat.shift(-n) / fiyat - 1.0
        anormal[n] = g.sub(g.median(axis=1), axis=0) * 100.0

    # Hisse -> kolon indeksi
    kol = {t: i for i, t in enumerate(fiyat.columns)}
    bil = bil[bil["hisse"].isin(kol)].copy()
    print(f"Panelde eşleşen: {len(bil)} satır ({bil['hisse'].nunique()} hisse)\n")

    deger = {}
    for n in UFUKLAR:
        A = anormal[n].to_numpy()
        s = np.full(len(bil), np.nan)
        ri = bil["giris_i"].to_numpy()
        ci = bil["hisse"].map(kol).to_numpy()
        ok = (ri >= 0) & (ri < A.shape[0])
        s[ok] = A[ri[ok], ci[ok]]
        deger[n] = s
        bil[f"a{n}"] = s

    sozluk = {}
    if os.path.exists(SOZLUK):
        sozluk = json.load(io.open(SOZLUK, encoding="utf-8"))["turler"]

    # Tür bazında özet
    #
    # İKİ AYRI t HESAPLANIR ve aradaki fark önemlidir:
    #   t_ham  : tüm olaylar bağımsızmış gibi. ŞİŞKİN — aynı hissenin bir
    #            haftadaki 5 bildiriminin 30 günlük pencereleri neredeyse
    #            aynı getiriyi ölçer, ayrıca aynı gün tüm piyasa birlikte
    #            hareket eder. İkisi de bağımsızlık varsayımını bozar.
    #   t_gun  : önce GÜN içinde ortalama alınır, sonra günler arasında t.
    #            Aynı günün kesitsel korelasyonunu söndürür (Fama-MacBeth
    #            mantığı). Kalan kusur: 30 günlük pencerelerin günler arası
    #            çakışması. Yani t_gun de bir ÜST sınırdır, ama t_ham'dan
    #            çok daha dürüsttür. Karar verirken t_gun'e bakılmalı.
    satir = []
    for konu, g in bil.groupby("konu"):
        kayit = {"konu": konu, "n": len(g)}
        for n in UFUKLAR:
            v = g[f"a{n}"].dropna()
            kayit[f"n{n}"] = len(v)
            if len(v) < min_ornek:
                kayit[f"med{n}"] = np.nan
                kayit[f"poz{n}"] = np.nan
                kayit[f"tg{n}"] = np.nan
                continue
            kayit[f"med{n}"] = v.median()
            kayit[f"poz{n}"] = (v > 0).mean() * 100.0
            # Gün içinde MEDYAN (kalın kuyruğa dayanıklı), sonra günler arası t
            # — aynı günün kesitsel korelasyonunu söndürür. Kalan kusur: 30
            # günlük pencerelerin günler arası çakışması, yani bu da ÜST sınır.
            gunluk = g.dropna(subset=[f"a{n}"]).groupby("tarih")[f"a{n}"].median()
            if len(gunluk) >= 10 and gunluk.std() > 0:
                kayit[f"tg{n}"] = gunluk.median() / (gunluk.std() / np.sqrt(len(gunluk)))
            else:
                kayit[f"tg{n}"] = np.nan
        sz = sozluk.get(konu, {})
        kayit["sozluk"] = {1: "+", -1: "-", 0: "0"}.get(sz.get("yon"), "?")
        kayit["sinif"] = sz.get("sinif", "-")
        satir.append(kayit)

    t = pd.DataFrame(satir).sort_values("n", ascending=False)
    t.to_csv("kap_olay_sonuc.csv", index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("TÜR BAZINDA MEDYAN ANORMAL GETİRİ (piyasa medyanına göre, %)")
    print("=" * 100)
    print(f"{'tür':<44}{'n':>6}{'7g':>8}{'30g':>8}{'poz%':>7}{'t_gün':>7}  söz")
    print("-" * 100)
    for _, r in t.iterrows():
        if r["n"] < min_ornek:
            continue

        def f(k, b="{:+.2f}"):
            return b.format(r[k]) if pd.notna(r[k]) else "   -  "
        yildiz = "*" if pd.notna(r["tg30"]) and abs(r["tg30"]) > 2.5 else " "
        print(f"{r['konu'][:44]:<44}{int(r['n']):>6}{f('med7'):>8}{f('med30'):>8}"
              f"{f('poz30','{:.0f}'):>7}{f('tg30','{:+.1f}'):>7}{yildiz}  {r['sozluk']}")
    print("-" * 100)
    print("MEDYAN anormal getiri — ORTALAMA DEĞİL (bkz. dosya başı: ortalama,")
    print("kalın kuyruk yüzünden her türü +%4 gösteriyordu). Taban gereği 0.")
    print("poz% = olayların yüzde kaçı pozitif bitti (taban 50).")
    print("t_gün = gün içi medyan, sonra günler arası t. * = |t|>2.5")
    print("söz = kap_sozluk.json'daki HİPOTEZ, ölçüm değil.")
    print("\nUYARI - TERS NEDENSELLİK: bazı türler fiyat hareketinin SONUCUDUR.")
    print("'Pay Bazında Devre Kesici' ve 'Olağan Dışı Fiyat ve Miktar")
    print("Hareketleri' hisse çok oynadığı için tetiklenir; yüksek çıkmaları")
    print("sinyal değil tanımın kendisidir. Sinyal olarak kullanılamaz.")
    print("\nkap_olay_sonuc.csv yazıldı.")


if __name__ == "__main__":
    mo = 20
    if "--min-ornek" in sys.argv:
        mo = int(sys.argv[sys.argv.index("--min-ornek") + 1])
    calistir(mo)
