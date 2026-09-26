"""Bir bildirim TÜRÜNÜ içeriğine göre ALT GRUPLARA ayırıp ölçer.

NEDEN
-----
`kap_olay.py` tür bazında ölçüyor. Ama bazı türler kendi içinde ZIT şeyler
barındırır ve ortalaması ikisini birbirine karıştırır. En net örnek sermaye
artırımı: BEDELLİ artırım ortakların payını sulandırır (teoriye göre olumsuz),
BEDELSİZ ise muhasebesel bir işlemdir (nötr). `kap_olay.py` bu türü -%2.01
(t=-3.1) ölçüyor; soru şu: bu tamamen bedellilerden mi geliyor?

Ayrım ÖZET alanından yapılır — 411 bin bildirimin %90'ında dolu ve indirilmiş
durumda, yani ek istek gerekmez. Özetin yön taşımadığı türlerde (kredi
derecelendirmesi: "yükseltildi" 2 kez, "düşürüldü" 0 kez geçiyor) bu yol
işlemez, gövde metni gerekir.

YÖNTEM kap_olay.py ile aynı, bilerek: giriş bildirimden sonraki kapanış,
30 günlük MEDYAN piyasaya göreli getiri, gün-kümelenmiş t, kasa kesimi.

Kullanım:
    python tools/kap_ayrim.py sermaye
"""

import glob
import io
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

VERI_DIZIN = "kap_veri"
PANEL = "backtest_panel_8y.pkl"
UFUKLAR = [7, 30]

# tur -> [(grup adi, ozet icinde aranan kelimeler), ...]
# Sira onemli: ilk eslesen grup kazanir.
AYRIMLAR = {
    "sermaye": {
        "tur": "Sermaye Artırımı - Azaltımı İşlemlerine İlişkin Bildirim",
        "gruplar": [
            ("BEDELLI (sulandirici)", ["bedelli", "Bedelli", "BEDELLİ", "bedellİ"]),
            ("BEDELSIZ (notr)", ["bedelsiz", "Bedelsiz", "BEDELSİZ"]),
            ("AZALTIM", ["azalt", "Azalt", "AZALT"]),
        ],
        "hipotez": "BEDELLI < BEDELSIZ olmali (sulandirma)",
        "beklenen_isaret": -1,
    },
    "ihale": {
        "tur": "İhale Süreci / Sonucu",
        "gruplar": [
            ("SONUC", ["Sonuc", "Sonuç", "sonuc", "sonuç", "kazan", "Kazan", "alınma", "Alınma"]),
            ("SUREC", ["Süreci", "süreci", "Surec", "sürec"]),
        ],
        "hipotez": "SONUC > SUREC olmali (sonuc kesinlesmis bilgi)",
        "beklenen_isaret": +1,
    },
}


def _kesim():
    try:
        return pd.Timestamp(json.load(io.open("veri_kasasi.json",
                                              encoding="utf-8"))["kesim"])
    except Exception:
        return None


def _olaylar(tur):
    kayit = []
    for y in glob.glob(os.path.join(VERI_DIZIN, "*.json")):
        for k in json.load(io.open(y, encoding="utf-8")):
            if (k.get("subject") or "").strip() != tur:
                continue
            ozet = (k.get("summary") or "")
            tarih = (k.get("publishDate") or "")[:10]
            for kod in (k.get("stockCodes") or "").split(","):
                kod = kod.strip().upper()
                if kod and tarih:
                    kayit.append((kod, tarih, ozet))
    df = pd.DataFrame(kayit, columns=["hisse", "tarih", "ozet"])
    df["tarih"] = pd.to_datetime(df["tarih"], format="%d.%m.%Y", errors="coerce")
    return df.dropna(subset=["tarih"])


def calistir(ad):
    cfg = AYRIMLAR[ad]
    bil = _olaylar(cfg["tur"])
    print(f"{cfg['tur']}\n{len(bil)} olay | hipotez: {cfg['hipotez']}\n")

    def grupla(o):
        for g, kelimeler in cfg["gruplar"]:
            if any(w in o for w in kelimeler):
                return g
        return "DIGER"
    bil["grup"] = bil["ozet"].map(grupla)

    with open(PANEL, "rb") as f:
        closes = pickle.load(f)["closes"]
    fiyat = pd.DataFrame(closes).sort_index()
    kes = _kesim()
    if kes is not None:
        fiyat = fiyat[fiyat.index <= kes]
        print(f"[KASA] panel {kes:%Y-%m-%d} tarihinde budandı")

    idx = fiyat.index
    poz = idx.searchsorted(bil["tarih"].values, side="right")
    ok = poz < len(idx)
    bil = bil[ok].copy()
    bil["gi"] = poz[ok]
    kol = {t: i for i, t in enumerate(fiyat.columns)}
    bil = bil[bil["hisse"].isin(kol)].copy()
    print(f"Panelde eşleşen: {len(bil)} olay\n")

    for n in UFUKLAR:
        g = fiyat.shift(-n) / fiyat - 1.0
        A = (g.sub(g.median(axis=1), axis=0) * 100).to_numpy()
        v = np.full(len(bil), np.nan)
        ri = bil["gi"].to_numpy()
        ci = bil["hisse"].map(kol).to_numpy()
        m = (ri >= 0) & (ri < A.shape[0])
        v[m] = A[ri[m], ci[m]]
        bil[f"a{n}"] = v

    print(f"{'grup':<26}{'n':>7}{'7g':>8}{'30g':>8}{'poz%':>7}{'t_gün':>8}")
    print("-" * 66)
    sonuc = {}
    for grup, gg in bil.groupby("grup"):
        s = [f"{grup[:26]:<26}{len(gg):>7}"]
        for n in UFUKLAR:
            x = gg[f"a{n}"].dropna()
            s.append(f"{x.median():>+8.2f}" if len(x) >= 20 else f"{'-':>8}")
        x30 = gg["a30"].dropna()
        if len(x30) >= 20:
            gun = gg.dropna(subset=["a30"]).groupby("tarih")["a30"].median()
            t = (gun.median() / (gun.std() / np.sqrt(len(gun)))
                 if len(gun) >= 10 and gun.std() > 0 else float("nan"))
            s.append(f"{(x30 > 0).mean()*100:>7.0f}")
            s.append(f"{t:>+8.1f}" if pd.notna(t) else f"{'-':>8}")
            sonuc[grup] = (x30.median(), t, len(x30))
        else:
            s += [f"{'-':>7}", f"{'-':>8}"]
        print("".join(s))
    print("-" * 66)
    print("30g MEDYAN piyasaya göreli getiri. t_gün = gün içi medyan sonrası")
    print("günler arası t. Gruplar aynı türün alt kümeleri, ayrım ÖZET metninden.")

    g = list(cfg["gruplar"])
    if len(g) >= 2 and g[0][0] in sonuc and g[1][0] in sonuc:
        a, b = sonuc[g[0][0]], sonuc[g[1][0]]
        fark = a[0] - b[0]
        # Beklenen isaret YAPILANDIRMADAN gelir. Once sabit "fark<0" yazilmisti
        # (sermaye durumuna gore) ve ihale ayriminda hipotezin TERSINI
        # "UYUYOR" diye onayliyordu - arac yanlis dogrulama basiyordu.
        bekl = cfg.get("beklenen_isaret", -1)
        uyar = (fark < 0) if bekl < 0 else (fark > 0)
        print("")
        print(f"HIPOTEZ TESTI: {g[0][0]} eksi {g[1][0]} = {fark:+.2f} puan")
        print(f"  beklenen: {cfg['hipotez']} (isaret {bekl:+d})")
        anlamli = ((pd.notna(a[1]) and abs(a[1]) > 2.5)
                   or (pd.notna(b[1]) and abs(b[1]) > 2.5))
        if not anlamli:
            print("  KARARSIZ - hicbir grup anlamli degil (|t|<2.5), fark "
                  "yorumlanamaz")
        else:
            print(f"  {'UYUYOR' if uyar else 'UYMUYOR'}")


if __name__ == "__main__":
    ad = sys.argv[1] if len(sys.argv) > 1 else ""
    if ad not in AYRIMLAR:
        raise SystemExit(f"kullanım: kap_ayrim.py [{' | '.join(AYRIMLAR)}]")
    calistir(ad)
