"""Haber sınıflandırıcısını FİYATA karşı test eder — her kelime öbeği ölçülür.

Soru (kullanıcı, 2026-09-27): filtre olumlu/olumsuzu doğru veriyor mu? Doğruyu
ne belirler: öbeğin ardından fiyat. Kâr getiren öbek olumlu, zarar getiren
olumsuz yazılmalı.

VERİ: tarihli ve fiyatla eşleşen tek büyük Türkçe metin arşivimiz KAP
bildirim ÖZETLERİ (kap_veri/, 2018-01 → 2024-08, ~370 bin özet). Google News
arşiv vermediği için gerçek haber manşeti geçmişi yok; KAP özetleri şirket
dilinde yazılmış kısa metinlerdir. Tür farkı SONUÇ OKUNURKEN akılda tutulmalı.

YÖNTEM (tools/kap_olay.py ile aynı fizik):
  - giriş = bildirimden sonraki işlem günü kapanışı; 7 ve 30 günlük getiri
  - anormal = hissenin getirisi − aynı gün panelin kesitsel MEDYANI
  - aynı hisse + aynı özet 30 gün içinde tekrar ediyorsa TEK olay sayılır
    (günlük geri alım bildirimleri pencereleri çakıştırıp t'yi şişirirdi)
  - t: gün içi medyan, sonra günler arası t (aynı günün korelasyonu söner)
  - veri kasası: panel kesimde budanır, kasaya taşan pencereler düşer

KARAR KURALI (bir öbek için, koşmadan önce sabitlendi):
  ölçülebilir  : n >= 100 olay ve >= 30 ayrı gün
  anlamlı      : |t| >= 3.0 VE iki yarının (2018-01→2021-04, 2021-05→2024-08)
                 medyanları tam dönemle AYNI işarette
  DOGRULANDI   : anlamlı ve yön varsayımla aynı
  TERS         : anlamlı ve yön varsayımın tersi  → yön çevrilir
  GURULTU      : ölçülebilir ama anlamlı değil     → öbek nötre çekilir
  OLCULEMEDI   : yeterli olay yok                  → dokunulmaz
~600 öbek test edildiği için eşik |t|>=2 değil 3 ve iki yarı tutarlılığı
şart — yoksa şans eseri "anlamlı" çıkanlar sisteme yazılırdı.

Çıktı: haber_olcum.json (news_engine bunu okur) + konsol raporu.

Kullanım:  python tools/haber_olcum.py
"""

import json
import os
import pickle
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging  # noqa: E402
logging.disable(logging.CRITICAL)

VERI_DIZIN = "kap_veri"
PANEL = "backtest_panel_8y.pkl"
CIKTI = "haber_olcum.json"
YARI = pd.Timestamp("2021-05-01")
MIN_N, MIN_GUN, MIN_T = 100, 30, 3.0

# Yönü BİLDİRİM TÜRÜ düzeyinde daha önce ölçülmüş öbekler (tools/kap_olay.py,
# kap_sozluk.json). Özet metninde eşiğin biraz altında kalsalar da (geri alım
# t=+2.3, sermaye artırımı t=-2.7; ikisi de iki yarıda aynı işaret) susturulmaz:
# tür düzeyindeki ölçüm daha güçlü ve bağımsız bir okuma.
KORUNAN = {
    "pay geri alımı": "Payların Geri Alınması türü: 30g %+2.63, t=+4.6, n=5158",
    "sermaye artırımı": "Sermaye Artırımı türü: 30g %-2.01, t=-3.1, n=3480",
    "sermaye artirimi": "Sermaye Artırımı türü: 30g %-2.01, t=-3.1, n=3480",
}


def _kasa_kesim():
    try:
        with open("veri_kasasi.json", encoding="utf-8") as f:
            return pd.Timestamp(json.load(f)["kesim"])
    except Exception:
        return None


def _olaylar() -> pd.DataFrame:
    kayit = []
    for ad in sorted(os.listdir(VERI_DIZIN)):
        if not ad.endswith(".json"):
            continue
        with open(os.path.join(VERI_DIZIN, ad), encoding="utf-8") as f:
            for k in json.load(f):
                ozet = (k.get("summary") or "").strip()
                tarih = (k.get("publishDate") or "")[:10]
                if not ozet or not tarih:
                    continue
                for kod in (k.get("stockCodes") or "").split(","):
                    kod = kod.strip().upper()
                    if kod:
                        kayit.append((kod, tarih, ozet))
    df = pd.DataFrame(kayit, columns=["hisse", "tarih", "ozet"])
    df["tarih"] = pd.to_datetime(df["tarih"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["tarih"]).sort_values(["hisse", "ozet", "tarih"])
    # Aynı hisse + aynı özet 30 gün içinde → tek olay
    onceki = df.groupby(["hisse", "ozet"])["tarih"].shift()
    df = df[onceki.isna() | ((df["tarih"] - onceki).dt.days > 30)]
    return df.reset_index(drop=True)


def _getiriler(df: pd.DataFrame) -> pd.DataFrame:
    with open(PANEL, "rb") as f:
        fiyat = pd.DataFrame(pickle.load(f)["closes"]).sort_index()
    kes = _kasa_kesim()
    if kes is not None:
        fiyat = fiyat[fiyat.index <= kes]
    kol = {t: i for i, t in enumerate(fiyat.columns)}
    df = df[df["hisse"].isin(kol)].copy()
    poz = fiyat.index.searchsorted(df["tarih"].values, side="right")
    df = df[poz < len(fiyat.index)].copy()
    df["ri"] = poz[poz < len(fiyat.index)]
    ci = df["hisse"].map(kol).to_numpy()
    for n in (7, 30):
        g = fiyat.shift(-n) / fiyat - 1.0
        A = (g.sub(g.median(axis=1), axis=0) * 100.0).to_numpy()
        df[f"a{n}"] = A[df["ri"].to_numpy(), ci]
    return df.dropna(subset=["a30"]).reset_index(drop=True)


def _istatistik(g: pd.DataFrame) -> dict:
    v = g["a30"]
    gun = g.groupby("tarih")["a30"].median()
    t = (gun.median() / (gun.std() / np.sqrt(len(gun)))
         if len(gun) >= 10 and gun.std() > 0 else np.nan)
    y1, y2 = g[g["tarih"] < YARI]["a30"], g[g["tarih"] >= YARI]["a30"]
    return {"n": int(len(v)), "gun": int(len(gun)),
            "med30": round(float(v.median()), 2) if len(v) else None,
            "med7": round(float(g["a7"].median()), 2) if len(v) else None,
            "poz": round(float((v > 0).mean() * 100), 1) if len(v) else None,
            "t": round(float(t), 2) if pd.notna(t) else None,
            "yari1": round(float(y1.median()), 2) if len(y1) >= 20 else None,
            "yari2": round(float(y2.median()), 2) if len(y2) >= 20 else None}


def _karar(s: dict, yon: int) -> str:
    if s["n"] < MIN_N or s["gun"] < MIN_GUN or s["t"] is None:
        return "OLCULEMEDI"
    isaret = np.sign(s["med30"])
    tutarli = (s["yari1"] is not None and s["yari2"] is not None
               and np.sign(s["yari1"]) == isaret and np.sign(s["yari2"]) == isaret)
    if abs(s["t"]) >= MIN_T and tutarli and isaret != 0:
        return "DOGRULANDI" if isaret == yon else "TERS"
    return "GURULTU"


def _katalog():
    """Sınıflandırıcının kullandığı HER öbek: (liste, anahtar, yön, ağırlık, kalıp)."""
    import news_engine as ne
    k = []
    for w, a in ne.FIN_POSITIVE_WEIGHTED:
        k.append(("FIN_POZ", w, +1, a, ne._kelime_kalibi(w)))
    for w, a in ne.FIN_NEGATIVE_WEIGHTED:
        k.append(("FIN_NEG", w, -1, a, ne._kelime_kalibi(w)))
    for w in ne.POSITIVE_WORDS:
        k.append(("KELIME_POZ", w, +1, 1, ne._kelime_kalibi(w)))
    for w in ne.NEGATIVE_WORDS:
        k.append(("KELIME_NEG", w, -1, 1, ne._kelime_kalibi(w)))
    for p in ne._STRONG_POS_PATTERNS:
        k.append(("GUCLU_POZ", p.pattern, +1, 3, p))
    for p in ne._STRONG_NEG_PATTERNS:
        k.append(("GUCLU_NEG", p.pattern, -1, 3, p))
    for p in ne._NEGATION_FLIP_TO_POSITIVE:
        k.append(("BAGLAM_POZ", p.pattern, +1, 3, p))
    for p in ne._NEGATION_FLIP_TO_NEGATIVE:
        k.append(("BAGLAM_NEG", p.pattern, -1, 3, p))
    return k


def calistir():
    t0 = time.time()
    import news_engine as ne
    df = _getiriler(_olaylar())
    metinler = df["ozet"].unique()
    print(f"Olay: {len(df)} (tekrarlar tek olaya indirildi) | tekil özet: {len(metinler)} | "
          f"{df['tarih'].min():%Y-%m-%d} → {df['tarih'].max():%Y-%m-%d} | {time.time()-t0:.0f}s")

    # 1) ÜRETİMDEKİ sınıflandırıcının genel isabeti. Ölçüm dosyası YÜKLENMEDEN
    #    koşar (ne.OLCUM_KAPALI) — yani bugünkü etiketler ölçülür, düzeltilmiş
    #    olanlar değil.
    ne.OLCUM_KAPALI = True
    ne._olcum_sifirla()
    etiket = {m: ne._classify(m) for m in metinler}
    df["etiket"] = df["ozet"].map(etiket)
    print(f"Sınıflandırma bitti ({time.time()-t0:.0f}s)\n")
    print("=" * 92)
    print("1) FİLTRENİN GENEL İSABETİ — etiket başına 30 gün sonraki piyasaya göre getiri")
    print("=" * 92)
    genel = {}
    for e in ("olumlu", "olumsuz", "notr"):
        s = _istatistik(df[df["etiket"] == e])
        genel[e] = s
        print(f"  {e:<8} n={s['n']:>7}  7g medyan %{s['med7']:+.2f}  30g medyan %{s['med30']:+.2f}  "
              f"poz %{s['poz']:.0f}  t={s['t']:+.1f}  (yarılar %{s['yari1']:+.2f} / %{s['yari2']:+.2f})")

    # 2) ÖBEK ÖBEK
    kat = _katalog()
    kucuk = {m: m.lower() for m in metinler}
    sonuc = []
    for i, (liste, anahtar, yon, agirlik, kalip) in enumerate(kat):
        # Önfiltre: kelime bankasında kökün ilk 3 harfi metinde yoksa regex'e gerek yok
        on = anahtar.lower()[:3] if liste.startswith(("FIN", "KELIME")) else None
        eslesen = [m for m in metinler
                   if (on is None or on in kucuk[m]) and kalip.search(m)]
        s = _istatistik(df[df["ozet"].isin(set(eslesen))]) if eslesen else \
            {"n": 0, "gun": 0, "med30": None, "med7": None, "poz": None,
             "t": None, "yari1": None, "yari2": None}
        s.update({"liste": liste, "anahtar": anahtar, "yon": yon, "agirlik": agirlik})
        s["karar"] = _karar(s, yon)
        if s["karar"] == "GURULTU" and anahtar in KORUNAN:
            s["karar"] = "KORUNDU"
            s["not"] = KORUNAN[anahtar]
        sonuc.append(s)
    print(f"\n{len(kat)} öbek ölçüldü ({time.time()-t0:.0f}s)")

    say = pd.Series([s["karar"] for s in sonuc]).value_counts().to_dict()
    print("\n" + "=" * 92)
    print(f"2) ÖBEK KARARLARI: {say}")
    print("=" * 92)
    for k in ("DOGRULANDI", "TERS", "KORUNDU", "GURULTU"):
        grup = sorted([s for s in sonuc if s["karar"] == k], key=lambda s: -s["n"])
        print(f"\n-- {k} ({len(grup)})")
        for s in grup[:40]:
            print(f"  {s['liste']:<11} {str(s['anahtar'])[:42]:<42} yön{s['yon']:+d} "
                  f"n={s['n']:>6} 30g %{s['med30']:+.2f} t={s['t']:+.1f} "
                  f"yarılar %{s['yari1'] if s['yari1'] is not None else float('nan'):+.2f}/"
                  f"%{s['yari2'] if s['yari2'] is not None else float('nan'):+.2f}")

    json.dump({"tarih": pd.Timestamp.now().strftime("%Y-%m-%d"),
               "yontem": "KAP özetleri 2018-01→2024-08, 30g piyasaya göre medyan getiri, "
                         f"n>={MIN_N}, |t|>={MIN_T}, iki yarı aynı işaret",
               "genel": genel, "obekler": sonuc},
              open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

    # 3) Kararlar UYGULANMIŞ sınıflandırıcıyla aynı test. Aynı veride
    #    seçildiği için örneklem-içidir; yalnız "etkisiz öbekleri susturmak
    #    bir şeyi bozdu mu" sorusuna cevap verir, yeni bir kanıt değildir.
    ne.OLCUM_KAPALI = False
    ne._olcum_sifirla()
    df["etiket2"] = df["ozet"].map({m: ne._classify(m) for m in metinler})
    print("\n" + "=" * 92)
    print("3) KARARLAR UYGULANDIKTAN SONRA — aynı test (örneklem-içi)")
    print("=" * 92)
    for e in ("olumlu", "olumsuz", "notr"):
        s = _istatistik(df[df["etiket2"] == e])
        print(f"  {e:<8} n={s['n']:>7}  7g medyan %{s['med7']:+.2f}  30g medyan %{s['med30']:+.2f}  "
              f"poz %{s['poz']:.0f}  t={s['t']:+.1f}  (yarılar %{s['yari1']:+.2f} / %{s['yari2']:+.2f})")
    print(f"\n{CIKTI} yazıldı | toplam {time.time()-t0:.0f}s")


if __name__ == "__main__":
    calistir()
