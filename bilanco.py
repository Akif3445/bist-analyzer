"""
bilanco — BIST şirketlerinin çeyreklik mali tabloları ve oran altyapısı.

Bu dosya ALTYAPIDIR: veriyi çeker, dondurur, oranları hesaplar ve ders
notlarındaki eşiklere göre değerlendirir. Skora ya da portföy kurulumuna
HİÇBİR şey bağlamaz — o adım ön kayıtlı bir testten sonra, onayla gelir.

ENFLASYON: tüm rakamlar resmi TÜFE ile sabit paraya çevrilir (TMS 29
geçişi yüzünden; bkz. _para_birimi ve _ttm_sabit). Bu yüzden büyüme oranları REELdir.

VERİ: borsapy → İş Yatırım. Her hisse için gelir tablosu, bilanço ve nakit
akışı, çeyreklik, ~40 çeyrek geriye (2016Q3'e kadar). Ham hâli
bilanco_veri/<HİSSE>.pkl'de DONDURULUR (gitignore; fiyat paneliyle aynı ilke:
bir kez indir, sonuçlar tekrarlanabilsin).

ÜÇ TUZAK, hepsi ölçülerek bulundu (2026-10-06):

1. Gelir tablosu ve nakit akışı YIL BAŞINDAN BİRİKİMLİDİR. BİM 2026Q2 satış
   450 mlr TL, Q1 213 mlr TL — Q2 rakamı altı aylık toplam. Çeyreklik akış =
   o çeyreğin birikimlisi − bir önceki çeyreğinki (Q1 hariç). Bunu atlamak
   her akış oranını yanlış yapardı. Bilanço kalemleri ise dönem sonu
   stoktur, dönüşüm gerekmez.
2. Bilançoda AYNI ADLI satırlar var ("Stoklar" dönen + duran, "Ticari
   Borçlar" kısa + uzun vade) ve borsapy 4 çeyreği aşan isteklerde parçaları
   satır ADINA göre birleştirirken bunları çaprazlıyor: 40 çeyrekte bilanço 70
   yerine 13.866 satır, "Stoklar" 1.024 kez. O yüzden borsapy'nin üst
   fonksiyonu KULLANILMAZ; İş Yatırım'ın MaliTablo uç noktası 4'er dönemlik
   parçalarla doğrudan çağrılır ve satırlar API'nin BENZERSİZ kalem koduyla
   (itemCode) anahtarlanır: 1AF dönen stok, 1BFA duran stok, 2AAGAA kısa vade
   ticari borç... Kod yıllar arası şablon farkından da etkilenmez. Tek çağrı
   üç tabloyu birden döndürür (kod öneki 1-2 bilanço, 3 gelir, 4 nakit).
3. Bilanço AÇIKLANDIĞI gün bilinir, çeyrek bittiği gün değil (1-2.5 ay
   sonra). Açıklanma tarihi KAP arşivinden (kap_veri/, "Finansal Rapor",
   year/period alanları) alınır; arşivin bittiği yerden sonrası için
   muhafazakâr yasal-süre tahmini kullanılır ve "tahmini" diye işaretlenir.

BANKALAR kapsam dışı: İş Yatırım kaynağında GARAN için iki biçimde de veri
yok (XI_29 ve UFRS), ve ders notundaki oranlar (stok, ticari borç, faiz
karşılama) bankaya zaten uymaz — faiz bankanın asıl işidir.

Kullanım:
    python bilanco.py cek              # evreni çek (kaldığı yerden devam eder)
    python bilanco.py ozet             # bilanco_veri/ içeriği
    python bilanco.py oran THYAO       # bir hissenin oran geçmişi + değerlendirme
    python bilanco.py kesit 2024Q2     # o çeyrekte tüm evren, sektöre göre karşılaştırmalı
"""

import json
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

KOK = os.path.dirname(os.path.abspath(__file__))
VERI_DIZIN = os.path.join(KOK, "bilanco_veri")
ESIK_DOSYA = os.path.join(KOK, "bilanco_esikler.json")
EVREN_DOSYA = os.path.join(KOK, "backtest_evren.json")
KAP_DIZIN = os.path.join(KOK, "kap_veri")

ISCI = 3            # İş Yatırım yavaşlayabiliyor; nazik çekim
CAGRI_SURE = 60     # tek çağrı en fazla bu kadar beklenir
CAGRI_ARA = 0.5     # aynı işçinin çağrıları arası bekleme (sn)

# Muhafazakâr açıklanma tahmini (KAP arşivinin bittiği yerden sonrası için):
# yasal süreler konsolide ara dönemde ~60-70 gün, yıllıkta ~70-75 gün; gecikme
# payıyla yukarı yuvarlandı. Erken tahmin ileriye bakma hatası olurdu,
# geç tahmin yalnızca veriyi biraz geç kullanmak demek — güvenli yön bu.
TAHMINI_GECIKME_GUN = {1: 75, 2: 75, 3: 75, 4: 100}


# ---------------------------------------------------------------- ÇEKİM

def _cagri(fn, *a, **kw):
    """Tek bir ağ çağrısı, süre sınırlı; askıda kalırsa bir kez daha denenir."""
    son = None
    for _ in range(2):
        with ThreadPoolExecutor(1) as ex:
            try:
                return ex.submit(fn, *a, **kw).result(timeout=CAGRI_SURE)
            except Exception as exc:       # zaman aşımı ya da ağ hatası
                son = exc
        time.sleep(3)
    raise son


def _hisse_cek(ticker: str) -> dict:
    """Bir hissenin ~40 çeyreklik tüm tablolarını kalem koduyla anahtarlı çeker.

    Dönüş tablosu: indeks = itemCode, 'ad' kolonu + '2026Q2' biçimli dönemler.
    Hiç veri yoksa (bankalar) DataNotAvailable benzeri bir hata yükseltir.
    """
    import borsapy
    prov = borsapy.Ticker(ticker)._get_isyatirim()
    donemler = prov._get_periods(datetime.now().year, True, count=40)
    url = f"{prov.BASE_URL}/Data.aspx/MaliTablo"
    boy = prov._MAX_PERIODS_PER_CALL
    satirlar = {}
    for i in range(0, len(donemler), boy):
        parca = donemler[i:i + boy]
        params = {"companyCode": ticker, "exchange": "TRY", "financialGroup": "XI_29"}
        for j, (y, p) in enumerate(parca, 1):
            params[f"year{j}"], params[f"period{j}"] = y, p
        veri = _cagri(lambda: prov._get(url, params=params).json())
        for x in (veri or {}).get("value") or []:
            kod = str(x.get("itemCode") or "")
            if not kod:
                continue
            r = satirlar.setdefault(kod, {"ad": str(x.get("itemDescTr") or "").strip()})
            for j, (y, p) in enumerate(parca, 1):
                v = x.get(f"value{j}")
                if v is not None:
                    try:
                        r[f"{y}Q{p // 3}"] = float(v)
                    except (TypeError, ValueError):
                        pass
        time.sleep(CAGRI_ARA)
    tablo = pd.DataFrame.from_dict(satirlar, orient="index")
    donem_kol = [c for c in tablo.columns if c != "ad"]
    if tablo.empty or tablo[donem_kol].notna().sum().sum() == 0:
        raise RuntimeError(f"{ticker}: İş Yatırım'da mali tablo yok (banka/finans olabilir)")
    tablo = tablo[["ad"] + _ceyrek_sirali(donem_kol)]
    return {"ticker": ticker, "cekildi": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "kaynak": "isyatirim MaliTablo XI_29", "tablo": tablo}


def _evren() -> list:
    with open(EVREN_DOSYA, encoding="utf-8") as f:
        return json.load(f)["tickers"]


def cek(yenile: bool = False) -> int:
    """Evrendeki her hisseyi çeker, diske dondurur. Var olanı atlar (devam eder)."""
    os.makedirs(VERI_DIZIN, exist_ok=True)
    evren = _evren()
    hata_dosya = os.path.join(VERI_DIZIN, "_hatalar.json")
    hatalar = {}
    if os.path.exists(hata_dosya):
        hatalar = json.load(open(hata_dosya, encoding="utf-8"))
    yapilacak = [t for t in evren
                 if yenile or not os.path.exists(os.path.join(VERI_DIZIN, f"{t}.pkl"))]
    print(f"Evren {len(evren)} | çekilecek {len(yapilacak)} | {ISCI} işçi")
    t0, ok = time.time(), 0
    with ThreadPoolExecutor(ISCI) as ex:
        isler = {ex.submit(_hisse_cek, t): t for t in yapilacak}
        for i, fut in enumerate(as_completed(isler), 1):
            t = isler[fut]
            try:
                d = fut.result()
                # Önce geçici dosyaya, sonra yerine: yazarken kesilirse yarım
                # bir .pkl kalmasın (bir sonraki koşu onu "var" sayıp atlardı).
                hedef = os.path.join(VERI_DIZIN, f"{t}.pkl")
                with open(hedef + ".tmp", "wb") as f:
                    pickle.dump(d, f)
                os.replace(hedef + ".tmp", hedef)
                hatalar.pop(t, None)
                ok += 1
            except Exception as exc:
                hatalar[t] = f"{type(exc).__name__}: {str(exc)[:80]}"
            if i % 25 == 0 or i == len(yapilacak):
                print(f"  {i}/{len(yapilacak)} | başarılı {ok} | hata {len(hatalar)} | "
                      f"{time.time() - t0:.0f}s")
                json.dump(hatalar, open(hata_dosya, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
    json.dump(hatalar, open(hata_dosya, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Bitti: {ok} yeni, {len(hatalar)} hatalı (bilanco_veri/_hatalar.json)")
    return 0


def _oku(ticker: str):
    p = os.path.join(VERI_DIZIN, f"{ticker}.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- DÜZENLEME

def _ceyrek_sirali(cols) -> list:
    """'2026Q2' biçimli kolonları eskiden yeniye sıralar."""
    return sorted(cols, key=lambda c: (int(str(c)[:4]), int(str(c)[-1])))


# ENFLASYON MUHASEBESİ (TMS 29). Şirketler 2023 yıl sonu raporundan itibaren
# tablolarını DÖNEM SONU parasıyla veriyor; öncesi tarihi maliyet. Ölçüldü:
# düzeltme yapılmadan BİM'in 2023Q3 satış büyümesi %312, özkaynak kârlılığı
# %124 çıkıyordu — Mart parasıyla Haziran parası birbirinden çıkarılıyordu.
# Çözüm: her rakam resmi TÜFE ile AYNI sabit paraya çevrilir, sonra hesaplanır.
# TÜFE (ENAG değil) çünkü şirketler yeniden düzenlemeyi TÜFE ile yapıyor.
TMS29_ILK = (2023, 4)        # ilk yeniden düzenlenmiş rapor: 2023 yıl sonu


def _tms29(c: str) -> bool:
    return (int(c[:4]), int(c[-1])) >= TMS29_ILK


_TUFE = None


def _tufe() -> pd.Series:
    """Aylık TÜFE endeksi (2005-01 = 100), ay başı tarihli. Diske dondurulur."""
    global _TUFE
    if _TUFE is not None:
        return _TUFE
    yol = os.path.join(VERI_DIZIN, "_tufe.pkl")
    if os.path.exists(yol):
        _TUFE = pd.read_pickle(yol)
        return _TUFE
    import borsapy
    df = borsapy.Inflation().tufe().sort_index()
    aylik = pd.to_numeric(df["MonthlyInflation"], errors="coerce").fillna(0) / 100
    endeks = 100 * (1 + aylik).cumprod()
    endeks.index = pd.to_datetime(endeks.index).to_period("M").to_timestamp()
    os.makedirs(VERI_DIZIN, exist_ok=True)
    endeks.to_pickle(yol)
    _TUFE = endeks
    return endeks


def _tufe_ay(y: int, m: int) -> float:
    e = _tufe()
    t = pd.Timestamp(year=y, month=m, day=1)
    if t in e.index:
        return float(e.loc[t])
    return float(e.iloc[-1]) if t > e.index[-1] else np.nan


def _tufe_ceyrek_sonu(c: str) -> float:
    return _tufe_ay(int(c[:4]), 3 * int(c[-1]))


def _tufe_ceyrek_ort(c: str) -> float:
    y, q = int(c[:4]), int(c[-1])
    return float(np.nanmean([_tufe_ay(y, m) for m in range(3 * q - 2, 3 * q + 1)]))


def _para_birimi(tablo: pd.DataFrame, cols) -> dict:
    """Her sütunun HANGİ PARAYLA yazıldığı: c → ("nokta", ay) ya da ("tarihi", None).

    İş Yatırım ara dönem sütunlarını ilk açıklandığı hâliyle tutar, ama YIL SONU
    (Q4) sütunlarını sonraki yılın raporundaki YENİDEN DÜZENLENMİŞ
    karşılaştırmalı rakamla değiştirir — yani Q4(Y), Aralık(Y+1) parasıyla
    yazılıdır. Ölçüldü (özkaynak): BİM 2022Q3 11 → 2022Q4 58 → 2023Q1 26
    mlr TL; ASELSAN ve BİM'de her yıl Q4→Q1 geçişinde özkaynak ~0.81-0.86 kat
    "düşüyor". Sabit kural yerine her hisse ve her yıl sonu için VERİDEN
    tespit edilir (USD işlevsel para birimli şirketler farklı davranabilir):
    Q4'ün komşularına (Q3(Y) ve Q1(Y+1)) göre şişkinliği, bir yıllık TÜFE
    oranına 1'den daha yakınsa sütun Aralık(Y+1) parasıyla yazılmış sayılır.
    Sonraki Q1 henüz yoksa (en son yıl sonu) kendi parasıyla kabul edilir.
    """
    ozk = pd.to_numeric(tablo.loc["2N", cols], errors="coerce") if "2N" in tablo.index         else pd.Series(np.nan, index=cols)
    if ozk.isna().all() and "1BL" in tablo.index:
        ozk = pd.to_numeric(tablo.loc["1BL", cols], errors="coerce")
    out = {}
    for c in cols:
        y, q = int(c[:4]), int(c[-1])
        # Yalnız 2022+ yıl sonları (TMS 29 karşılaştırmalıları ilk kez 2023
        # yıllık raporunda) ve Y+1 yıllık raporu YAYIMLANMIŞSA (Q4(Y+1) var):
        # yoksa Q4(Y) henüz kimse tarafından yeniden yazılmamıştır. İlk sürüm
        # bu iki koşulu taşımıyordu ve BİM 2017Q4/2020Q4/2025Q4'ü yanlış işaretledi.
        if q == 4 and y >= 2022 and f"{y + 1}Q4" in cols:
            q3, q1n = f"{y}Q3", f"{y + 1}Q1"
            v, a, b = ozk.get(c), ozk.get(q3), ozk.get(q1n)
            if all(pd.notna(x) and x > 0 for x in (v, a, b)):
                r = np.log(v / np.sqrt(a * b))
                f = np.log(_tufe_ay(y + 1, 12) / _tufe_ay(y, 12))
                if abs(r - f) < abs(r):
                    out[c] = ("nokta", (y + 1, 12), "olcum")
                    continue
                out[c] = (("nokta", (y, 12), "olcum") if _tms29(c)
                          else ("tarihi", None, "olcum"))
                continue
            # Komşu çeyrek yok (çoğu 2023 halka arzı: 2022 için yalnız yıllık
            # rakam var) → tespit yapılamaz. ÇOĞUNLUK kuralı uygulanır ve
            # işaretlenir: ölçülebilen hisselerde Q4 sütunu 2022'de 322/402,
            # 2023'te 368/439, 2024'te 407/457 oranında sonraki yılın
            # parasıyla yazılı çıktı (gruplar net ayrışıyor: şişkinlik medyanı
            # 2.34 ve 1.30'a karşı 1.01 ve 1.08). Varsayım yapılmadan ADGYO'nun
            # 2022 reel satış büyümesi 221 kat çıkıyordu.
            out[c] = ("nokta", (y + 1, 12), "varsayim")
            continue
        out[c] = (("nokta", (y, 3 * q), "tarih") if _tms29(c)
                  else ("tarihi", None, "tarih"))
    return out


def _rejim(para: dict, c: str) -> str:
    return para[c][0]


def _reel_birikimli(v, c: str, para: dict):
    """Bir YTD değerini sabit paraya çevirir: nokta → o ayın TÜFE'si, tarihi →
    yıl başından o çeyreğe kadarki ayların ortalama TÜFE'si (birikim boyunca)."""
    tur, ay = para[c][:2]
    if tur == "nokta":
        return v / _tufe_ay(*ay) * 100
    y, q = int(c[:4]), int(c[-1])
    return v / np.nanmean([_tufe_ay(y, m) for m in range(1, 3 * q + 1)]) * 100


def _ttm_sabit(birikimli: pd.DataFrame, para: dict) -> pd.DataFrame:
    """Son 12 ay akışı, SABİT PARAYLA, doğrudan birikimli rakamlardan.

    Q4'te yıllık rakamın kendisidir. Diğer çeyreklerde
        TTM(Y,q) = YTD(Y,q) + Yıllık(Y-1) − YTD(Y-1,q)
    ve üç parça da AYNI muhasebe rejimindeyse kabul edilir; karışıksa NaN.
    Neden: yeniden düzenlenmiş (TMS 29) bir yıllık rakam ile tarihi maliyetli
    9 aylık rakamın farkı ekonomik bir şey ölçmez — parasal kayıp, yeniden
    değerleme gibi düzeltmelerin tamamı o farka yığılır (ASELSAN 2023'te
    negatif marj tam olarak buydu). TMS 29 geçişindeki birkaç çeyrek bu yüzden
    boş kalır; uydurmak yerine boş bırakmak tercih edildi.
    """
    df = birikimli.copy()
    cols = _ceyrek_sirali([c for c in df.columns if c != "ad"])
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for c in cols:
        y, q = int(c[:4]), int(c[-1])
        if q == 4:
            out[c] = _reel_birikimli(df[c], c, para)
            continue
        fy, pq = f"{y - 1}Q4", f"{y - 1}Q{q}"
        if fy not in df.columns or pq not in df.columns:
            out[c] = np.nan
            continue
        if len({_rejim(para, c), _rejim(para, fy), _rejim(para, pq)}) > 1:
            out[c] = np.nan
            continue
        out[c] = (_reel_birikimli(df[c], c, para) + _reel_birikimli(df[fy], fy, para)
                  - _reel_birikimli(df[pq], pq, para))
    return out


def _sabit_bakiye(bak: pd.DataFrame, cols, para: dict) -> pd.DataFrame:
    """Bilanço bakiyelerini yazıldıkları paranın TÜFE'siyle sabit paraya çevirir."""
    out = bak[cols].apply(pd.to_numeric, errors="coerce").copy()
    for c in cols:
        tur, ay = para[c][:2]
        out[c] = out[c] / (_tufe_ay(*ay) if tur == "nokta" else _tufe_ceyrek_sonu(c)) * 100
    return out


def _kalem(tablo: pd.DataFrame, cols, kod: str = None, ad: str = None) -> pd.Series:
    """Kalem koduna (ya da kod yoksa tam ada) göre sayısal seri."""
    if kod and kod in tablo.index:
        return pd.to_numeric(tablo.loc[kod, cols], errors="coerce")
    if ad:
        es = tablo.index[tablo["ad"] == ad]
        if len(es):
            return pd.to_numeric(tablo.loc[es[0], cols], errors="coerce")
    return pd.Series(np.nan, index=cols)


# ---------------------------------------------------------------- AÇIKLANMA TARİHİ

_KAP_TARIH = None


def _kap_aciklanma() -> dict:
    """(hisse, yıl, çeyrek) → ilk 'Finansal Rapor' yayın zamanı (KAP arşivi)."""
    global _KAP_TARIH
    if _KAP_TARIH is not None:
        return _KAP_TARIH
    out = {}
    if os.path.isdir(KAP_DIZIN):
        for ad in os.listdir(KAP_DIZIN):
            if not ad.endswith(".json"):
                continue
            for k in json.load(open(os.path.join(KAP_DIZIN, ad), encoding="utf-8")):
                if (k.get("subject") or "") != "Finansal Rapor":
                    continue
                y, p = k.get("year"), k.get("period")
                if not y or not p:
                    continue
                try:
                    ts = datetime.strptime(k["publishDate"], "%d.%m.%Y %H:%M:%S")
                except Exception:
                    continue
                for kod in (k.get("stockCodes") or "").split(","):
                    kod = kod.strip().upper()
                    if kod:
                        anahtar = (kod, int(y), int(p))
                        if anahtar not in out or ts < out[anahtar]:
                            out[anahtar] = ts
    _KAP_TARIH = out
    return out


def aciklanma(ticker: str, ceyrek: str) -> tuple:
    """(tarih, kaynak) — kaynak 'kap' ya da 'tahmini'."""
    y, q = int(ceyrek[:4]), int(ceyrek[-1])
    ts = _kap_aciklanma().get((ticker, y, q))
    if ts is not None:
        return pd.Timestamp(ts), "kap"
    donem_sonu = pd.Timestamp(year=y, month=3 * q, day=1) + pd.offsets.MonthEnd(0)
    return donem_sonu + pd.Timedelta(days=TAHMINI_GECIKME_GUN[q]), "tahmini"


# ---------------------------------------------------------------- ORANLAR

def _bol(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    return r.where(np.isfinite(r))


def oranlar(ticker: str) -> pd.DataFrame:
    """Bir hissenin çeyrek çeyrek oran tablosu (satır = çeyrek).

    Akışlar son 12 ay (TTM), bakiyeler dönem başı-sonu ortalaması — böylece
    mevsimsellik söner ve oranlar yıllık tanımlarla aynı ölçekte olur.
    """
    d = _oku(ticker)
    if d is None:
        return pd.DataFrame()
    t = d["tablo"]
    cols = _ceyrek_sirali([c for c in t.columns if c != "ad"])
    para = _para_birimi(t, cols)
    akis = _ttm_sabit(t[t.index.str[0].isin(["3", "4"])], para)   # gelir + nakit: TTM, sabit para
    rejim = pd.Series({c: _rejim(para, c) for c in cols})
    bak = _sabit_bakiye(t.loc[t.index.str[0].isin(["1", "2"])], cols, para)   # bilanço

    # Gelir tablosu (kodlar İş Yatırım XI_29 şablonundan; giderler mutlak değer —
    # bazı kalemler "(-)" adlı olduğu hâlde pozitif geliyor, BİM 3HC gibi)
    satis = _kalem(akis, cols, "3C")
    smm = _kalem(akis, cols, "3CA").abs()
    # FVÖK: İş Yatırım'ın "Finansman Gideri Öncesi Faaliyet Kârı" satırı. Vergi
    # öncesi kâr + finansman gideri diye geri hesaplanmaz: enflasyon
    # muhasebesindeki parasal kazanç kalemleri vergi öncesi kârı şişiriyor.
    _fk = t.index[t["ad"] == "Finansman Gideri Öncesi Faaliyet Karı/Zararı"]
    fvok = _kalem(akis, cols, _fk[0] if len(_fk) else None)
    if fvok.isna().all():
        fvok = _kalem(akis, cols, "3DF")                  # yoksa esas faaliyet kârı
    fin_gid = _kalem(akis, cols, "3HC").abs()
    if fin_gid.isna().all():
        fin_gid = _kalem(akis, cols, "4BB").abs()         # nakit akışı dipnotu
    net_kar = _kalem(akis, cols, "3L")
    ana_kar = _kalem(akis, cols, "3Z")
    faal_nakit = _kalem(akis, cols, "4C")
    # Bilanço
    stok = _kalem(bak, cols, "1AF")
    alacak = _kalem(bak, cols, "1AC")
    t_borc = _kalem(bak, cols, "2AAGAA")
    donen = _kalem(bak, cols, "1A")
    aktif = _kalem(bak, cols, "1BL")
    ozk = _kalem(bak, cols, "2N")
    ana_ozk = _kalem(bak, cols, "2O")

    # Akışlar zaten son 12 ay (TTM) — _ttm_sabit.
    s4, smm4, fvok4, fin4 = satis, smm, fvok, fin_gid
    nk4, ak4, fn4 = net_kar, ana_kar, faal_nakit

    def _ort(seri):      # dönem başı-sonu ortalaması; rejimler farklıysa yalnız dönem sonu
        ayni = rejim == rejim.shift(4)
        return ((seri + seri.shift(4)) / 2).where(ayni, seri)

    def _yillik(seri):   # bir yıl öncesine oran; rejimler farklıysa NaN
        return seri.shift(4).where(rejim == rejim.shift(4))

    o = pd.DataFrame(index=cols)
    # Faaliyet etkinliği
    o["stok_gun"] = _bol(_ort(stok), smm4) * 365
    o["alacak_gun"] = _bol(_ort(alacak), s4) * 365
    o["ticari_borc_gun"] = _bol(_ort(t_borc), smm4) * 365
    o["nakit_donusum_gun"] = o["stok_gun"] + o["alacak_gun"] - o["ticari_borc_gun"]
    o["donen_varlik_devir"] = _bol(s4, _ort(donen))
    o["ticari_borc_devir"] = _bol(smm4, _ort(t_borc))
    # Kârlılık
    o["net_kar_marji"] = _bol(nk4, s4)
    o["roa"] = _bol(nk4, _ort(aktif))
    o["roe"] = _bol(ak4, _ort(ana_ozk))
    # Sabit yükümlülükleri karşılama gücü
    o["faiz_karsilama"] = _bol(fvok4, fin4)
    o["net_kar_finansman"] = _bol(nk4, fin4)
    # Kâr kalitesi (kâr nakitle destekleniyor mu)
    o["nakit_kar_orani"] = _bol(fn4, nk4)
    # Büyüme (REEL, yıllık — tüm rakamlar zaten sabit TÜFE parasıyla):
    # son 12 ay vs bir önceki 12 ay; bakiyede bir yıl önceki dönem sonu.
    o["satis_buyume"] = _bol(s4, _yillik(s4)) - 1
    o["kar_buyume"] = _bol(nk4, _yillik(nk4).abs()) - np.sign(_yillik(nk4))
    o["ozkaynak_buyume"] = _bol(ozk, _yillik(ozk)) - 1
    # İstisna kuralı için: zarar dizisi ve özkaynak erimesi
    # Zarar dizisi: son 12 ayı zararlı geçen ARDIŞIK çeyrek sayısı. İstisna
    # kuralı (bilanco.degerlendir) ≤4 çeyrek = en fazla bir yıllık zarar.
    zarar = (net_kar < 0).astype(float).where(net_kar.notna())
    o["ust_uste_zarar"] = zarar.groupby((zarar != zarar.shift()).cumsum()).cumsum().where(zarar == 1, 0)
    _oz1 = _yillik(ozk)
    o["ozkaynak_kuculdu"] = (ozk < _oz1).astype(float).where(ozk.notna() & _oz1.notna())
    # Ham büyüklükler (sektör karşılaştırması ve sağlamlık kontrolleri için)
    o["satis_ttm"] = s4
    o["net_kar_ttm"] = nk4
    # Muhasebe rejimi: 2023 öncesi tarihi maliyet — stok ve duran varlıklar
    # eski fiyatla yazılı, TÜFE'ye bölmek bunu düzeltmez. Analiz ayırabilsin.
    o["muhasebe"] = ["tms29" if para[c][0] == "nokta" else "tarihi" for c in cols]
    o["yil_sonu_sonraki_para"] = [para[c][0] == "nokta" and para[c][1][0] > int(c[:4])
                                  for c in cols]
    # True: sütunun hangi parayla yazıldığı ölçülemedi, çoğunluk kuralıyla varsayıldı
    o["para_varsayim"] = [para[c][2] == "varsayim" for c in cols]
    # Açıklanma
    ac = [aciklanma(ticker, c) for c in cols]
    o["aciklanma"] = [a[0] for a in ac]
    o["aciklanma_kaynak"] = [a[1] for a in ac]
    o.index.name = "ceyrek"
    return o


# ---------------------------------------------------------------- EŞİKLER

def esikler() -> dict:
    with open(ESIK_DOSYA, encoding="utf-8") as f:
        return json.load(f)["oranlar"]


def degerlendir(satir: pd.Series, sektor_medyan: pd.Series = None) -> dict:
    """Bir hisse-çeyreği ders notu kriterlerine göre değerlendirir.

    Eşik verilmişse eşiğe, verilmemişse SEKTÖR MEDYANINA göre ("mümasil
    şirketlerle karşılaştır"). Dönüş: {oran: True/False/None}; None = veri yok
    ya da karşılaştırılacak bir şey yok. Hiçbir puan üretmez — sayım altyapısı.
    """
    sonuc = {}
    for ad, k in esikler().items():
        if ad.startswith("_"):
            continue
        v = satir.get(ad)
        if v is None or pd.isna(v):
            sonuc[ad] = None
            continue
        ref = k.get("esik")
        if ref is None and sektor_medyan is not None:
            ref = sektor_medyan.get(ad)
        if ref is None or pd.isna(ref):
            sonuc[ad] = None
            continue
        sonuc[ad] = bool(v <= ref) if k["yon"] == "az" else bool(v >= ref)
    # İstisna: tek dönemlik zarar + özkaynak küçülmüyorsa marj kriteri geçer
    # (ders notu: "ASELSAN gibi sağlam şirket bir dönem zarar etse de sorun
    # olmayabilir"). İsim listesi yerine kural — öznel olmasın.
    if sonuc.get("net_kar_marji") is False:
        if satir.get("ust_uste_zarar", 0) <= 4 and satir.get("ozkaynak_kuculdu") == 0:
            sonuc["net_kar_marji"] = True
    return sonuc


def kesit(ceyrek: str) -> pd.DataFrame:
    """Tüm evrenin o çeyrekteki oranları + sektör medyanına göre değerlendirme."""
    from bist_analyzer import _sector_of
    satirlar = []
    for t in _evren():
        o = oranlar(t)
        if o.empty or ceyrek not in o.index:
            continue
        r = o.loc[ceyrek].copy()
        r["ticker"], r["sektor"] = t, _sector_of(t)
        satirlar.append(r)
    df = pd.DataFrame(satirlar).set_index("ticker")
    if df.empty:
        return df
    sayisal = [c for c in esikler() if not c.startswith("_") and c in df.columns]
    medyan = df.groupby("sektor")[sayisal].median()
    deg = []
    for t, r in df.iterrows():
        d = degerlendir(r, medyan.loc[r["sektor"]] if r["sektor"] in medyan.index else None)
        gecen = sum(1 for v in d.values() if v is True)
        olculen = sum(1 for v in d.values() if v is not None)
        deg.append({"ticker": t, "gecen": gecen, "olculen": olculen,
                    **{f"ok_{k}": v for k, v in d.items()}})
    return df.join(pd.DataFrame(deg).set_index("ticker"))


# ---------------------------------------------------------------- CLI

def ozet() -> int:
    if not os.path.isdir(VERI_DIZIN):
        print("bilanco_veri/ yok — önce: python bilanco.py cek")
        return 0
    # "_" ile başlayanlar yardımcı dosyalar (_tufe.pkl), hisse değil
    dosyalar = [f for f in os.listdir(VERI_DIZIN) if f.endswith(".pkl") and not f.startswith("_")]
    hatalar = {}
    hp = os.path.join(VERI_DIZIN, "_hatalar.json")
    if os.path.exists(hp):
        hatalar = json.load(open(hp, encoding="utf-8"))
    ceyrek_say, ilk = [], []
    for f in dosyalar:
        d = pickle.load(open(os.path.join(VERI_DIZIN, f), "rb"))
        c = _ceyrek_sirali([k for k in d["tablo"].columns if k != "ad"])
        ceyrek_say.append(len(c))
        if c:
            ilk.append(c[0])
    print(f"Evren {len(_evren())} | çekilmiş {len(dosyalar)} | hatalı {len(hatalar)}")
    if ceyrek_say:
        print(f"Çeyrek sayısı: medyan {int(np.median(ceyrek_say))}, en az {min(ceyrek_say)} | "
              f"en eski {min(ilk)}")
    tur = {}
    for v in hatalar.values():
        tur[v.split(":")[0]] = tur.get(v.split(":")[0], 0) + 1
    if tur:
        print("Hata türleri:", tur)
    return 0


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else ""
    if komut == "cek":
        sys.exit(cek(yenile="--yenile" in sys.argv))
    if komut == "ozet":
        sys.exit(ozet())
    if komut == "oran" and len(sys.argv) > 2:
        pd.set_option("display.width", 200, "display.max_columns", 30)
        o = oranlar(sys.argv[2].upper())
        print(o.drop(columns=["satis_ttm", "net_kar_ttm"]).tail(8).round(3).T.to_string())
        sys.exit(0)
    if komut == "kesit" and len(sys.argv) > 2:
        pd.set_option("display.width", 200, "display.max_columns", 30)
        k = kesit(sys.argv[2])
        print(k[["sektor", "gecen", "olculen"]].sort_values("gecen", ascending=False).to_string())
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
