"""
pipeline_backtest — Tam-pipeline tarihsel simülasyon (Roadmap-A, Ekim sunumu).

BUGÜNKÜ portföy kurma hattını geçmişte point-in-time işletir:
    likidite filtresi (o günkü hacimle) → momentum skoru → SMA200/skor eşiği
    → sektör limiti (max 2) → korelasyon elemesi (>0.85) → ATR-ters ağırlık
    → aylık rebalans → işlem maliyeti (round-trip %0.4 × devir)

Kontrol grupları (bilimsel kıyas):
    - XU100 endeksi
    - Eşit ağırlık BIST-30 (aylık rebalans, aynı maliyet)
    - Rastgele-6 portföy (50 tohum ortalaması + %5-%95 bandı)

Çıktı: pipeline_nav.csv (günlük NAV serileri) + konsol özeti
(nominal, CAGR, ENAG-reel, vs XU100, MaxDD, Sharpe).

Dürüstlük notları (sunumda belirtilecek):
    - Evren bugünün likit listesi → delist olanlar yok (survivorship bias,
      hafif iyimserlik). Kısmi telafi: her tarihte O GÜNÜN hacim filtresi.
    - Sentiment + analist primi tarihsel üretilemez → bu simülasyon sistemin
      TEKNİK çekirdeğini test eder; canlı sistem üstüne haber/prim katmanı ekler.
    - Rebalanslar arası stop kontrolü yok (kalibrasyon geniş stopların
      sonuçları az değiştirdiğini gösterdi); ATR stoplar canlıda ek sigorta.

Tekrarlanabilirlik: evren backtest_evren.json'a, fiyat paneli yerel
backtest_panel_<yil>y.pkl'e DONDURULUR. Tüm komutlar bu sabit paneli kullanır,
yani aynı komut her koşuda aynı sayıyı verir (bkz. TEKRARLANABİLİRLİK ÇAPASI).

Kullanım: python pipeline_backtest.py dondur [yil=5]        # evren+panel dondur
          python pipeline_backtest.py run   [yil_sayisi=5]   # tam simülasyon
          python pipeline_backtest.py karsilastir [yil=5]    # yönetilen vs yönetimsiz
          python pipeline_backtest.py sweep [yil_sayisi=5]   # yönetim kuralı taraması
          python pipeline_backtest.py validate [yil=8]        # örneklem-dışı doğrulama
"""

import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Konsol kodlaması: Windows'ta varsayılan cp1254, çıktıdaki '→' gibi karakterleri
# kodlayamayıp UnicodeEncodeError ile koşuyu ÇÖKERTİYORDU — hem de sonuçlar
# hesaplandıktan sonra, yazdırma sırasında. Saatlerce koşan bir simülasyonun
# çıktısını bir ok işareti yüzünden kaybetmek kabul edilemez.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from bist_analyzer import (  # noqa: E402
    get_scan_universe, _sector_of, InflationEngine, BIST_STOCKS,
    BacktestEngine, _UNIVERSE_CACHE,
)

MAX_POS      = 6          # orta/Dengeli profili
SECTOR_CAP   = 2
CORR_LIMIT   = 0.85
SCORE_MIN    = 50.0
MIN_TL_VOL   = 20_000_000
COST_RT      = 0.004      # round-trip %0.4 (uygulamayla tutarlı)
RANDOM_SEEDS = 50


# TEKRARLANABİLİRLİK ÇAPASI (2026-08-03)
#
# Ölçüldü: aynı komut iki kez koşturulduğunda panel 468 ve 464 hisse çıktı
# (yfinance bazı sembolleri rastgele boş döndürüyor) ve 5 yıllık yönetimsiz
# getiri %9738 → %7810 oynadı. 6 pozisyonluk yoğun bir portföyde iki üç
# hissenin varlığı bileşik sonucu ~%20 kaydırıyor.
#
# Akademik sunumda savunulacak bir sayı, koşuya göre değişemez. Bu yüzden:
#   1. EVREN dosyaya sabitlenir (backtest_evren.json — depoya girer, küçük)
#   2. FİYAT PANELİ diske önbelleklenir (yerel, .gitignore — birebir tekrar)
#   3. Başarısız semboller tek tek YENİDEN denenir (rastgele boş dönüşe karşı)
#   4. feats sözlüğü SIRALI kurulur — dict sırası eşitlik bozmalarını ve
#      rastgele-kontrol grubunun seçimini etkiliyordu
# Panel silinirse donmuş evrenden yeniden indirilir; sembol listesi sabit
# kaldığı için sonuç yine aynı çıkar.
EVREN_DOSYA     = "backtest_evren.json"
PANEL_ONBELLEK  = "backtest_panel_{yil}y.pkl"


def _download_panel(tickers, years, tekrar=2):
    """Kapanış/hacim panelleri (chunked bulk) + eksikleri yeniden deneme."""
    closes, vols = {}, {}
    period = f"{years + 2}y"

    def _al(sembol_listesi, boyut):
        for i in range(0, len(sembol_listesi), boyut):
            parti = sembol_listesi[i:i+boyut]
            try:
                bulk = yf.download(parti, period=period, interval="1d",
                                   auto_adjust=True, progress=False, group_by="ticker")
                for s in parti:
                    t = s[:-3]
                    if t in closes:
                        continue
                    try:
                        df = bulk[s] if isinstance(bulk.columns, pd.MultiIndex) else bulk
                        cl = df["Close"].dropna()
                        if len(cl) >= 260:
                            closes[t] = cl
                            vols[t] = df["Volume"].reindex(cl.index).fillna(0)
                    except Exception:
                        continue
            except Exception as exc:
                print(f"  indirme hatası (parti {i}): {exc}")

    syms = [t + ".IS" for t in tickers]
    _al(syms, 100)
    print(f"  veri: {len(closes)}/{len(syms)} sembol (ilk geçiş)")

    # Eksikleri küçük partilerle yeniden dene — yfinance toplu sorguda
    # sembolleri sebepsiz boş döndürebiliyor, tekil istekte gelebiliyor.
    for tur in range(tekrar):
        eksik = [s for s in syms if s[:-3] not in closes]
        if not eksik:
            break
        print(f"  eksik {len(eksik)} sembol yeniden deneniyor ({tur+1}/{tekrar})...")
        _al(eksik, 10)
        print(f"  veri: {len(closes)}/{len(syms)} sembol")
    return closes, vols


def _panel_hazirla(years=5, yenile=False, evren_yenile=False):
    """Donmuş evren + önbellekli fiyat paneli. Aynı girdi → aynı sayı.

    İki ayrı tazeleme düğmesi var ve AYRI olmaları şart:
      yenile=True       → yalnız fiyat panelini yeniden indirir (evren sabit)
      evren_yenile=True → evreni de canlı TV'den yeniden çeker

    Neden ayrı: evren tek bir çapadır ve farklı yıl pencereleri (5y, 8y) AYNI
    listeden beslenmelidir. Başlangıçta `yenile` ikisini birden yeniliyordu ve
    `dondur 8` çalıştırılınca 5 yıllık sonuçların dayandığı liste sessizce
    üzerine yazıldı: iki panel iki farklı evrene oturdu, dosya ise yalnız
    yenisini gösteriyordu. Yani panel silinip yeniden kurulsa 5y sayıları
    tutmayacaktı — tekrarlanabilirlik çapasının tam da kendisi kırılmıştı.
    Likidite sıralaması gün içi hacimle her gün değiştiği için evreni
    tazelemek nadir ve bilinçli bir karar olmalıdır.
    """
    import hashlib
    import json
    import os
    import pickle

    onbellek = PANEL_ONBELLEK.format(yil=years)
    if not yenile and not evren_yenile and os.path.exists(onbellek):
        with open(onbellek, "rb") as f:
            d = pickle.load(f)
        print(f"Panel ÖNBELLEKTEN: {len(d['closes'])} hisse | dondurma {d['tarih']} "
              f"| parmak izi {d['iz']}")
        return d["closes"], d["vols"]

    # Evren: dosyadan oku (varsa), yoksa canlı çek ve dondur
    if os.path.exists(EVREN_DOSYA) and not evren_yenile:
        with open(EVREN_DOSYA, encoding="utf-8") as f:
            meta = json.load(f)
        uni = meta["tickers"]
        print(f"Evren DOSYADAN: {len(uni)} hisse (donduruldu {meta['tarih']})")
    else:
        uni = list(get_scan_universe())      # likidite sıralı (bkz. get_scan_universe)
        meta = {
            "tarih": datetime.now().strftime("%Y-%m-%d"),
            "kaynak": "get_scan_universe('BIST') — TradingView, hacme göre azalan",
            "not": "Tekrarlanabilirlik çapası: backtest bu SABİT listeyi kullanır. "
                   "Listeyi değiştirmek geçmiş sonuçları karşılaştırılamaz kılar.",
            "tickers": uni,
        }
        with open(EVREN_DOSYA, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print(f"Evren DONDURULDU → {EVREN_DOSYA} ({len(uni)} hisse)")

    closes, vols = _download_panel(uni, years)
    iz = hashlib.sha256(
        "|".join(f"{t}:{len(closes[t])}:{closes[t].iloc[-1]:.4f}"
                 for t in sorted(closes)).encode()).hexdigest()[:12]
    with open(onbellek, "wb") as f:
        pickle.dump({"closes": closes, "vols": vols, "iz": iz,
                     "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "evren": len(uni)}, f)
    print(f"Panel ÖNBELLEĞE yazıldı → {onbellek} | {len(closes)}/{len(uni)} hisse "
          f"| parmak izi {iz}")
    return closes, vols


def _precompute(closes, vols):
    """Hisse başına 1 kez: momentum skor serisi, ATR%, TL hacim, getiri paneli.

    SIRALI kurulur: dict ekleme sırası indirme sırasına bağlıydı, bu da
    _pick_at'teki eşit skorların bozulmasını ve rastgele-kontrol grubunun
    çekilişini koşudan koşuya değiştiriyordu.
    """
    feats = {}
    for t in sorted(closes):
        cl = closes[t]
        try:
            df = pd.DataFrame({"Close": cl, "High": cl, "Low": cl, "Volume": vols[t]})
            scores, atr_s, _rsi = BacktestEngine._vectorized_scores(df, style="momentum")
            feats[t] = {
                "close": cl,
                "score": scores,
                "atr_pct": (atr_s / cl * 100).fillna(99),
                "sma200": cl.rolling(200, min_periods=100).mean(),
                "tlvol": (cl * vols[t]).rolling(30, min_periods=10).mean(),
                "ret": cl.pct_change(),
            }
        except Exception:
            continue
    return feats


def _pick_at(date, feats, ret_panel):
    """Bugünkü propose() mantığının point-in-time kopyası → {ticker: weight}."""
    cands = []
    for t, f in feats.items():
        cl = f["close"]
        ix = cl.index.searchsorted(date, side="right") - 1
        if ix < 260:
            continue
        d = cl.index[ix]
        if (date - d).days > 7:      # o civarda veri yoksa (delist/askı) alma
            continue
        price, score = float(cl.iloc[ix]), float(f["score"].iloc[ix])
        sma, tlv, atr = float(f["sma200"].iloc[ix] or 0), float(f["tlvol"].iloc[ix] or 0), float(f["atr_pct"].iloc[ix])
        if tlv < MIN_TL_VOL or price <= 0 or not sma or price < sma or score < SCORE_MIN:
            continue
        mom3 = (price / float(cl.iloc[max(0, ix-63)]) - 1) * 100
        cands.append((t, score + mom3 * 0.4, atr))
    cands.sort(key=lambda x: -x[1])

    picks, sec_n, kept = [], {}, []
    win = ret_panel.loc[:date].tail(120)
    for t, _rank, atr in cands:
        sec = _sector_of(t)
        if sec_n.get(sec, 0) >= SECTOR_CAP:
            continue
        if kept and t in win.columns:
            cors = win[kept].corrwith(win[t]).abs()
            if (cors > CORR_LIMIT).any():
                continue
        picks.append((t, atr))
        kept.append(t) if t in win.columns else None
        sec_n[sec] = sec_n.get(sec, 0) + 1
        if len(picks) >= MAX_POS:
            break
    if not picks:
        return {}
    inv = np.array([1 / max(a, 0.5) for _, a in picks])
    w = inv / inv.sum()
    w = np.minimum(w, 0.30); w = w / w.sum()
    return {t: float(wi) for (t, _a), wi in zip(picks, w)}


def _walk(weights_by_date, closes, dates):
    """Aylık hedef ağırlıklar → günlük NAV (devir maliyetli).

    Ağırlıklar rebalanslar ARASINDA fiyatla sürüklenir (_walk_managed ile aynı
    kurgu). Önceden `cur_w` ay boyunca sabit tutuluyordu; bu, portföyün her gün
    sıfır maliyetle hedef ağırlığa döndürüldüğü anlamına gelir — yani yönetimsiz
    referans modeli aslında GÜNLÜK ÜCRETSİZ REBALANS yapan bir modeldi. Bu,
    kontrol grubunu haksız yere güçlendirir (bedava volatilite hasadı) ve
    rebalans günündeki devir hesabını da bozar: sürüklenme yokken hedef
    değişmediğinde devir 0 çıkar, gerçekte ise sürüklenmiş ağırlıkları hedefe
    çekmek maliyetlidir. Yönetilen/yönetimsiz karşılaştırması ancak ikisi de
    aynı fizik kurallarına tabiyken anlamlıdır.
    """
    nav, cur_w = [1.0], {}
    rebal = sorted(weights_by_date)
    ri = 0
    for i in range(1, len(dates)):
        d0, d1 = dates[i-1], dates[i]
        # günlük getiri + ağırlık kayması (fiyatı okunamayan isim ağırlığını korur)
        r = 0.0
        drift = {}
        for t, w in cur_w.items():
            g = 0.0
            cl = closes.get(t)
            if cl is not None:
                try:
                    p0 = cl.asof(d0); p1 = cl.asof(d1)
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                        g = p1 / p0 - 1
                        r += w * g
                except Exception:
                    g = 0.0
            drift[t] = w * (1 + g)
        v = nav[-1] * (1 + r)
        # Payları yeniden normalize et: _pick_at ağırlıkları 1.0'a topluyor
        # (nakit yok), dolayısıyla toplam = 1 + r ve bölme payları tam korur.
        toplam = sum(drift.values())
        cur_w = {t: w / toplam for t, w in drift.items()} if toplam > 0 else {}
        # rebalans günü mü?
        while ri < len(rebal) and rebal[ri] <= d1:
            new_w = weights_by_date[rebal[ri]]
            turnover = sum(abs(new_w.get(t, 0) - cur_w.get(t, 0))
                           for t in set(new_w) | set(cur_w)) / 2
            v *= (1 - COST_RT * turnover)
            cur_w = dict(new_w)
            ri += 1
        nav.append(v)
    return pd.Series(nav, index=dates)


def _stats(nav, xu, label):
    tot = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    reel = InflationEngine.real_return(tot, str(nav.index[0].date()), str(nav.index[-1].date()))
    reel_cagr = ((1 + reel / 100) ** (1 / yrs) - 1) * 100
    xu_tot = (xu.iloc[-1] / xu.iloc[0] - 1) * 100
    dd = ((nav / nav.cummax()) - 1).min() * 100
    dr = _reel_seri(nav).pct_change().dropna()      # Sharpe REEL getiriden
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    print(f"{label:<24} nominal %{tot:>8.0f} | CAGR %{cagr:>5.1f} | ENAG-reel %{reel:>7.1f} "
          f"(yıllık %{reel_cagr:>+5.1f}) | vsXU100 {tot-xu_tot:>+7.0f}p | MaxDD %{dd:>5.1f} | Sharpe {sharpe:.2f}")
    return {"label": label, "nominal": tot, "cagr": cagr, "reel": reel,
            "reel_yillik": reel_cagr, "maxdd": dd, "sharpe": sharpe}


def run(years=5):
    t0 = time.time()
    print(f"=== Tam-Pipeline Simülasyonu | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    print(f"Öznitelikler: {len(feats)} hisse ({time.time()-t0:.0f}s)")

    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})

    # Ortak tarih ekseni + aylık rebalans günleri
    xu = yf.download("XU100.IS", period=f"{years+1}y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(xu.columns, pd.MultiIndex):
        xu.columns = xu.columns.get_level_values(0)
    xu = xu["Close"].dropna()
    start = xu.index[-1] - pd.DateOffset(years=years)
    dates = xu.index[xu.index >= start]
    rebal_dates = pd.date_range(dates[0], dates[-1], freq="MS")
    rebal_dates = [dates[dates.searchsorted(d)] for d in rebal_dates if d <= dates[-1]]

    # 1) SİSTEM
    wbd = {}
    for d in rebal_dates:
        w = _pick_at(d, feats, ret_panel)
        if w:
            wbd[d] = w
    print(f"Rebalans: {len(wbd)}/{len(rebal_dates)} ayda portföy kurulabildi")
    nav_sys = _walk(wbd, closes, list(dates))

    # 2) EŞİT AĞIRLIK BIST-30
    b30 = [t for t in BIST_STOCKS["BIST 30"] if t in closes]
    wbd_b30 = {d: {t: 1/len(b30) for t in b30} for d in rebal_dates}
    nav_b30 = _walk(wbd_b30, closes, list(dates))

    # 3) RASTGELE-6 (50 tohum)
    rng_navs = []
    all_t = sorted(feats)   # deterministik: dict sırası koşudan koşuya değişiyordu
    for seed in range(RANDOM_SEEDS):
        rng = np.random.default_rng(seed)
        wbd_r = {}
        for d in rebal_dates:
            elig = [t for t in all_t
                    if feats[t]["close"].index.searchsorted(d) > 260]
            if len(elig) >= MAX_POS:
                sel = rng.choice(elig, MAX_POS, replace=False)
                wbd_r[d] = {t: 1/MAX_POS for t in sel}
        rng_navs.append(_walk(wbd_r, closes, list(dates)))
    rng_final = np.array([n.iloc[-1] for n in rng_navs])
    nav_rng_mean = pd.concat(rng_navs, axis=1).mean(axis=1)

    xu_n = xu.loc[dates] / xu.loc[dates].iloc[0]

    print(f"\n{'='*118}")
    s1 = _stats(nav_sys, xu_n, "SİSTEM (pipeline)")
    s2 = _stats(nav_b30, xu_n, "Eşit ağırlık BIST-30")
    s3 = _stats(nav_rng_mean, xu_n, "Rastgele-6 (ortalama)")
    s4 = _stats(xu_n, xu_n, "XU100")
    beat = (nav_sys.iloc[-1] > np.percentile(rng_final, 95))
    print(f"\nRastgele dağılımda sistemin yeri: son NAV {nav_sys.iloc[-1]:.2f} | "
          f"rastgele %5-%95 bandı [{np.percentile(rng_final,5):.2f}, {np.percentile(rng_final,95):.2f}] "
          f"→ {'>%95 (rastgeleden anlamlı iyi)' if beat else 'bandın içinde (rastgeleden ayrışamadı)'}")

    out = pd.DataFrame({"sistem": nav_sys, "bist30_esit": nav_b30,
                        "rastgele_ort": nav_rng_mean, "xu100": xu_n})
    out.to_csv("pipeline_nav.csv")
    print(f"\npipeline_nav.csv yazıldı ({len(out)} gün) | toplam süre {time.time()-t0:.0f}s")




# YÖNETİM KURALI TARAMASI (2026-07-30)
#
# PortfolioDoctor'ın kuralları profesyonel pratikten gelen makul varsayılanlardı
# ama BIST verisinde ölçülmemişti. Burada aynı point-in-time hattın üstüne
# GÜNLÜK yönetim katmanı eklenir ve parametreler tek tek süpürülür.
#
# Dürüstlük notu: 5-6 yıl tek piyasa demek, parametreler arası küçük farkların
# GÜRÜLTÜ olduğu demektir. Bu yüzden her sonuç yıl-bazında da kırılır; bir
# değer ancak çoğu alt dönemde tutarlıysa "iyi" sayılır. Tek bir toplam sayıya
# bakıp parametre seçmek, tam da kaçınmak istediğimiz aşırı-uyum hatasıdır.

# CANLI KURALLARDAN TÜRETİLİR — elle kopyalanmaz.
#
# Burası eskiden bağımsız bir sözlüktü ve sürüklendi: PortfolioDoctor.KURAL'da
# takas_farki 12 → 6 yapıldığında bu kopya 12'de kaldı, yani tarama üretimde
# artık geçerli olmayan bir "varsayılan"a karşı ölçüm yapacaktı. Aynı sayının
# iki yerde yaşaması, tam da bu projede birkaç kez canımızı yakan hata sınıfı
# (bkz. _walk vs _walk_managed). Tek kaynak: bist_analyzer.PortfolioDoctor.KURAL.
def _mgmt_varsayilan() -> dict:
    from bist_analyzer import PortfolioDoctor
    return {k: float(PortfolioDoctor.KURAL[k]) for k in
            ("trailing_pct", "dusus_freni", "max_pozisyon", "skor_cikis", "takas_farki")}


MGMT_VARSAYILAN = _mgmt_varsayilan()


def _skor_at(feats, t, date):
    """t hissesinin `date` tarihindeki point-in-time skoru (yoksa None)."""
    f = feats.get(t)
    if f is None:
        return None
    ix = f["score"].index.searchsorted(date, side="right") - 1
    if ix < 0:
        return None
    v = float(f["score"].iloc[ix])
    return v if np.isfinite(v) else None


def _walk_managed(weights_by_date, closes, dates, feats, par):
    """Aylık hedefler + GÜNLÜK yönetim (trailing stop, skor çıkışı, tavan,
    düşüş freni, takas eşiği). Nakit gerçek pozisyondur (getirisi 0)."""
    trail = par["trailing_pct"] / 100.0
    fren  = par["dusus_freni"] / 100.0
    tavan = par["max_pozisyon"] / 100.0
    skor_c = par["skor_cikis"]
    takas  = par["takas_farki"]

    nav = [1.0]
    cur = {}          # ticker -> {"w": ağırlık, "peak": zirve fiyat}
    nav_peak = 1.0    # 90 günlük pencere zirvesi olarak güncellenir
    rebal = sorted(weights_by_date)
    ri = 0
    islem_maliyeti_toplam = 0.0

    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]

        # 1) Günlük getiri + ağırlık kayması (nakit 0 getirir)
        r = 0.0
        yeni = {}
        for t, st in cur.items():
            cl = closes.get(t)
            if cl is None:
                yeni[t] = st
                continue
            try:
                p0, p1 = cl.asof(d0), cl.asof(d1)
            except Exception:
                yeni[t] = st
                continue
            if not (pd.notna(p0) and pd.notna(p1) and p0 > 0):
                yeni[t] = st
                continue
            g = p1 / p0 - 1
            r += st["w"] * g
            yeni[t] = {"w": st["w"] * (1 + g), "peak": max(st["peak"], float(p1))}
        cur = yeni
        v = nav[-1] * (1 + r)
        # 90 günlük pencere zirvesi (tüm zamanlar zirvesi freni kalıcı kilitler)
        nav_peak = max(nav[-90:] + [v])

        # Ağırlıkları normalize et (toplam = portföy değeri payı)
        toplam = sum(s["w"] for s in cur.values())
        if toplam > 1.0 + 1e-9:      # fiyat artışıyla 1'i aşamaz; koruma
            for s in cur.values():
                s["w"] /= toplam

        # 2) GÜNLÜK YÖNETİM
        satilan_w = 0.0
        for t in list(cur):
            cl = closes.get(t)
            if cl is None:
                continue
            try:
                p = cl.asof(d1)
            except Exception:
                continue
            if not (pd.notna(p) and p > 0):
                continue
            # a) takip eden stop
            if cur[t]["peak"] > 0 and float(p) <= cur[t]["peak"] * (1 - trail):
                satilan_w += cur[t]["w"]; del cur[t]; continue
            # b) skor çıkışı
            sk = _skor_at(feats, t, d1)
            if sk is not None and sk < skor_c:
                satilan_w += cur[t]["w"]; del cur[t]; continue
            # c) pozisyon tavanı
            if cur[t]["w"] > tavan:
                satilan_w += cur[t]["w"] - tavan
                cur[t]["w"] = tavan
        if satilan_w > 0:
            v *= (1 - COST_RT / 2 * satilan_w)     # tek yön maliyet
            islem_maliyeti_toplam += COST_RT / 2 * satilan_w

        # 3) Rebalans günü: takas eşiğiyle hedefe yaklaş
        while ri < len(rebal) and rebal[ri] <= d1:
            hedef = weights_by_date[rebal[ri]]
            frende = v < nav_peak * (1 - fren)
            if frende:
                # Düşüş freni: yeni alım yok, mevcut korunur
                ri += 1
                continue
            yeni_w = {}
            for t, w in hedef.items():
                if t in cur:
                    yeni_w[t] = w
                    continue
                # Tutulanların en zayıfını bul; aday onu takas_farki kadar geçmeli
                if cur:
                    zayif = min(cur, key=lambda x: (_skor_at(feats, x, d1) or 0))
                    z_sk = _skor_at(feats, zayif, d1) or 0
                    a_sk = _skor_at(feats, t, d1) or 0
                    if a_sk - z_sk < takas:
                        continue
                yeni_w[t] = w
            # Hedefte olmayan ama tutulanlar: skoru hâlâ iyiyse korunur
            for t, st in cur.items():
                if t not in yeni_w:
                    sk = _skor_at(feats, t, d1)
                    if sk is not None and sk >= skor_c:
                        yeni_w[t] = st["w"]
            top = sum(yeni_w.values())
            if top > 1.0:
                yeni_w = {t: w / top for t, w in yeni_w.items()}
            devir = sum(abs(yeni_w.get(t, 0) - cur.get(t, {}).get("w", 0))
                        for t in set(yeni_w) | set(cur)) / 2
            v *= (1 - COST_RT * devir)
            islem_maliyeti_toplam += COST_RT * devir
            yeni_cur = {}
            for t, w in yeni_w.items():
                pk = cur[t]["peak"] if t in cur else 0.0
                try:
                    p = closes[t].asof(d1)
                    pk = max(pk, float(p) if pd.notna(p) else 0.0)
                except Exception:
                    pass
                yeni_cur[t] = {"w": w, "peak": pk}
            cur = yeni_cur
            ri += 1

        nav.append(v)

    s = pd.Series(nav, index=dates)
    s.attrs["maliyet"] = islem_maliyeti_toplam
    return s


def _reel_seri(nav):
    """NAV serisini ENAG ile reelleştirir (Sharpe için ŞART)."""
    try:
        from bist_analyzer import InflationEngine as _IE
        d0 = nav.index[0].strftime("%Y-%m-%d"); d1 = nav.index[-1].strftime("%Y-%m-%d")
        defl = _IE.deflator(d0, d1)
        n = max(1, (nav.index[-1] - nav.index[0]).days)
        g = defl ** (1.0 / n) - 1
        idx = pd.Series([(1 + g) ** ((d - nav.index[0]).days) for d in nav.index],
                        index=nav.index)
        return nav / idx
    except Exception:
        return nav


def _ozet(nav):
    """Toplam getiri (nominal), Sharpe (REEL getiriden), MaxDD.

    Sharpe nominal getiriden hesaplanırsa yüksek enflasyonda 5 kata kadar
    şişer — enflasyon ortalamayı büyütür ama günlük oynaklığı büyütmez.
    """
    tot = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    dg = _reel_seri(nav).pct_change().dropna()
    sharpe = (dg.mean() / dg.std() * np.sqrt(252)) if dg.std() > 0 else 0.0
    dd = ((nav / nav.cummax()) - 1).min() * 100
    return tot, float(sharpe), float(dd)


def sweep(years=5):
    """Yönetim parametrelerini tek tek süpür — her biri için toplam getiri,
    Sharpe, MaxDD ve YIL BAZINDA tutarlılık."""
    t0 = time.time()
    print(f"=== Yönetim Kuralı Taraması | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})
    print(f"Panel: {len(feats)} hisse ({time.time()-t0:.0f}s)")

    xu = yf.download("XU100.IS", period=f"{years+1}y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(xu.columns, pd.MultiIndex):
        xu.columns = xu.columns.get_level_values(0)
    xu = xu["Close"].dropna()
    start = xu.index[-1] - pd.DateOffset(years=years)
    dates = list(xu.index[xu.index >= start])
    rebal_dates = pd.date_range(dates[0], dates[-1], freq="MS")
    rebal_dates = [dates[pd.Index(dates).searchsorted(d)] for d in rebal_dates if d <= dates[-1]]

    wbd = {}
    for d in rebal_dates:
        w = _pick_at(d, feats, ret_panel)
        if w:
            wbd[d] = w
    print(f"Aylık portföy: {len(wbd)}/{len(rebal_dates)} ({time.time()-t0:.0f}s)\n")

    # Referans: yönetimsiz (sadece aylık rebalans) — mevcut sistem
    nav_base = _walk(wbd, closes, dates)
    b_tot, b_sh, b_dd = _ozet(nav_base)
    print(f"{'YÖNETİMSİZ (referans)':<34} getiri %{b_tot:+8.1f} | Sharpe {b_sh:5.2f} | MaxDD %{b_dd:6.1f}\n")

    taranacak = {
        "trailing_pct": [10, 15, 20, 25, 100],   # 100 = takip eden stop KAPALI
        "dusus_freni":  [10, 15, 20, 100],       # 100 = fren kapalı
        "max_pozisyon": [20, 25, 30, 40],
        "takas_farki":  [0, 6, 12, 20],
    }
    sonuclar = {}
    for par_ad, degerler in taranacak.items():
        print(f"--- {par_ad} ---")
        satirlar = []
        for d in degerler:
            par = dict(MGMT_VARSAYILAN); par[par_ad] = float(d)
            nav = _walk_managed(wbd, closes, dates, feats, par)
            tot, sh, dd = _ozet(nav)
            # Yıl bazında tutarlılık
            yil_get = {}
            for y, grp in nav.groupby(nav.index.year):
                if len(grp) > 5:
                    yil_get[y] = (grp.iloc[-1] / grp.iloc[0] - 1) * 100
            satirlar.append((d, tot, sh, dd, yil_get))
            isaret = " *" if d == MGMT_VARSAYILAN[par_ad] else ""
            print(f"  {par_ad}={d:<5} getiri %{tot:+8.1f} | Sharpe {sh:5.2f} | "
                  f"MaxDD %{dd:6.1f} | maliyet %{nav.attrs['maliyet']*100:.1f}{isaret}")
        # En iyi ile varsayılan arasındaki fark anlamlı mı?
        eniyi = max(satirlar, key=lambda x: x[1])
        varsay = next(x for x in satirlar if x[0] == MGMT_VARSAYILAN[par_ad])
        fark = eniyi[1] - varsay[1]
        # Yıl bazında kaç yılda en iyi, varsayılanı geçiyor?
        yillar = sorted(set(varsay[4]) & set(eniyi[4]))
        ustun = sum(1 for y in yillar if eniyi[4][y] > varsay[4][y])
        print(f"  → en iyi {eniyi[0]} (varsayılan {MGMT_VARSAYILAN[par_ad]}), "
              f"fark %{fark:+.1f}, {ustun}/{len(yillar)} yılda üstün")
        if fark > 5 and ustun >= max(3, len(yillar) - 1):
            print(f"  ÖNERİ: {par_ad} {MGMT_VARSAYILAN[par_ad]} → {eniyi[0]} "
                  f"(fark büyük VE yıllar arası tutarlı)")
            # Öneriyi kaydet — UYGULAMAZ. Editör onayıyla uygulanır (bkz.
            # bist_analyzer._guard_write / PortfolioDoctor.kural_ayarla).
            try:
                from bist_analyzer import _PMDB as _DB
                _DB.execute("""CREATE TABLE IF NOT EXISTS pm_param_oneri (
                        parametre TEXT PRIMARY KEY, mevcut REAL, onerilen REAL,
                        kanit TEXT, tarih TEXT, kaynak TEXT)""")
                _DB.execute("INSERT OR REPLACE INTO pm_param_oneri "
                            "(parametre, mevcut, onerilen, kanit, tarih, kaynak) "
                            "VALUES (?,?,?,?,?,?)",
                            (par_ad, float(MGMT_VARSAYILAN[par_ad]), float(eniyi[0]),
                             f"{years}y tarihsel tarama: getiri farkı %{fark:+.0f}, "
                             f"{ustun}/{len(yillar)} yılda üstün, Sharpe "
                             f"{varsay[2]:.2f}→{eniyi[2]:.2f}",
                             datetime.now().strftime("%Y-%m-%d"), "pipeline_sweep"))
                print("  (öneri kaydedildi — uygulamak editör onayına bağlı)")
            except Exception as _e:
                print(f"  (öneri kaydedilemedi: {_e})")
        else:
            print(f"  karar: DEĞİŞTİRME — fark gürültü sınırında "
                  f"veya yıllar arası tutarsız")
        sonuclar[par_ad] = satirlar
        print()

    print(f"Tamamlandı ({time.time()-t0:.0f}s)")
    return sonuclar



def validate(years=8):
    """ÖRNEKLEM-DIŞI DOĞRULAMA — taramanın en büyük zaafını kapatır.

    sweep() en iyi parametreyi TEST ETTİĞİ verinin üstünde bulur (in-sample).
    Bu, "sistem kendi buldu" iddiasını çürütür: yeterince parametre denenirse
    her veri setinde bir şey "en iyi" çıkar.

    Burada dönem ikiye bölünür:
      EĞİTİM (ilk %60)  → en iyi parametre BURADA seçilir
      TEST   (son %40)  → seçilen değer BURADA sınanır, seçime hiç katılmaz

    Eğitimde seçilen değer testte de varsayılanı geçiyorsa bulgu gerçektir.
    Geçmiyorsa gürültüye uyum yapılmış demektir ve parametre DEĞİŞTİRİLMEMELİDİR.

    Uzun pencere (varsayılan 8 yıl) bilinçli: 2018 TL krizi ve 2020 Covid
    çöküşünü içerir, yani stop/fren gibi sigortaların işe yaradığı dönemler
    örnekleme girer. 5 yıllık pencere sadece boğaydı.
    """
    t0 = time.time()
    print(f"=== Örneklem-Dışı Doğrulama | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})

    xu = yf.download("XU100.IS", period=f"{years+1}y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(xu.columns, pd.MultiIndex):
        xu.columns = xu.columns.get_level_values(0)
    xu = xu["Close"].dropna()
    start = xu.index[-1] - pd.DateOffset(years=years)
    dates = list(xu.index[xu.index >= start])
    kesme = int(len(dates) * 0.60)
    egitim, test = dates[:kesme], dates[kesme:]
    print(f"Panel: {len(feats)} hisse | EĞİTİM {egitim[0]:%Y-%m} → {egitim[-1]:%Y-%m} "
          f"| TEST {test[0]:%Y-%m} → {test[-1]:%Y-%m} ({time.time()-t0:.0f}s)")

    rebal_dates = pd.date_range(dates[0], dates[-1], freq="MS")
    rebal_dates = [dates[pd.Index(dates).searchsorted(d)] for d in rebal_dates if d <= dates[-1]]
    wbd = {}
    for d in rebal_dates:
        w = _pick_at(d, feats, ret_panel)
        if w:
            wbd[d] = w
    w_eg = {d: w for d, w in wbd.items() if d <= egitim[-1]}
    w_te = {d: w for d, w in wbd.items() if d > egitim[-1]}
    print(f"Aylık portföy: eğitim {len(w_eg)}, test {len(w_te)}")


    taranacak = {
        "trailing_pct": [10, 15, 20, 25, 100],
        "dusus_freni":  [10, 15, 20, 100],
        "max_pozisyon": [20, 25, 30, 40],
        "takas_farki":  [0, 6, 12, 20],
    }
    print(f"{'parametre':<14} {'varsayılan':>10} {'eğitimde en iyi':>16} "
          f"{'TESTTE varsayılan':>18} {'TESTTE seçilen':>15}   karar")
    print("-" * 96)
    onaylanan = {}
    for par_ad, degerler in taranacak.items():
        # 1) EĞİTİMDE seç
        en_iyi, en_iyi_get = None, -1e18
        for d in degerler:
            par = dict(MGMT_VARSAYILAN); par[par_ad] = float(d)
            g = _ozet(_walk_managed(w_eg, closes, egitim, feats, par))[0]
            if g > en_iyi_get:
                en_iyi, en_iyi_get = d, g
        # 2) TESTTE sına (seçime katılmayan dönem)
        par_v = dict(MGMT_VARSAYILAN)
        par_s = dict(MGMT_VARSAYILAN); par_s[par_ad] = float(en_iyi)
        t_v = _ozet(_walk_managed(w_te, closes, test, feats, par_v))[0]
        t_s = _ozet(_walk_managed(w_te, closes, test, feats, par_s))[0]
        gecti = (en_iyi != MGMT_VARSAYILAN[par_ad]) and (t_s > t_v)
        karar = "DOĞRULANDI" if gecti else (
            "varsayılan zaten en iyi" if en_iyi == MGMT_VARSAYILAN[par_ad]
            else "REDDEDİLDİ (testte tutmadı)")
        print(f"{par_ad:<14} {MGMT_VARSAYILAN[par_ad]:>10} {en_iyi:>16} "
              f"{t_v:>17.1f}% {t_s:>14.1f}%   {karar}")
        if gecti:
            onaylanan[par_ad] = en_iyi
        # Öneri kaydını örneklem-dışı kanıtla güncelle / doğrulanmayanı SİL.
        # sweep() in-sample seçer; nihai söz bu doğrulamanındır.
        try:
            from bist_analyzer import _PMDB as _DB
            _DB.execute("""CREATE TABLE IF NOT EXISTS pm_param_oneri (
                    parametre TEXT PRIMARY KEY, mevcut REAL, onerilen REAL,
                    kanit TEXT, tarih TEXT, kaynak TEXT)""")
            if gecti:
                _DB.execute("INSERT OR REPLACE INTO pm_param_oneri "
                            "(parametre, mevcut, onerilen, kanit, tarih, kaynak) "
                            "VALUES (?,?,?,?,?,?)",
                            (par_ad, float(MGMT_VARSAYILAN[par_ad]), float(en_iyi),
                             f"ÖRNEKLEM-DIŞI DOĞRULANDI ({years}y): eğitim "
                             f"{egitim[0]:%Y-%m}–{egitim[-1]:%Y-%m} seçti, test "
                             f"{test[0]:%Y-%m}–{test[-1]:%Y-%m} onayladı "
                             f"(%{t_v:.0f} → %{t_s:.0f})",
                             datetime.now().strftime("%Y-%m-%d"), "validate_oos"))
            else:
                _DB.execute("DELETE FROM pm_param_oneri WHERE parametre=? AND kaynak!=?",
                            (par_ad, "validate_oos"))
        except Exception:
            pass

    print()
    if onaylanan:
        print("Örneklem-dışı DOĞRULANAN öneriler:", onaylanan)
    else:
        print("Hiçbir parametre değişikliği örneklem-dışı doğrulanmadı —")
        print("sweep'in bulduğu farklar bu pencerede gürültüye uyum çıktı.")
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")
    return onaylanan


def walkforward(years=8, katman=5):
    """WALK-FORWARD DOĞRULAMA — validate'in tek bölmesinin yerine çok katman.

    NEDEN: validate() pencereyi bir kez %60/%40 böler. 8 yıllık pencerede bu,
    2018 TL krizi ile 2020 Covid çöküşünün İKİSİNİ DE eğitim yarısına atar;
    yani parametreler yalnız boğa piyasasında sınanmış olur. Tek bölme ayrıca
    şansa açıktır: bir parametre bir kez tesadüfen tutabilir.

    NASIL: genişleyen pencere. Dönem katman+1 dilime bölünür ve her adımda
    geçmişin tamamıyla öğrenip BİR SONRAKİ dilimi sınarız:

        Eğitim [0]        → Test [1]
        Eğitim [0,1]      → Test [2]
        Eğitim [0,1,2]    → Test [3]
        ...

    Her test dilimi seçime hiç katılmaz ve kronolojik olarak eğitimden
    SONRA gelir — ileriye bakma (look-ahead) yok. Sonuç tek bir "tuttu"
    değil, "5 katmanın 4'ünde tuttu" gibi bir sayım olur.

    Kabul ölçütü validate'ten SIKI: bir değişiklik ancak katmanların
    çoğunda VE anlamlı bir farkla (>%5) kazanırsa önerilir. validate'in
    çıplak `t_s > t_v` eşiği %0.1'lik farkı bile onaylıyordu.
    """
    t0 = time.time()
    print(f"=== Walk-Forward Doğrulama | {years} yıl | {katman} katman | "
          f"{datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})

    xu = yf.download("XU100.IS", period=f"{years+1}y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(xu.columns, pd.MultiIndex):
        xu.columns = xu.columns.get_level_values(0)
    xu = xu["Close"].dropna()
    dates = list(xu.index[xu.index >= xu.index[-1] - pd.DateOffset(years=years)])
    rebal_dates = pd.date_range(dates[0], dates[-1], freq="MS")
    rebal_dates = [dates[pd.Index(dates).searchsorted(d)] for d in rebal_dates if d <= dates[-1]]
    wbd = {}
    for d in rebal_dates:
        w = _pick_at(d, feats, ret_panel)
        if w:
            wbd[d] = w
    print(f"Panel {len(feats)} hisse | {len(wbd)} rebalans | "
          f"{dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d} ({time.time()-t0:.0f}s)\n")

    # Dilim sınırları: katman+1 eşit parça
    n = len(dates)
    sinir = [int(n * i / (katman + 1)) for i in range(katman + 2)]
    print("Katmanlar:")
    for k in range(katman):
        eg_son, te_bas, te_son = sinir[k + 1], sinir[k + 1], sinir[k + 2] - 1
        print(f"  {k+1}. eğitim {dates[0]:%Y-%m}→{dates[eg_son-1]:%Y-%m} | "
              f"TEST {dates[te_bas]:%Y-%m}→{dates[te_son]:%Y-%m}")
    print()

    # ---- BÖLÜM 1: DÖNEM DÖNEM YÖNETİMLİ vs YÖNETİMSİZ ----
    #
    # Parametre taramasından ÖNCE gelen ve daha temel olan soru: yönetim
    # katmanı işe yarıyor mu? Bugüne dek tek bir toplam pencerede ölçüldü ve
    # cevap iki kez ters döndü (önce _walk'un sabit ağırlığı yüzünden, sonra
    # takas_farki yanlış ayarlı olduğu için). Tek sayı yerine dönem dönem
    # bakmak, sonucun bir döneme mi yaslandığını gösterir.
    #
    # NAV'lar TEK bir sürekli koşudan alınır, dönem başında sıfırlanmaz —
    # gerçek portföy de her çeyrekte sıfırdan kurulmuyor. Böylece her dönemin
    # başındaki giriş maliyeti yapay olarak tekrarlanmıyor.
    nav_u_tam = _walk(wbd, closes, dates)
    nav_m_tam = _walk_managed(wbd, closes, dates, feats, dict(MGMT_VARSAYILAN))

    def _dilim(nav, bas, son):
        s = nav.iloc[bas:son + 1]
        getiri = (s.iloc[-1] / s.iloc[0] - 1) * 100
        dd = ((s / s.cummax()) - 1).min() * 100
        return getiri, dd

    print("BÖLÜM 1 — Dönem dönem YÖNETİMLİ vs YÖNETİMSİZ "
          f"(varsayılan parametreler: takas_farki={MGMT_VARSAYILAN['takas_farki']:g})")
    print(f"{'dönem':<22} {'yönetimsiz':>22} {'yönetilen':>22}   kazanan")
    print("-" * 96)
    y_kazandi = 0
    for k in range(katman + 1):
        b, s = sinir[k], sinir[k + 1] - 1
        gu, ddu = _dilim(nav_u_tam, b, s)
        gm, ddm = _dilim(nav_m_tam, b, s)
        kazanan = "YÖNETİLEN" if gm > gu else "yönetimsiz"
        if gm > gu:
            y_kazandi += 1
        print(f"{dates[b]:%Y-%m}→{dates[s]:%Y-%m}      "
              f"%{gu:>9.1f} (dd %{ddu:>6.1f})  %{gm:>9.1f} (dd %{ddm:>6.1f})   {kazanan}")
    print("-" * 96)
    print(f"Yönetim {y_kazandi}/{katman + 1} dönemde getiriyi geçti.\n")

    # ---- BÖLÜM 2: PARAMETRE SEÇİMLERİ ÖRNEKLEM-DIŞI TUTUYOR MU ----
    taranacak = {
        "trailing_pct": [10, 15, 20, 25, 100],
        "dusus_freni":  [10, 15, 20, 100],
        "max_pozisyon": [20, 25, 30, 40],
        "takas_farki":  [0, 3, 6, 9, 12, 20],
    }
    print("BÖLÜM 2 — Parametre seçimleri örneklem-dışı tutuyor mu\n")

    ozet = {}
    for par_ad, degerler in taranacak.items():
        print(f"--- {par_ad} (varsayılan {MGMT_VARSAYILAN[par_ad]:g}) ---")
        kazanma, secimler = 0, []
        for k in range(katman):
            eg_dates = dates[:sinir[k + 1]]
            te_dates = dates[sinir[k + 1]:sinir[k + 2]]
            w_eg = {d: w for d, w in wbd.items() if d <= eg_dates[-1]}
            w_te = {d: w for d, w in wbd.items() if eg_dates[-1] < d <= te_dates[-1]}
            if not w_eg or not w_te:
                print(f"  {k+1}. katman atlandı (yeterli rebalans yok)")
                continue
            # 1) EĞİTİMDE seç
            en_iyi, en_iyi_get = None, -1e18
            for d in degerler:
                par = dict(MGMT_VARSAYILAN); par[par_ad] = float(d)
                g = _ozet(_walk_managed(w_eg, closes, eg_dates, feats, par))[0]
                if g > en_iyi_get:
                    en_iyi, en_iyi_get = d, g
            # 2) TESTTE sına
            par_v = dict(MGMT_VARSAYILAN)
            par_s = dict(MGMT_VARSAYILAN); par_s[par_ad] = float(en_iyi)
            t_v = _ozet(_walk_managed(w_te, closes, te_dates, feats, par_v))[0]
            t_s = _ozet(_walk_managed(w_te, closes, te_dates, feats, par_s))[0]
            # Anlamlı fark şartı: %5'ten küçük iyileşme gürültü sayılır
            fark = (t_s - t_v) / max(abs(t_v), 1e-9) * 100
            tuttu = (en_iyi != MGMT_VARSAYILAN[par_ad]) and fark > 5.0
            if tuttu:
                kazanma += 1
            secimler.append(en_iyi)
            print(f"  {k+1}. katman: eğitim seçti {en_iyi:>4g} | test varsayılan "
                  f"%{t_v:>9.1f} → seçilen %{t_s:>9.1f} ({fark:+6.1f}%) "
                  f"{'✓' if tuttu else '·'}")
        gecerli = len(secimler)
        ozet[par_ad] = (kazanma, gecerli, secimler)
        if gecerli:
            print(f"  → {kazanma}/{gecerli} katmanda varsayılanı anlamlı biçimde geçti\n")
    print("=" * 96)
    print(f"{'parametre':<14} {'katman':>8}  {'eğitimin seçimleri':<28} karar")
    print("-" * 96)
    for par_ad, (kazanma, gecerli, secimler) in ozet.items():
        # SIKI KABUL: katmanların çoğunda kazanmalı (yarıdan fazlası)
        onay = gecerli >= 3 and kazanma > gecerli / 2
        karar = ("DEĞİŞTİR → " + str(max(set(secimler), key=secimler.count))) if onay else "DEĞİŞTİRME"
        print(f"{par_ad:<14} {kazanma:>4}/{gecerli:<3}  "
              f"{str(secimler):<28} {karar}")
    print("=" * 96)
    print("Not: bir katmanda kazanmak şans olabilir; çoğunlukta kazanmak zordur.")
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")


def karsilastir(years=5):
    """YÖNETİLEN vs YÖNETİMSİZ — sunumun merkez tablosu, tek komutla tekrarlanır.

    Bu karşılaştırma 2026-08-03'e kadar YANLIŞTI ve yanlışlığı sonucu tersine
    çeviriyordu. Sebep: _walk_managed ağırlıkları fiyatla sürüklerken _walk
    onları rebalanslar arasında SABİT tutuyordu, yani yönetimsiz referans her
    gün sıfır maliyetle hedef ağırlığa dönen, gerçekleştirilemez bir portföydü.
    Trend piyasada bu her gün kazananı budamak demek — referansı yapay olarak
    zayıflatıyordu. İki taraf aynı fiziğe tabi olunca yönetimin katkısı
    getiride ARTI değil EKSİ çıkıyor; kazanç düşüş korumasında.

    Donmuş panel kullanır (bkz. _panel_hazirla), yani sayı koşudan koşuya
    değişmez — akademik sunumda savunulabilirliğin ön şartı.
    """
    t0 = time.time()
    print(f"=== Yönetim Karşılaştırması | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})

    xu = yf.download("XU100.IS", period=f"{years+1}y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(xu.columns, pd.MultiIndex):
        xu.columns = xu.columns.get_level_values(0)
    xu = xu["Close"].dropna()
    start = xu.index[-1] - pd.DateOffset(years=years)
    dates = list(xu.index[xu.index >= start])
    rebal_dates = pd.date_range(dates[0], dates[-1], freq="MS")
    rebal_dates = [dates[pd.Index(dates).searchsorted(d)] for d in rebal_dates if d <= dates[-1]]
    wbd = {}
    for d in rebal_dates:
        w = _pick_at(d, feats, ret_panel)
        if w:
            wbd[d] = w
    print(f"Panel {len(feats)} hisse | {len(wbd)}/{len(rebal_dates)} rebalans "
          f"| {dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d} ({time.time()-t0:.0f}s)\n")
    print(f"Yönetim parametreleri: {MGMT_VARSAYILAN}\n")

    xu_n = xu.loc[dates] / xu.loc[dates].iloc[0]
    nav_u = _walk(wbd, closes, dates)
    nav_m = _walk_managed(wbd, closes, dates, feats, dict(MGMT_VARSAYILAN))

    print("=" * 118)
    su = _stats(nav_u, xu_n, "YÖNETİMSİZ (aylık reb.)")
    sm = _stats(nav_m, xu_n, "YÖNETİLEN (günlük)")
    sx = _stats(xu_n, xu_n, "XU100")
    print("=" * 118)
    print(f"\nYÖNETİMİN KATKISI: reel CAGR {sm['reel_yillik']-su['reel_yillik']:+.1f} puan | "
          f"MaxDD {sm['maxdd']-su['maxdd']:+.1f} puan | Sharpe {sm['sharpe']-su['sharpe']:+.2f}")
    print(f"İşlem maliyeti (yönetilen, kümülatif): %{nav_m.attrs.get('maliyet', 0)*100:.1f}")
    if sm["reel_yillik"] < su["reel_yillik"]:
        print("OKUMA: Yönetim getiri ÜRETMİYOR, risk AZALTIYOR — düşüşü kısmanın "
              "bedeli reel getiriden ödeniyor. Sunumda böyle sunulmalı.")
    else:
        print("OKUMA: Yönetim hem getiriyi hem riski iyileştiriyor.")

    out = pd.DataFrame({"yonetimsiz": nav_u, "yonetilen": nav_m, "xu100": xu_n})
    out.to_csv("pipeline_yonetim.csv")
    print(f"\npipeline_yonetim.csv yazıldı | toplam süre {time.time()-t0:.0f}s")


if __name__ == "__main__":
    yrs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    _cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if _cmd == "run":
        run(yrs)
    elif _cmd == "sweep":
        sweep(yrs)
    elif _cmd == "validate":
        validate(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif _cmd == "dondur":
        # Paneli yeniden indirir; EVREN dosyasına DOKUNMAZ. Farklı yıl
        # pencereleri aynı evrenden beslenmeli (bkz. _panel_hazirla).
        _panel_hazirla(yrs, yenile=True)
        print("Panel donduruldu (evren değişmedi).")
    elif _cmd == "dondur-evren":
        # Ölçüm dönemini SIFIRDAN başlatır: evreni canlı TV'den yeniden çeker.
        # Bundan sonra ESKİ SONUÇLAR KARŞILAŞTIRILAMAZ; mevcut panellerin
        # hepsi bu yeni evrenden yeniden kurulmalıdır.
        print("DİKKAT: evren yeniden çekiliyor — eski sonuçlar karşılaştırılamaz "
              "hâle gelecek ve TÜM panel önbellekleri silinip yeniden kurulmalı.")
        _panel_hazirla(yrs, evren_yenile=True)
        print("Evren + panel donduruldu.")
    elif _cmd == "karsilastir":
        karsilastir(yrs)
    elif _cmd == "walkforward":
        walkforward(int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                    int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    else:
        print(__doc__)
