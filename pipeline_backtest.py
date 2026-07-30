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

Kullanım: python pipeline_backtest.py run   [yil_sayisi=5]   # tam simülasyon
          python pipeline_backtest.py sweep [yil_sayisi=5]   # yönetim kuralı taraması
"""

import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

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


def _download_panel(tickers, years):
    """Kapanış/hacim panelleri (chunked bulk)."""
    closes, vols = {}, {}
    period = f"{years + 2}y"
    syms = [t + ".IS" for t in tickers]
    for i in range(0, len(syms), 100):
        try:
            bulk = yf.download(syms[i:i+100], period=period, interval="1d",
                               auto_adjust=True, progress=False, group_by="ticker")
            for s in syms[i:i+100]:
                t = s[:-3]
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
        print(f"  veri: {min(i+100, len(syms))}/{len(syms)} sembol, {len(closes)} kullanılabilir")
    return closes, vols


def _precompute(closes, vols):
    """Hisse başına 1 kez: momentum skor serisi, ATR%, TL hacim, getiri paneli."""
    feats = {}
    for t, cl in closes.items():
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
    """Aylık hedef ağırlıklar → günlük NAV (devir maliyetli)."""
    nav, cur_w = [1.0], {}
    rebal = sorted(weights_by_date)
    ri = 0
    for i in range(1, len(dates)):
        d0, d1 = dates[i-1], dates[i]
        # günlük getiri
        r = 0.0
        for t, w in cur_w.items():
            cl = closes.get(t)
            if cl is None:
                continue
            try:
                p0 = cl.asof(d0); p1 = cl.asof(d1)
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    r += w * (p1 / p0 - 1)
            except Exception:
                continue
        v = nav[-1] * (1 + r)
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
    dr = nav.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    print(f"{label:<24} nominal %{tot:>8.0f} | CAGR %{cagr:>5.1f} | ENAG-reel %{reel:>7.1f} "
          f"(yıllık %{reel_cagr:>+5.1f}) | vsXU100 {tot-xu_tot:>+7.0f}p | MaxDD %{dd:>5.1f} | Sharpe {sharpe:.2f}")
    return {"label": label, "nominal": tot, "cagr": cagr, "reel": reel,
            "reel_yillik": reel_cagr, "maxdd": dd, "sharpe": sharpe}


def run(years=5):
    t0 = time.time()
    print(f"=== Tam-Pipeline Simülasyonu | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    universe = get_scan_universe()
    print(f"Evren: {len(universe)} hisse (bugünün likit listesi — survivorship notu raporda)")

    closes, vols = _download_panel(universe, years)
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
    all_t = list(feats.keys())
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

MGMT_VARSAYILAN = {
    "trailing_pct": 15.0,   # zirveden geri verme toleransı
    "dusus_freni":  15.0,   # portföy düşüşünde yeni alım freni
    "max_pozisyon": 30.0,   # tek pozisyon tavanı (oran olarak 0.30)
    "skor_cikis":   43.0,   # skor bu seviyenin altına düşerse çık
    "takas_farki":  12.0,   # aday, tutulandan bu kadar iyiyse değiştir
}


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


def _ozet(nav):
    """Toplam getiri, Sharpe, MaxDD."""
    tot = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    dg = nav.pct_change().dropna()
    sharpe = (dg.mean() / dg.std() * np.sqrt(252)) if dg.std() > 0 else 0.0
    dd = ((nav / nav.cummax()) - 1).min() * 100
    return tot, float(sharpe), float(dd)


def sweep(years=5):
    """Yönetim parametrelerini tek tek süpür — her biri için toplam getiri,
    Sharpe, MaxDD ve YIL BAZINDA tutarlılık."""
    t0 = time.time()
    print(f"=== Yönetim Kuralı Taraması | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    universe = get_scan_universe()
    closes, vols = _download_panel(universe, years)
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

if __name__ == "__main__":
    yrs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    _cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if _cmd == "run":
        run(yrs)
    elif _cmd == "sweep":
        sweep(yrs)
    else:
        print(__doc__)
