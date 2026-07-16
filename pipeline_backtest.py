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

Kullanım: python pipeline_backtest.py run [yil_sayisi=5]
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


if __name__ == "__main__":
    yrs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(yrs)
    else:
        print(__doc__)
