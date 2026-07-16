"""
weight_calibration — Skor bileşeni ağırlıklarını tarihsel veriden kalibre etme aracı.

compute_bist_score() içindeki katsayıların (Teknik/Sentiment/Prim/Değerleme)
ampirik dayanağını üretir. Yöntem: her bileşen skorunu geçmişin her noktasında
point-in-time hesapla, ileri getirilerle (1/3/7/14/30 gün) Spearman korelasyonunu
(Information Coefficient, Grinold-Kahn) ölç, ağırlıkları IC ile orantılı öner.

Kullanım:
    python weight_calibration.py build-tech        # Teknik skor paneli (~20k gözlem, ~5 dk)
    python weight_calibration.py build-sentiment   # GDELT haber tonu paneli (~8 dk, rate-limit'li)
    python weight_calibration.py analyze           # IC analizi + önerilen ağırlıklar

Çıktılar (app klasörüne yazılır):
    calibration_tech.csv       ticker, date, tech_score, fwd_1..fwd_30
    calibration_sentiment.csv  ticker, date, tone (GDELT günlük ton)

Not: GDELT tonu uygulamanın kendi sentiment skoruyla birebir aynı şey değil —
tarihsel proxy. RSS geçmişe dönük çekilemediği için tek ücretsiz tarihsel
haber duygu kaynağı GDELT'tir (2017+, TR medya dahil).
"""

import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from bist_analyzer import TechnicalEngine, BIST_STOCKS, _tech_style_for
from news_engine import TICKER_TO_TR

TECH_CSV = "calibration_tech.csv"
SENT_CSV = "calibration_sentiment.csv"

# İleri getiri ufukları (işlem günü — Sinyal Takip'teki takvim günlerine yakın)
HORIZONS = [1, 3, 7, 14, 30]

# Panel örnekleme sıklığı (işlem günü) — 5 = haftalık
SAMPLE_STEP = 5

# Teknik skor için gereken minimum geçmiş bar sayısı (SMA200 + 52 hafta)
MIN_HISTORY = 260
WINDOW = 320


def _all_tickers() -> list:
    return sorted({t for cats in BIST_STOCKS.values() for t in cats})


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance MultiIndex kolonlarını düzleştir."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def build_tech():
    tickers = _all_tickers()
    rows = []
    t0 = time.time()

    for n, ticker in enumerate(tickers, 1):
        try:
            df = yf.download(f"{ticker}.IS", period="6y", interval="1d",
                             auto_adjust=True, progress=False)
            df = _flatten_columns(df)
            if df is None or df.empty or len(df) < MIN_HISTORY + 40:
                print(f"[{n}/{len(tickers)}] {ticker}: yetersiz veri ({0 if df is None else len(df)} bar) — atlandı")
                continue

            close = df["Close"]
            max_h = max(HORIZONS)
            count = 0

            for i in range(MIN_HISTORY, len(df) - max_h, SAMPLE_STEP):
                window = df.iloc[max(0, i - WINDOW):i + 1]
                tr = TechnicalEngine.compute(window)  # skor: dengeli (orijinal)
                score = tr.score
                score_momentum = TechnicalEngine._compute_score(tr, "momentum")
                # Hibrit: uygulamanın gerçek stil seçimi (BIST-30 → dengeli, diğer → momentum)
                score_hybrid = (score if _tech_style_for(ticker, "BIST") == "dengeli"
                                else score_momentum)

                p0 = float(close.iloc[i])
                if p0 <= 0:
                    continue

                row = {
                    "ticker": ticker,
                    "date": df.index[i].strftime("%Y-%m-%d"),
                    "tech_score": score,
                    "tech_score_momentum": score_momentum,
                    "tech_score_hybrid": score_hybrid,
                    "is_bist30": int(ticker in BIST_STOCKS["BIST 30"]),
                    # Alt sinyaller — hangi gösterge gerçekten öngörücü, ayrıştırma için
                    "rsi": tr.rsi,
                    "golden_cross": int(tr.golden_cross),
                    "sma_gap_pct": tr.sma_gap_pct,
                    "above_sma50": int(tr.price_above_sma50),
                    "above_sma200": int(tr.price_above_sma200),
                    "macd_bullish": int(tr.macd_bullish),
                    "macd_hist_pct": round(tr.macd_histogram / p0 * 100, 4) if p0 else 0.0,
                    "bb_position": tr.bb_position,
                    "week52_position": tr.week52_position,
                    "stoch_k": tr.stoch_k,
                    "volume_breakout": int(tr.volume_breakout),
                    "obv_up": 1 if tr.obv_trend == "yukari" else (-1 if tr.obv_trend == "asagi" else 0),
                    "adx": tr.adx,
                }
                for h in HORIZONS:
                    row[f"fwd_{h}"] = round((float(close.iloc[i + h]) / p0 - 1) * 100, 4)
                rows.append(row)
                count += 1

            print(f"[{n}/{len(tickers)}] {ticker}: {count} gözlem ({time.time()-t0:.0f}s)")

        except Exception as exc:
            print(f"[{n}/{len(tickers)}] {ticker}: HATA — {exc}")

    panel = pd.DataFrame(rows)
    panel.to_csv(TECH_CSV, index=False)
    print(f"\nToplam {len(panel)} gözlem -> {TECH_CSV} ({time.time()-t0:.0f}s)")


def _gdelt_tone(query: str, start: str, end: str, max_retry: int = 4) -> list:
    """GDELT timelinetone — günlük ton serisi. 429'da bekleyip yeniden dener."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "timelinetone",
        "startdatetime": start,
        "enddatetime": end,
        "format": "json",
    }
    for attempt in range(max_retry):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            return data.get("timeline", [{}])[0].get("data", [])
        except (requests.RequestException, ValueError):
            time.sleep(10 * (attempt + 1))
    return []


def build_sentiment():
    tickers = _all_tickers()
    rows = []
    t0 = time.time()

    for n, ticker in enumerate(tickers, 1):
        company = TICKER_TO_TR.get(ticker)
        if not company:
            print(f"[{n}/{len(tickers)}] {ticker}: TR şirket adı yok — atlandı")
            continue

        # Şirket adı + borsa bağlamı: alakasız haberleri (uçuş, market kampanyası) azaltır
        query = f'"{company}"'
        points = _gdelt_tone(query, "20200701000000", datetime.now().strftime("%Y%m%d%H%M%S"))

        for p in points:
            d = p.get("date", "")[:8]
            if len(d) == 8:
                rows.append({
                    "ticker": ticker,
                    "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}",
                    "tone": round(float(p.get("value", 0.0)), 4),
                })

        print(f"[{n}/{len(tickers)}] {ticker} ({company}): {len(points)} gün ({time.time()-t0:.0f}s)")
        time.sleep(5.5)  # GDELT rate limit: 5 sn/istek

    panel = pd.DataFrame(rows)
    panel.to_csv(SENT_CSV, index=False)
    print(f"\nToplam {len(panel)} satır -> {SENT_CSV} ({time.time()-t0:.0f}s)")


def analyze():
    tech = pd.read_csv(TECH_CSV)
    print(f"Teknik panel: {len(tech)} gözlem, {tech['ticker'].nunique()} hisse, "
          f"{tech['date'].min()} → {tech['date'].max()}\n")

    fwd_cols = [f"fwd_{h}" for h in HORIZONS]

    # 1) TEKNİK SKOR IC'leri
    # Doğru yöntem (Grinold-Kahn): her tarih için hisseler-arası (kesitsel)
    # Spearman korelasyonu, sonra tarihlerin ortalaması + t-istatistiği.
    # Pooled korelasyon piyasa betasıyla kirlenir (BIST'te enflasyon dönemi!).
    print("=" * 60)
    print("TEKNİK SKOR — Kesitsel IC (tarih bazlı, Grinold-Kahn)")
    print("=" * 60)
    tech_ic = {}
    for c in fwd_cols:
        daily = (tech.groupby("date")
                     .apply(lambda g: g["tech_score"].corr(g[c], method="spearman"),
                            include_groups=False)
                     .dropna())
        mean_ic = daily.mean()
        t_stat  = mean_ic / (daily.std() / np.sqrt(len(daily))) if len(daily) > 2 else 0.0
        tech_ic[c] = mean_ic
        sig = "***" if abs(t_stat) > 2.6 else ("*" if abs(t_stat) > 2.0 else "")
        print(f"  {c:>8}: IC = {mean_ic:+.4f}  (t={t_stat:+.2f}) {sig}")

    # 2) Desil analizi — piyasadan arındırılmış (tarih bazında demean edilmiş) getiri
    print("\nDesil analizi (7g ileri getiri, piyasa-göreli %):")
    tech["fwd_7_rel"] = tech["fwd_7"] - tech.groupby("date")["fwd_7"].transform("mean")
    tech["decile"] = pd.qcut(tech["tech_score"], 10, labels=False, duplicates="drop")
    dec = tech.groupby("decile")["fwd_7_rel"].agg(["mean", "count"])
    for d, r in dec.iterrows():
        bar = "#" * int(abs(r["mean"]) * 40)
        sign = "+" if r["mean"] >= 0 else "-"
        print(f"  D{int(d)+1:>2}: {r['mean']:+.3f}%  (n={int(r['count'])})  {sign}{bar}")

    # 3) SENTIMENT IC'leri (GDELT paneli varsa)
    sent_ic = {}
    try:
        sent = pd.read_csv(SENT_CSV)
        # Son 7 günlük ortalama ton (uygulamanın haber lookback'iyle uyumlu)
        sent["date"] = pd.to_datetime(sent["date"])
        sent = sent.sort_values(["ticker", "date"])
        sent["tone_7d"] = (sent.groupby("ticker")["tone"]
                               .transform(lambda s: s.rolling(7, min_periods=3).mean()))
        sent["date"] = sent["date"].dt.strftime("%Y-%m-%d")

        merged = tech.merge(sent[["ticker", "date", "tone_7d"]],
                            on=["ticker", "date"], how="inner").dropna(subset=["tone_7d"])
        print("\n" + "=" * 60)
        print(f"HABER TONU (GDELT, 7g ort.) — Kesitsel IC  [eşleşen gözlem: {len(merged)}]")
        print("=" * 60)
        for c in fwd_cols:
            daily = (merged.groupby("date")
                           .apply(lambda g: g["tone_7d"].corr(g[c], method="spearman")
                                  if len(g) >= 8 else np.nan,
                                  include_groups=False)
                           .dropna())
            if len(daily) < 10:
                print(f"  {c:>8}: yetersiz kesit")
                continue
            mean_ic = daily.mean()
            t_stat  = mean_ic / (daily.std() / np.sqrt(len(daily)))
            sent_ic[c] = mean_ic
            sig = "***" if abs(t_stat) > 2.6 else ("*" if abs(t_stat) > 2.0 else "")
            print(f"  {c:>8}: IC = {mean_ic:+.4f}  (t={t_stat:+.2f}) {sig}")
    except FileNotFoundError:
        print("\n(Sentiment paneli yok — önce 'build-sentiment' çalıştırın)")

    # 4) ÖNERİLEN AĞIRLIKLAR
    if sent_ic:
        print("\n" + "=" * 60)
        print("ÖNERİLEN AĞIRLIK ORANI (Teknik vs Sentiment)")
        print("=" * 60)
        # Uygulamanın sinyal ufkuna uygun ortalama IC (7/14/30 gün)
        mid = ["fwd_7", "fwd_14", "fwd_30"]
        t_ic = np.mean([max(tech_ic[c], 0) for c in mid])
        s_ic = np.mean([max(sent_ic[c], 0) for c in mid])
        total = t_ic + s_ic
        if total > 0:
            print(f"  Ortalama IC (7-30g): Teknik={t_ic:.4f}  Sentiment={s_ic:.4f}")
            print(f"  IC-orantılı bölüşüm: Teknik %{t_ic/total*100:.0f}  Sentiment %{s_ic/total*100:.0f}")
            print(f"  (Mevcut uygulama: Teknik %50 / Sentiment %50 — 35/35'in normalize hali)")
        else:
            print("  Her iki IC de <= 0 — bu horizonda öngörü gücü tespit edilemedi.")


def analyze_stops():
    """ATR stop × hedef katsayı ızgara simülasyonu.

    Giriş olayları: panel tarihlerinde momentum skorunun kesitsel en üst
    %20'sine giren hisseler (sistemin gerçek seçim mantığına en yakın vekil).
    Her olay için k×ATR stop / m×ATR hedef / H gün maksimum tutma kuralıyla
    ileriye yürütülür: önce stop mu kesilir, hedef mi vurulur, süre mi dolar?
    Not: aynı gün ikisi de gerçekleşirse STOP sayılır (muhafazakâr varsayım).
    """
    tech = pd.read_csv(TECH_CSV)
    score_col = "tech_score_momentum" if "tech_score_momentum" in tech.columns else "tech_score"

    # Giriş olayları: tarih bazında üst %20
    tech["q80"] = tech.groupby("date")[score_col].transform(lambda s: s.quantile(0.80))
    events = tech[tech[score_col] >= tech["q80"]][["ticker", "date"]]
    print(f"Giriş olayı: {len(events)} (üst %20 momentum, {events['ticker'].nunique()} hisse)")

    # OHLC + ATR hazırlığı (hisse başına bir kez)
    data = {}
    for n, tk in enumerate(sorted(events["ticker"].unique()), 1):
        try:
            df = yf.download(f"{tk}.IS", period="6y", interval="1d",
                             auto_adjust=True, progress=False)
            df = _flatten_columns(df)
            if df is None or len(df) < 300:
                continue
            close, high, low = df["Close"], df["High"], df["Low"]
            prev_c = close.shift(1)
            tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=14).mean()
            data[tk] = {
                "dates": df.index.strftime("%Y-%m-%d").tolist(),
                "close": close.to_numpy(float), "high": high.to_numpy(float),
                "low": low.to_numpy(float), "atr": atr.to_numpy(float),
                "ix": {d: i for i, d in enumerate(df.index.strftime("%Y-%m-%d"))},
            }
        except Exception:
            continue
        if n % 20 == 0:
            print(f"  fiyat verisi {n} hisse...")

    HORIZON_HOLD = {"kisa": 15, "orta": 60, "uzun": 180}
    K_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 99.0]   # 99 = stopsuz
    M_GRID = [2.5, 4.0, 6.0, 99.0]             # 99 = hedefsiz

    for hz, H in HORIZON_HOLD.items():
        results = []
        for k in K_GRID:
            for m in M_GRID:
                rets, days_held, wins = [], [], 0
                for tk, d in events.itertuples(index=False):
                    dd = data.get(tk)
                    if dd is None:
                        continue
                    i = dd["ix"].get(d)
                    if i is None or i + 2 >= len(dd["close"]) or np.isnan(dd["atr"][i]):
                        continue
                    entry = dd["close"][i]
                    if entry <= 0:
                        continue
                    stop   = entry - k * dd["atr"][i]
                    target = entry + m * dd["atr"][i]
                    end    = min(i + H, len(dd["close"]) - 1)
                    lows, highs = dd["low"][i+1:end+1], dd["high"][i+1:end+1]
                    hit_s = np.argmax(lows <= stop)   if (lows <= stop).any()   else 10**6
                    hit_t = np.argmax(highs >= target) if (highs >= target).any() else 10**6
                    if hit_s <= hit_t and hit_s < 10**6:      # stop önce (eşitlikte stop)
                        ret, held = (stop / entry - 1) * 100, hit_s + 1
                    elif hit_t < hit_s:                        # hedef önce
                        ret, held = (target / entry - 1) * 100, hit_t + 1
                    else:                                      # süre doldu
                        ret, held = (dd["close"][end] / entry - 1) * 100, end - i
                    rets.append(ret)
                    days_held.append(held)
                    wins += ret > 0
                if len(rets) < 100:
                    continue
                results.append({
                    "stop_k": k, "hedef_m": m, "n": len(rets),
                    "ort_getiri": round(float(np.mean(rets)), 2),
                    "medyan": round(float(np.median(rets)), 2),
                    "kazanma_%": round(wins / len(rets) * 100, 1),
                    "ort_gun": round(float(np.mean(days_held)), 1),
                    # Günlük verimlilik: sermaye kilitli kaldığı süreye göre getiri
                    "getiri_per_gun": round(float(np.mean(rets)) / max(float(np.mean(days_held)), 1), 3),
                })
        rdf = pd.DataFrame(results).sort_values("ort_getiri", ascending=False)
        cur_k = {"kisa": 1.5, "orta": 2.5, "uzun": 3.5}[hz]
        cur_m = {"kisa": 2.5, "orta": 4.0, "uzun": 6.0}[hz]
        print(f"\n{'='*70}\n{hz.upper()} vade (max {H} işlem günü) — mevcut: stop {cur_k}×ATR / hedef {cur_m}×ATR\n{'='*70}")
        print(rdf.head(8).to_string(index=False))
        cur = rdf[(rdf['stop_k'] == cur_k) & (rdf['hedef_m'] == cur_m)]
        if not cur.empty:
            print(f"--- mevcut ayarın sırası: {rdf.index.get_loc(cur.index[0]) + 1}/{len(rdf)} "
                  f"(ort %{cur['ort_getiri'].iloc[0]})")


def _xsic_rank(df, col, fwd):
    """Kesitsel Spearman IC — scipy'siz (rank + pearson); ort. IC ve t döner."""
    def _one(g):
        if len(g) < 8:
            return np.nan
        return g[col].rank().corr(g[fwd].rank())
    daily = df.groupby("date").apply(_one, include_groups=False).dropna()
    if len(daily) < 10:
        return np.nan, 0.0
    m = float(daily.mean())
    t = m / (float(daily.std()) / np.sqrt(len(daily)))
    return m, t


def stability():
    """Aylık parametre kararlılık koşusu (Roadmap-D).

    Taze panel kurar, kilit parametreleri ölçer, Turso'daki calib_history
    tablosuna yazar ve tüm geçmiş koşuların karşılaştırma tablosunu basar.
    Ekim sunumundaki 'parametreler zamanla kararlı mı?' kanıtı buradan çıkar.
    """
    build_tech()
    tech = pd.read_csv(TECH_CSV)
    bugun = datetime.now().strftime("%Y-%m-%d")

    metrikler = {}
    for ad, col, fwd in [
        ("ic_momentum_14g", "tech_score_momentum", "fwd_14"),
        ("ic_momentum_30g", "tech_score_momentum", "fwd_30"),
        ("ic_52hafta_30g",  "week52_position",     "fwd_30"),
        ("ic_eski_skor_30g", "tech_score",         "fwd_30"),
    ]:
        m, t = _xsic_rank(tech, col, fwd)
        metrikler[ad] = round(m, 4)
        metrikler[ad + "_t"] = round(t, 2)
    # Rejim güncelliği: sadece son 6 ayın kesitsel IC'si
    son6 = tech[tech["date"] >= (datetime.now() - timedelta(days=183)).strftime("%Y-%m-%d")]
    m6, t6 = _xsic_rank(son6, "tech_score_momentum", "fwd_30")
    metrikler["ic_momentum_30g_son6ay"] = round(m6, 4) if not np.isnan(m6) else None
    metrikler["gozlem_sayisi"] = len(tech)

    from bist_analyzer import _PMDB
    _PMDB.execute("""
        CREATE TABLE IF NOT EXISTS calib_history (
            run_date TEXT, metric TEXT, value REAL,
            PRIMARY KEY (run_date, metric)
        )""")
    _PMDB.execute_batch([
        ("INSERT OR REPLACE INTO calib_history (run_date, metric, value) VALUES (?,?,?)",
         (bugun, k, v)) for k, v in metrikler.items() if v is not None])
    print(f"\ncalib_history'ye yazıldı ({bugun}): {len(metrikler)} metrik")

    # Kararlılık tablosu — tüm koşular yan yana
    rows = _PMDB.execute("SELECT run_date, metric, value FROM calib_history ORDER BY run_date")["rows"]
    if rows:
        hist = pd.DataFrame(rows).pivot(index="metric", columns="run_date", values="value")
        print("\nPARAMETRE KARARLILIK TABLOSU (koşular yan yana):")
        print(hist.to_string())
        print("\nOkuma: ic_momentum_* satırları pozitif ve t>2 kaldıkça sistemin"
              "\ntemel varsayımı geçerli demektir; işaret dönerse yeniden kalibrasyon şart.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build-tech":
        build_tech()
    elif cmd == "build-sentiment":
        build_sentiment()
    elif cmd == "analyze":
        analyze()
    elif cmd == "analyze-stops":
        analyze_stops()
    elif cmd == "stability":
        stability()
    else:
        print(__doc__)
