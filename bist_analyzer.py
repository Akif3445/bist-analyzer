"""
BIST Smart Investment Assistant
"""

import json
import logging
import os
import pickle  # Yalnızca eski BLOB verilerini okumak için (geriye dönük uyumluluk)
import sqlite3
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv olmadan da çalışır (bulut ortamında st.secrets kullanılır)

try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_OK = True
except ImportError:
    OPTION_MENU_OK = False

try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False

warnings.filterwarnings("ignore")

# Dosya yolları
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_IS_CLOUD = (os.environ.get("STREAMLIT_SHARING_MODE")
             or os.environ.get("STREAMLIT_SERVER_HEADLESS")
             or os.path.exists("/mount/src"))
_DATA_DIR = "/tmp" if (_IS_CLOUD and os.path.isdir("/tmp")) else _APP_DIR
DB_PATH   = os.path.join(_DATA_DIR, "bist_cache.db")
LOG_PATH  = os.path.join(_DATA_DIR, "bist_analyzer.log")

# Loglama
_log_handlers = [logging.StreamHandler()]
try:
    _log_handlers.insert(0, logging.FileHandler(LOG_PATH, encoding="utf-8"))
except (PermissionError, OSError):
    pass  # Bulut ortamında dosya yazılamayabilir — sadece stdout'a yaz

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_log_handlers,
)
log = logging.getLogger("bist")


# Secrets
def _get_secret(key: str, default: str = "") -> str:
    """st.secrets veya os.getenv'den anahtar okur."""
    try:
        import streamlit as _st
        if hasattr(_st, "secrets") and key in _st.secrets:
            return str(_st.secrets[key]).strip()
    except Exception:
        pass
    return (os.getenv(key) or default).strip()


# ── Yazma koruması ────────────────────────────────────────────────────────
# Halka açık demo (Streamlit Cloud) Turso yazma kimlik bilgileriyle koşar;
# auth yoktur. Anonim ziyaretçinin paylaşılan veriyi (portföyler, NAV, ENAG
# reel-getiri tablosu) bozmasını engellemek için: APP_EDIT_KEY secret'ı
# TANIMLIYSA tüm yazma işlemleri o anahtarla kilit açmayı gerektirir.
# Anahtar tanımlı DEĞİLSE (yerel geliştirme) davranış eskisi gibi serbesttir.
def _writes_locked() -> bool:
    """Yazma işlemleri kilitli mi? (secret tanımlı ve oturum açılmamışsa evet)"""
    if not _get_secret("APP_EDIT_KEY"):
        return False   # yerel/anahtarsız → serbest
    try:
        return not st.session_state.get("_edit_unlocked", False)
    except Exception:
        return False


def _guard_write() -> bool:
    """Yazma butonlarından önce çağrılır. Kilitliyse kullanıcıyı uyarır ve
    False döner (çağıran işlem yapmaz). Serbestse True."""
    if not _writes_locked():
        return True
    st.warning("🔒 Bu demo salt-görüntüleme modunda. Değişiklik yapmak için "
               "kenar çubuğundan editör anahtarını girin.")
    return False


# News Engine
try:
    from news_engine import analyze_news, render_news_panel, NewsAnalysisResult
    NEWS_ENGINE_AVAILABLE = True
except ImportError:
    NEWS_ENGINE_AVAILABLE = False

# Phase 2 modules
try:
    from faz2_modules import (
        DataCache,
        KAPScraper,
        TargetPriceFetcher,
        WatchlistManager,
        SchedulerService,
        AlertManager,
        enhanced_news_analysis,
        render_kap_disclosures,
        render_target_prices,
    )
    FAZ2_AVAILABLE = True
except ImportError:
    FAZ2_AVAILABLE = False

# Configuration

def _yf_symbol(ticker: str, market: str = "BIST") -> str:
    """yfinance sembol formatı."""
    t = ticker.upper().strip()
    if market == "US":
        return t.replace(".IS", "")
    else:
        return t if t.endswith(".IS") else t + ".IS"


def _default_index(market: str = "BIST") -> str:
    """Piyasa ana endeksi."""
    return "^GSPC" if market == "US" else "XU100.IS"


ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")

SECTOR_FK_AVERAGES = {
    "Banka":      5.2,
    "Holdingler": 8.1,
    "Teknoloji":  22.5,
    "Perakende":  14.3,
    "Enerji":     9.8,
    "Insaat":     7.6,
    "Sanayi":     11.2,
    "Genel":      10.0,
}

# Kalibrasyon (weight_calibration.py, Temmuz 2026): teknik momentum skoru
# IC +0.045 (t=2.8) ile anlamlı; GDELT haber tonu proxy'si her ufukta zayıf (t<1).
# IC oranı tekniğe %64-80 pay veriyor — temkinli adım olarak 45/25 seçildi
# (haber motorumuz KAP + TR kaynaklarıyla GDELT'ten iyi olabilir).
# Sinyal Takip verisi biriktikçe yeniden kalibre edilecek.
WEIGHTS = {
    "teknik":    45,   # Teknik analiz (momentum stili — kalibrasyonla doğrulandı)
    "sentiment": 25,   # Haber sentiment (kalibrasyonda öngörü gücü zayıf çıktı)
    "prim":      20,   # Fiyat primi / hedef fiyat (tarihsel test edilemedi)
    "deger":     10,   # Değerleme (F/K, PD/DD) (tarihsel test edilemedi)
}

# Sinyal kayıtlarına yazılan skor sistemi versiyonu — Sinyal Takip'te v1/v2
# isabet karşılaştırması için. v1: kontrarian teknik + 35/35/20/10 ağırlıklar
# (2026-07-13 öncesi). v2: momentum teknik + 45/25/20/10 (kalibrasyon sonrası).
SCORING_VERSION = "v2"

# BISTScore JSON serialization

def _score_to_json(score: "BISTScore") -> str:
    """BISTScore -> JSON string."""
    df_json = None
    if score.stock.df is not None and not score.stock.df.empty:
        try:
            df_json = score.stock.df.to_json(orient="split", date_format="iso")
        except Exception:
            pass

    info_safe: dict = {}
    for k, v in (score.stock.info or {}).items():
        try:
            json.dumps({k: v})
            info_safe[k] = v
        except (TypeError, ValueError):
            pass

    tc_dict = None
    tc = getattr(score, "_target_consensus", None)
    if tc is not None:
        try:
            tc_dict = asdict(tc)
        except Exception:
            pass

    data = {
        "ticker": score.ticker,
        "total_score": score.total_score,
        "signal": score.signal,
        "signal_color": score.signal_color,
        "teknik_score": score.teknik_score,
        "sentiment_score": score.sentiment_score,
        "prim_score": score.prim_score,
        "deger_score": score.deger_score,
        "technical": asdict(score.technical),
        "sentiment": {
            "score": score.sentiment.score,
            "positive_count": score.sentiment.positive_count,
            "negative_count": score.sentiment.negative_count,
            "headlines": list(score.sentiment.headlines or []),
            "raw_score": score.sentiment.raw_score,
        },
        "valuation": asdict(score.valuation),
        "stock": {
            "ticker": score.stock.ticker,
            "pe_ratio": score.stock.pe_ratio,
            "pb_ratio": score.stock.pb_ratio,
            "roe": score.stock.roe,
            "sector": score.stock.sector,
            "current_price": float(score.stock.current_price),
            "market_cap": score.stock.market_cap,
            "error": score.stock.error,
            "df_json": df_json,
            "info": info_safe,
        },
        "kap_disclosures": list(getattr(score, "_kap_disclosures", []) or []),
        "material_events": list(getattr(score, "_material_events", []) or []),
        "target_consensus": tc_dict,
        "news_result": _news_result_to_dict(getattr(score, "_news_result", None)),
    }
    return json.dumps(data, ensure_ascii=False, default=str)


def _news_result_to_dict(nr) -> Optional[dict]:
    """NewsAnalysisResult -> dict."""
    if nr is None:
        return None
    try:
        return {
            "score":            nr.score,
            "positive_count":   nr.positive_count,
            "negative_count":   nr.negative_count,
            "neutral_count":    nr.neutral_count,
            "total_news":       nr.total_news,
            "filtered_count":   nr.filtered_count,
            "headlines":        list(nr.headlines or []),
            "news_items":       list(nr.news_items or []),
            "kap_disclosures":  list(nr.kap_disclosures or []),
            "material_events":  list(nr.material_events or []),
            "data_sources":     list(nr.data_sources or []),
            "data_quality":     nr.data_quality,
            "language":         nr.language,
            "warning":          nr.warning,
            "sentiment_engine": nr.sentiment_engine,
        }
    except Exception:
        return None


def _news_result_from_dict(d: Optional[dict]):
    """dict -> NewsAnalysisResult."""
    if not d:
        return None
    if not NEWS_ENGINE_AVAILABLE:
        return None
    try:
        from news_engine import NewsAnalysisResult
        return NewsAnalysisResult(
            score=d.get("score", 50.0),
            positive_count=d.get("positive_count", 0),
            negative_count=d.get("negative_count", 0),
            neutral_count=d.get("neutral_count", 0),
            total_news=d.get("total_news", 0),
            filtered_count=d.get("filtered_count", 0),
            headlines=d.get("headlines", []),
            news_items=d.get("news_items", []),
            kap_disclosures=d.get("kap_disclosures", []),
            material_events=d.get("material_events", []),
            data_sources=d.get("data_sources", []),
            data_quality=d.get("data_quality", "notr"),
            language=d.get("language", "BOTH"),
            warning=d.get("warning", ""),
            sentiment_engine=d.get("sentiment_engine", "kelime_sayma"),
        )
    except Exception:
        return None


def _score_from_json(raw) -> Optional["BISTScore"]:
    """JSON/bytes -> BISTScore. Eski pickle formatini da destekler."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            return pickle.loads(raw)
        except Exception:
            return None

    try:
        data = json.loads(raw)

        tech_data = data.get("technical", {})
        technical = TechnicalResult(**{
            k: v for k, v in tech_data.items()
            if k in TechnicalResult.__dataclass_fields__
        })

        s_data = data.get("sentiment", {})
        sentiment = SentimentResult(
            score=s_data.get("score", 50.0),
            positive_count=s_data.get("positive_count", 0),
            negative_count=s_data.get("negative_count", 0),
            headlines=s_data.get("headlines", []),
            raw_score=s_data.get("raw_score", 0.0),
        )

        val_data = data.get("valuation", {})
        valuation = ValuationResult(**{
            k: v for k, v in val_data.items()
            if k in ValuationResult.__dataclass_fields__
        })

        s = data.get("stock", {})
        stock = StockData(
            ticker=s.get("ticker", data.get("ticker", "")),
            info=s.get("info", {}),
            pe_ratio=s.get("pe_ratio"),
            pb_ratio=s.get("pb_ratio"),
            roe=s.get("roe"),
            sector=s.get("sector", "Genel"),
            current_price=float(s.get("current_price", 0.0)),
            market_cap=s.get("market_cap"),
            error=s.get("error"),
        )
        df_json = s.get("df_json")
        if df_json:
            try:
                stock.df = pd.read_json(StringIO(df_json), orient="split")
            except Exception:
                stock.df = pd.DataFrame()

        score = BISTScore(
            ticker=data["ticker"],
            total_score=data["total_score"],
            signal=data["signal"],
            signal_color=data["signal_color"],
            teknik_score=data["teknik_score"],
            sentiment_score=data["sentiment_score"],
            prim_score=data["prim_score"],
            deger_score=data["deger_score"],
            technical=technical,
            sentiment=sentiment,
            valuation=valuation,
            stock=stock,
        )

        score._kap_disclosures = data.get("kap_disclosures", [])
        score._material_events = data.get("material_events", [])
        score._news_result = _news_result_from_dict(data.get("news_result"))

        tc_data = data.get("target_consensus")
        if tc_data and FAZ2_AVAILABLE:
            try:
                from faz2_modules import TargetPriceConsensus
                score._target_consensus = TargetPriceConsensus(**tc_data)
            except Exception:
                score._target_consensus = None
        else:
            score._target_consensus = None

        return score
    except Exception:
        return None


# Analiz gecmisi DB

class AnalysisHistoryDB:
    """Analiz sonuclarini SQLite'a kaydeder ve sorgular."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._init_table()

    def _init_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    ticker TEXT PRIMARY KEY,
                    last_analyzed TEXT,
                    total_score REAL,
                    signal TEXT,
                    price REAL,
                    teknik_score REAL,
                    sentiment_score REAL,
                    prim_score REAL,
                    deger_score REAL,
                    rsi REAL,
                    golden_cross INTEGER,
                    macd_bullish INTEGER,
                    analysis_count INTEGER DEFAULT 1
                )
            """)
            try:
                conn.execute("ALTER TABLE analysis_history ADD COLUMN full_data BLOB")
            except sqlite3.OperationalError:
                pass

            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    message TEXT,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    is_read INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    signal TEXT,
                    signal_date TEXT,
                    signal_price REAL,
                    check_date TEXT,
                    check_price REAL,
                    status TEXT DEFAULT 'Bekliyor'
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT UNIQUE,
                    buy_price REAL,
                    qty INTEGER,
                    add_date TEXT
                )
            """)

            # Backtest tablolari
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    ticker TEXT,
                    entry_date TEXT,
                    entry_price REAL,
                    exit_date TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    return_pct REAL,
                    hold_days INTEGER,
                    entry_score REAL,
                    stop_loss REAL,
                    take_profit REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    ticker TEXT,
                    date TEXT,
                    score REAL,
                    rsi REAL,
                    price REAL,
                    UNIQUE(run_id, ticker, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    run_date TEXT,
                    ticker TEXT,
                    period TEXT,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    win_rate REAL,
                    avg_return_pct REAL,
                    total_return_pct REAL,
                    max_drawdown_pct REAL,
                    best_trade_pct REAL,
                    worst_trade_pct REAL,
                    avg_hold_days REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_backtest_cache (
                    ticker TEXT,
                    date TEXT,
                    sentiment TEXT,
                    count INTEGER,
                    fetched_at TEXT,
                    PRIMARY KEY (ticker, date)
                )
            """)

            # Performans indexleri
            for _idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_history_ticker ON analysis_history(ticker)",
                "CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker)",
                "CREATE INDEX IF NOT EXISTS idx_alerts_unread ON alerts(is_read)",
                "CREATE INDEX IF NOT EXISTS idx_accuracy_ticker ON accuracy_log(ticker)",
                "CREATE INDEX IF NOT EXISTS idx_backtest_runid ON backtest_trades(run_id)",
                "CREATE INDEX IF NOT EXISTS idx_backtest_scores_runid ON backtest_daily(run_id)",
                "CREATE INDEX IF NOT EXISTS idx_summary_runid ON backtest_summary(run_id)",
                "CREATE INDEX IF NOT EXISTS idx_news_cache_ticker ON news_backtest_cache(ticker)",
            ]:
                try:
                    conn.execute(_idx_sql)
                except sqlite3.OperationalError:
                    pass  # Tablo henüz oluşturulmamış olabilir — sorun değil
            conn.commit()

    def add_alert(self, ticker: str, message: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO alerts (ticker, message) VALUES (?, ?)", (ticker, message))
                conn.commit()
        except sqlite3.OperationalError:
            pass

    def get_unread_alerts(self) -> list:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT id, ticker, message, created_at FROM alerts WHERE is_read = 0 ORDER BY created_at DESC")
                return [{"id": r[0], "ticker": r[1], "message": r[2], "time": r[3]} for r in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []

    def mark_alerts_read(self, alert_ids: list = None) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                if alert_ids:
                    placeholders = ",".join("?" * len(alert_ids))
                    conn.execute(f"UPDATE alerts SET is_read = 1 WHERE id IN ({placeholders})", alert_ids)
                else:
                    conn.execute("UPDATE alerts SET is_read = 1 WHERE is_read = 0")
                conn.commit()
        except sqlite3.OperationalError:
            pass

    def save(self, score: "BISTScore") -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        t   = score.technical
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT analysis_count FROM analysis_history WHERE ticker = ?",
                (score.ticker,)
            ).fetchone()
            count = (existing[0] + 1) if existing else 1
            score_data = _score_to_json(score)
            conn.execute("""
                INSERT OR REPLACE INTO analysis_history
                    (ticker, last_analyzed, total_score, signal, price,
                     teknik_score, sentiment_score, prim_score, deger_score,
                     rsi, golden_cross, macd_bullish, analysis_count, full_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                score.ticker, now, score.total_score, score.signal,
                score.stock.current_price,
                score.teknik_score, score.sentiment_score,
                score.prim_score, score.deger_score,
                t.rsi, int(t.golden_cross), int(t.macd_bullish), count, score_data
            ))
            
            if score.signal in ["AL", "GUCLU AL", "SAT", "GUCLU SAT"]:
                now_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("""
                    INSERT INTO accuracy_log (ticker, signal, signal_date, signal_price, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (score.ticker, score.signal, now_dt, score.stock.current_price, "Bekliyor"))

            conn.commit()

        # Yeni doğrulama sistemine de kaydet
        self.record_signal(score.ticker, score.signal, score.total_score,
                           score.stock.current_price, source="live")

    def load_full(self, ticker: str) -> Optional[tuple[str, "BISTScore"]]:
        """Bir hissenin son analiz tarihini ve tam objesini döner (last_analyzed, obj)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT last_analyzed, full_data FROM analysis_history WHERE ticker = ?",
                (ticker,)
            ).fetchone()
        if row and row[1]:
            obj = _score_from_json(row[1])
            if obj:
                return row[0], obj
        return None

    def load_all(self) -> list:
        """Tüm geçmiş analizleri son analiz tarihine göre sıralı döner."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM analysis_history ORDER BY last_analyzed DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, ticker: str) -> None:
        """Bir hisseyi geçmişten siler."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM analysis_history WHERE ticker = ?", (ticker,))
            conn.commit()

    # Portfolio Methods
    def add_portfolio(self, ticker: str, buy_price: float, qty: int) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO portfolio (ticker, buy_price, qty, add_date)
                VALUES (?, ?, ?, ?)
            """, (ticker.upper(), buy_price, qty, now))
            conn.commit()
            
    def remove_portfolio(self, ticker: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker.upper(),))
            conn.commit()
            
    def get_portfolio(self) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM portfolio ORDER BY add_date DESC").fetchall()
        return [dict(r) for r in rows]

    # Accuracy / Backtest Methods
    def evaluate_accuracy(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM accuracy_log WHERE status = 'Bekliyor'").fetchall()
            
            now_dt = datetime.now()
            pending_tickers = []
            valid_rows = []
            for r in rows:
                if (now_dt - datetime.strptime(r["signal_date"], "%Y-%m-%d %H:%M:%S")).days >= 3:
                    valid_rows.append(r)
                    if r["ticker"] not in pending_tickers:
                        pending_tickers.append(r["ticker"])
                        
            if not pending_tickers: return

            # BUG FIX: BIST hisseleri .IS uzantısı olmadan indirilemiyordu —
            # sinyaller sessizce "Bekliyor"da kalıyordu. US hisseleri ham kalır.
            sym_map = {t: _yf_symbol(t, "US" if t in US_POPULAR_TICKERS else "BIST")
                       for t in pending_tickers}
            symbols = list(sym_map.values())
            try:
                df = yf.download(symbols, period="5d", group_by="ticker", progress=False)
                for r in valid_rows:
                    t = r["ticker"]
                    sym = sym_map.get(t, t)
                    try:
                        d = df[sym] if len(symbols) > 1 else df
                    except (KeyError, TypeError):
                        continue
                    d = d.dropna()
                    if not d.empty and "Close" in d:
                        c_price = float(d["Close"].iloc[-1])
                        s_price = r["signal_price"]
                        if not s_price or s_price == 0:
                            continue
                        pct = ((c_price - s_price) / s_price) * 100
                        
                        status = "Basarisiz"
                        if r["signal"] in ["AL", "GUCLU AL"] and pct >= 2.0:
                            status = "Basarili"
                        elif r["signal"] in ["SAT", "GUCLU SAT"] and pct <= -2.0:
                            status = "Basarili"
                            
                        conn.execute("""
                            UPDATE accuracy_log 
                            SET check_date = ?, check_price = ?, status = ?
                            WHERE id = ?
                        """, (now_dt.strftime("%Y-%m-%d %H:%M:%S"), c_price, status, r["id"]))
                conn.commit()
            except Exception as exc:
                log.warning("Accuracy log güncelleme hatası: %s", exc)

    def get_accuracy_stats(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM accuracy_log WHERE status != 'Bekliyor'").fetchone()[0]
                if total == 0: return 0.0, 0, 0
                successful = conn.execute("SELECT COUNT(*) FROM accuracy_log WHERE status = 'Basarili'").fetchone()[0]
                return round((successful / total) * 100, 1), successful, total
        except sqlite3.OperationalError:
            return 0.0, 0, 0

    # SKOR DOĞRULAMA SİSTEMİ (Score Validation)

    # Takip edilen zaman dilimleri: (gün_sayısı, iş_günü_karşılığı, kolon_prefix)
    VALIDATION_PERIODS = [
        (1,  1,  "1d"),
        (3,  2,  "3d"),
        (7,  5,  "7d"),
        (14, 10, "14d"),
        (30, 21, "30d"),
    ]

    def _init_validation_table(self):
        """accuracy_validation tablosunu oluştur + eksik kolonları ekle."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_validation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    signal TEXT,
                    score REAL,
                    signal_date TEXT,
                    signal_price REAL,
                    source TEXT DEFAULT 'live',
                    check_7d_date TEXT,
                    check_7d_price REAL,
                    return_7d_pct REAL,
                    result_7d TEXT DEFAULT 'Bekliyor',
                    check_30d_date TEXT,
                    check_30d_price REAL,
                    return_30d_pct REAL,
                    result_30d TEXT DEFAULT 'Bekliyor',
                    score_version TEXT DEFAULT 'v1'
                )
            """)
            # Eski tabloya skor versiyonu kolonu ekle (2026-07 kalibrasyonu öncesi = v1)
            try:
                conn.execute("ALTER TABLE accuracy_validation ADD COLUMN score_version TEXT DEFAULT 'v1'")
            except sqlite3.OperationalError:
                pass
            # Eski tabloya yeni kolonları ekle (varsa atla)
            for prefix in ("1d", "3d", "14d"):
                for col_suffix in ("date TEXT", "price REAL", "pct REAL"):
                    parts = col_suffix.split()
                    if len(parts) < 2:
                        continue
                    col_name = f"check_{prefix}_{parts[0]}"
                    try:
                        conn.execute(f"ALTER TABLE accuracy_validation ADD COLUMN {col_name} {parts[1]}")
                    except sqlite3.OperationalError:
                        pass
                try:
                    conn.execute(f"ALTER TABLE accuracy_validation ADD COLUMN return_{prefix}_pct REAL")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute(f"ALTER TABLE accuracy_validation ADD COLUMN result_{prefix} TEXT DEFAULT 'Bekliyor'")
                except sqlite3.OperationalError:
                    pass
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_val_ticker ON accuracy_validation(ticker)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_val_signal ON accuracy_validation(signal)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_val_source ON accuracy_validation(source)")
            except sqlite3.OperationalError:
                pass
            conn.commit()

    def record_signal(self, ticker: str, signal: str, score: float, price: float, source: str = "live"):
        """
    """
        if signal in ("NOTR", "HATA") or price <= 0:
            return
        self._init_validation_table()
        today = datetime.now().strftime("%Y-%m-%d")
        now   = datetime.now().strftime("%Y-%m-%d %H:%M")
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Aynı gün zaten kayıt var mı?
                existing = conn.execute("""
                    SELECT id FROM accuracy_validation
                    WHERE ticker = ? AND signal_date LIKE ? AND source = ?
                """, (ticker, f"{today}%", source)).fetchone()
                if existing:
                    conn.execute("""
                        UPDATE accuracy_validation
                        SET signal = ?, score = ?, signal_price = ?, signal_date = ?, score_version = ?
                        WHERE id = ?
                    """, (signal, round(score, 1), round(price, 2), now, SCORING_VERSION, existing[0]))
                else:
                    conn.execute("""
                        INSERT INTO accuracy_validation
                            (ticker, signal, score, signal_date, signal_price, source, score_version)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (ticker, signal, round(score, 1), now, round(price, 2), source, SCORING_VERSION))
                conn.commit()
        except Exception as exc:
            log.warning("record_signal hatası: %s", exc)

    def check_pending_signals(self):
        """Bekleyen sinyalleri kontrol et: 1g, 3g, 7g, 14g, 30g sonraki fiyatları güncelle."""
        self._init_validation_table()
        now = datetime.now()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # Herhangi bir periyotta hâlâ bekleyen kayıtları çek
                pending = conn.execute("""
                    SELECT * FROM accuracy_validation
                    WHERE result_7d = 'Bekliyor'
                       OR result_30d = 'Bekliyor'
                       OR result_1d = 'Bekliyor'
                       OR result_3d = 'Bekliyor'
                       OR result_14d = 'Bekliyor'
                """).fetchall()
                if not pending:
                    return

                tickers = list({r["ticker"] for r in pending})
                # Her ticker için market belirle (source'a bak)
                _ticker_market = {}
                for r in pending:
                    try:
                        src = r["source"] or "live"
                    except (KeyError, IndexError):
                        src = "live"
                    mkt = "US" if "us" in src.lower() else "BIST"
                    _ticker_market[r["ticker"]] = mkt
                symbols = [_yf_symbol(t, _ticker_market.get(t, "BIST")) for t in tickers]

                try:
                    df = yf.download(symbols, period="2mo", group_by="ticker",
                                     auto_adjust=True, progress=False)
                except Exception:
                    return

                for r in pending:
                    try:
                        sig_date = datetime.strptime(r["signal_date"][:16], "%Y-%m-%d %H:%M")
                    except Exception:
                        continue
                    days_passed = (now - sig_date).days
                    try:
                        _src = r["source"] or "live"
                    except (KeyError, IndexError):
                        _src = "live"
                    _mkt = "US" if "us" in _src.lower() else "BIST"
                    sym = _yf_symbol(r["ticker"], _mkt)

                    try:
                        if len(tickers) > 1:
                            col = df[sym]["Close"].dropna()
                        else:
                            col = df["Close"].dropna()
                    except (KeyError, TypeError):
                        continue

                    if col.empty:
                        continue
                    current_price = float(col.iloc[-1])
                    sig_price = r["signal_price"]
                    if not sig_price or sig_price <= 0:
                        continue

                    # Tüm periyotları kontrol et
                    for cal_days, _biz_days, prefix in self.VALIDATION_PERIODS:
                        result_col = f"result_{prefix}"
                        try:
                            current_result = r[result_col]
                        except (KeyError, IndexError):
                            current_result = "Bekliyor"
                        if current_result == "Bekliyor" and days_passed >= cal_days:
                            ret_pct = round((current_price - sig_price) / sig_price * 100, 2)
                            result = self._evaluate_signal(r["signal"], ret_pct, cal_days)
                            conn.execute(f"""
                                UPDATE accuracy_validation
                                SET check_{prefix}_date = ?, check_{prefix}_price = ?,
                                    return_{prefix}_pct = ?, result_{prefix} = ?
                                WHERE id = ?
                            """, (now.strftime("%Y-%m-%d"), round(current_price, 2),
                                  ret_pct, result, r["id"]))

                conn.commit()
        except Exception as exc:
            log.warning("check_pending_signals hatası: %s", exc)

    @staticmethod
    def _evaluate_signal(signal: str, return_pct: float, days: int = 30) -> str:
        """
        Sinyal başarılı mı? Süreye göre eşik değişir:
        1g: %0.5, 3g: %1.0, 7g: %1.5, 14g: %2.0, 30g: %2.0
        """
        thresholds = {1: 0.5, 3: 1.0, 7: 1.5, 14: 2.0, 30: 2.0}
        thr = thresholds.get(days, 2.0)
        if signal in ("AL", "GUCLU AL"):
            return "Basarili" if return_pct >= thr else "Basarisiz"
        elif signal in ("SAT", "GUCLU SAT"):
            return "Basarili" if return_pct <= -thr else "Basarisiz"
        return "Belirsiz"

    def get_validation_report(self) -> dict:
        """Skor doğrulama raporu için tüm verileri döner."""
        self._init_validation_table()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM accuracy_validation
                    ORDER BY signal_date DESC
                """).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def run_backfill(self, tickers: list, months_back: int = 12,
                     interval_days: int = 7, progress_cb=None) -> int:
        """
    """
        self._init_validation_table()
        count = 0
        total_days_back = months_back * 30
        steps_per_ticker = total_days_back // interval_days
        total = len(tickers) * steps_per_ticker

        for t_idx, ticker in enumerate(tickers):
            try:
                sym = _yf_symbol(ticker, "BIST")
                df = yf.download(sym, period=f"{months_back + 2}mo",
                                 auto_adjust=True, progress=False)
                if df.empty or "Close" not in df.columns:
                    continue

                # MultiIndex fix
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                close = df["Close"].dropna()
                if len(close) < 60:
                    continue

                # Her interval_days günde bir test noktası
                for step_i in range(steps_per_ticker, 0, -1):
                    step = t_idx * steps_per_ticker + (steps_per_ticker - step_i)
                    if progress_cb:
                        progress_cb(ticker, step, total)

                    target_date = datetime.now() - timedelta(days=step_i * interval_days)
                    mask = close.index <= pd.Timestamp(target_date)
                    if mask.sum() < 60:
                        continue
                    hist = close[mask]
                    signal_price = float(hist.iloc[-1])
                    signal_date = hist.index[-1]

                    if signal_price <= 0:
                        continue

                    # Her periyot için gelecek fiyatı bul
                    future_mask = close.index > pd.Timestamp(signal_date)
                    future = close[future_mask]
                    if len(future) < 2:
                        continue

                    # O tarih için teknik skor hesapla
                    hist_df = df[df.index <= pd.Timestamp(signal_date)].tail(250)
                    if len(hist_df) < 50:
                        continue

                    try:
                        tech = TechnicalEngine.compute(hist_df, style=_tech_style_for(ticker, "BIST"))
                        score_val = tech.score
                        signal_str, _ = _score_to_signal(score_val)
                    except Exception:
                        continue

                    if signal_str in ("NOTR", "HATA"):
                        continue

                    # Her periyot için fiyat ve sonuç hesapla
                    period_data = {}
                    for cal_days, biz_days, prefix in self.VALIDATION_PERIODS:
                        if len(future) > biz_days:
                            fp = float(future.iloc[min(biz_days, len(future) - 1)])
                            fd = future.index[min(biz_days, len(future) - 1)]
                            ret = round((fp - signal_price) / signal_price * 100, 2)
                            res = self._evaluate_signal(signal_str, ret, cal_days)
                            period_data[prefix] = (fd.strftime("%Y-%m-%d"), round(fp, 2), ret, res)

                    if not period_data:
                        continue

                    try:
                        sig_date_str = signal_date.strftime("%Y-%m-%d %H:%M")
                        with sqlite3.connect(self.db_path) as conn:
                            # Dinamik SQL oluştur
                            cols = ["ticker", "signal", "score", "signal_date", "signal_price",
                                    "source", "score_version"]
                            vals = [ticker, signal_str, round(score_val, 1),
                                    sig_date_str, round(signal_price, 2), "backfill", SCORING_VERSION]
                            for prefix, (d, p, r, res) in period_data.items():
                                cols += [f"check_{prefix}_date", f"check_{prefix}_price",
                                         f"return_{prefix}_pct", f"result_{prefix}"]
                                vals += [d, p, r, res]

                            placeholders = ",".join("?" * len(vals))
                            col_str = ",".join(cols)
                            conn.execute(f"""
                                INSERT OR IGNORE INTO accuracy_validation ({col_str})
                                VALUES ({placeholders})
                            """, vals)
                            conn.commit()
                        count += 1
                    except Exception as exc:
                        log.debug("Backfill DB insert hatası (%s): %s", ticker, exc)

            except Exception as exc:
                log.warning("Backfill hatası (%s): %s", ticker, exc)
                continue

        return count


_history_db = AnalysisHistoryDB()

# 1. DATA FETCHER

@dataclass
class StockData:
    ticker: str
    df: pd.DataFrame         = field(default_factory=pd.DataFrame)
    info: dict               = field(default_factory=dict)
    pe_ratio: Optional[float]  = None
    pb_ratio: Optional[float]  = None
    roe: Optional[float]       = None
    sector: str              = "Genel"
    current_price: float     = 0.0
    market_cap: Optional[float] = None
    error: Optional[str]     = None


class DataFetcher:
    @staticmethod
    def fetch(ticker: str, period: str = "1y", market: str = "BIST") -> StockData:
        yt = _yf_symbol(ticker, market)

        data = StockData(ticker=ticker.upper())
        try:
            ystock = yf.Ticker(yt)
            df = ystock.history(period=period, auto_adjust=True)

            if df.empty:
                data.error = f"{ticker} icin veri bulunamadi. Hisse kodunu kontrol edin."
                return data

            # Kolon normalizasyonu (yfinance versiyon farklılıkları)
            # yfinance 0.2.40+ bazen MultiIndex veya (Price, Ticker) tuple döndürür
            if isinstance(df.columns, pd.MultiIndex):
                # ("Close", "THYAO.IS") → "Close"
                df.columns = df.columns.get_level_values(0)
            # Lowercase gelirse ("close" → "Close") normalize et
            col_rename = {c: c.strip().title() for c in df.columns
                          if c.strip().title() in ("Open","High","Low","Close","Volume")}
            if col_rename:
                df = df.rename(columns=col_rename)
            # Gerekli kolonlar yoksa hata ver
            if "Close" not in df.columns:
                data.error = f"{ticker} verisi beklenen formatta değil (Close kolonu bulunamadı)."
                return data

            data.df            = df
            data.current_price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
            try:
                info               = ystock.info or {}
            except Exception:
                info               = {}
            data.info          = info
            data.pe_ratio      = info.get("trailingPE") or info.get("forwardPE")
            data.pb_ratio      = info.get("priceToBook")
            data.roe           = info.get("returnOnEquity")
            data.market_cap    = info.get("marketCap")
            data.sector        = DataFetcher._map_sector(info.get("sector", ""))

        except Exception as exc:
            data.error = f"Veri cekme hatasi: {exc}"
            log.error("DataFetcher.fetch(%s): %s\n%s", ticker, exc, traceback.format_exc())

        return data

    @staticmethod
    def _map_sector(yahoo_sector: str) -> str:
        mapping = {
            "Financial Services": "Banka",
            "Technology":         "Teknoloji",
            "Energy":             "Enerji",
            "Consumer Cyclical":  "Perakende",
            "Consumer Defensive": "Perakende",
            "Industrials":        "Sanayi",
            "Real Estate":        "Insaat",
        }
        return mapping.get(yahoo_sector, "Genel")


# 2. NEWS / SENTIMENT

@dataclass
class SentimentResult:
    score: float         = 50.0
    positive_count: int  = 0
    negative_count: int  = 0
    headlines: list      = field(default_factory=list)
    raw_score: float     = 0.0


class NewsScraper:
    POSITIVE_WORDS = [
        "buyume", "kar", "rekor", "artis", "guclu", "olumlu", "basari",
        "ihracat", "yatirim", "temettu", "hedef", "asti", "uzerinde",
        "pozitif", "toparlanma", "yukselis", "genisleme", "anlasma",
    ]
    NEGATIVE_WORDS = [
        "zarar", "dusus", "risk", "kriz", "olumsuz", "baski", "endise",
        "daralma", "kayip", "uyari", "azalis", "negatif", "zayif",
        "sorun", "belirsizlik", "cekilme",
    ]

    @staticmethod
    def fetch_headlines(ticker: str, max_results: int = 15, market: str = "BIST") -> list:
        yt = _yf_symbol(ticker, market)
        url = f"https://finance.yahoo.com/quote/{yt}/news/"
        headers = {"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0 Safari/537.36"}
        headlines = []
        for _attempt in range(2):
            try:
                resp = requests.get(url, headers=headers, timeout=12)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")
                for item in soup.select("h3.Mb\\(5px\\), h3[class*='title']")[:max_results]:
                    text = item.get_text(strip=True)
                    if text:
                        headlines.append(text)
                break  # success
            except requests.exceptions.Timeout:
                continue  # retry once
            except Exception:
                break
        return headlines

    @classmethod
    def analyze(cls, ticker: str) -> SentimentResult:
        headlines = cls.fetch_headlines(ticker)
        result    = SentimentResult(headlines=headlines)
        for headline in headlines:
            lower = headline.lower()
            result.positive_count += sum(1 for w in cls.POSITIVE_WORDS if w in lower)
            result.negative_count += sum(1 for w in cls.NEGATIVE_WORDS if w in lower)
        total = result.positive_count + result.negative_count
        result.score = (
            round((result.positive_count / total) * 100, 1) if total else 50.0
        )
        result.raw_score = result.positive_count - result.negative_count
        return result


# 3. TECHNICAL ANALYSIS ENGINE

@dataclass
class TechnicalResult:
    # Mevcut göstergeler
    sma50: float         = 0.0
    sma200: float        = 0.0
    rsi: float           = 50.0
    macd: float          = 0.0
    macd_signal: float   = 0.0
    macd_histogram: float = 0.0   # MACD - Signal (büyüklük = momentum gücü)
    current_price: float = 0.0
    golden_cross: bool        = False
    price_above_sma50: bool   = False
    price_above_sma200: bool  = False
    rsi_oversold: bool        = False
    rsi_overbought: bool      = False
    macd_bullish: bool        = False
    volume_breakout: bool     = False
    relative_strength: float  = 0.0

    # Yeni göstergeler
    # ADX — Trend gücü (0-100). >25 güçlü trend, <20 yatay piyasa.
    adx: float           = 0.0
    adx_strong: bool     = False   # ADX > 25

    # Bollinger Bantları
    bb_upper: float      = 0.0
    bb_lower: float      = 0.0
    bb_middle: float     = 0.0
    bb_position: float   = 0.5    # 0=alt bant, 1=üst bant. <0.2 ucuz, >0.8 pahalı
    bb_squeeze: bool     = False  # Bantlar daraldı mı? (volatilite patlama öncesi)

    # Stochastic Oscillator (14,3)
    stoch_k: float       = 50.0   # %K
    stoch_d: float       = 50.0   # %D (K'nın 3 günlük ortalaması)
    stoch_oversold: bool  = False  # K < 20
    stoch_overbought: bool = False # K > 80

    # OBV — On-Balance Volume (akıllı para takibi)
    obv_trend: str       = "notr"  # "yukari" | "asagi" | "notr"
    obv_divergence: bool = False   # Fiyat düşerken OBV yükseliyorsa gizli birikim

    # ATR — Average True Range (oynaklık / risk ölçüsü)
    atr: float           = 0.0
    atr_pct: float       = 0.0    # ATR / fiyat * 100 (normalize volatilite %)

    # 52 Haftalık pozisyon
    week52_high: float   = 0.0
    week52_low: float    = 0.0
    week52_position: float = 0.5  # 0=dip, 1=zirve. 0.2-0.5 ideal alım bölgesi

    # SMA mesafe oranları (golden cross gücü için)
    sma_gap_pct: float   = 0.0    # (SMA50 - SMA200) / SMA200 * 100

    score: float         = 50.0


class TechnicalEngine:
    @staticmethod
    def compute(df: pd.DataFrame, style: str = "dengeli") -> TechnicalResult:
        result = TechnicalResult()
        if df is None or df.empty or len(df) < 50:
            return result

        close = df["Close"].dropna()
        if close.empty:
            return result

        high  = df["High"]  if "High"  in df.columns else close
        low   = df["Low"]   if "Low"   in df.columns else close
        vol   = df["Volume"] if "Volume" in df.columns else None

        result.current_price = float(close.iloc[-1])

        # SMA
        sma50_series = close.rolling(50).mean().dropna()
        result.sma50  = float(sma50_series.iloc[-1]) if not sma50_series.empty else result.current_price
        if len(close) >= 200:
            sma200_series = close.rolling(200).mean().dropna()
        else:
            sma200_series = close.rolling(len(close)).mean().dropna()
        result.sma200 = float(sma200_series.iloc[-1]) if not sma200_series.empty else result.current_price
        result.golden_cross        = result.sma50 > result.sma200
        result.price_above_sma50   = result.current_price > result.sma50
        result.price_above_sma200  = result.current_price > result.sma200
        # SMA mesafe oranı — golden cross'un ne kadar güçlü olduğunu ölçer
        if result.sma200 > 0:
            result.sma_gap_pct = round((result.sma50 - result.sma200) / result.sma200 * 100, 2)

        # RSI
        result.rsi            = TechnicalEngine._rsi(close, 14)
        result.rsi_oversold   = result.rsi < 30
        result.rsi_overbought = result.rsi > 70

        # MACD
        try:
            ema12       = close.ewm(span=12, adjust=False).mean()
            ema26       = close.ewm(span=26, adjust=False).mean()
            macd_line   = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            if not macd_line.empty and not signal_line.empty:
                result.macd           = float(macd_line.iloc[-1])
                result.macd_signal    = float(signal_line.iloc[-1])
                result.macd_histogram = float(macd_line.iloc[-1] - signal_line.iloc[-1])
                result.macd_bullish   = result.macd > result.macd_signal
        except Exception as exc:
            log.warning("MACD hesaplama hatası: %s", exc)

        # Hacim Kırılımı
        if vol is not None and len(vol) >= 10 and len(close) >= 2:
            try:
                recent_vol = float(vol.iloc[-1])
                avg_vol_10 = float(vol.rolling(10).mean().iloc[-1])
                if avg_vol_10 > 0 and recent_vol > (avg_vol_10 * 1.5) and float(close.iloc[-1]) > float(close.iloc[-2]):
                    result.volume_breakout = True
            except (IndexError, ValueError):
                pass

        # Bollinger Bantları (20, 2σ)
        if len(close) >= 20:
            try:
                sma20      = close.rolling(20, min_periods=1).mean()
                std20      = close.rolling(20, min_periods=2).std().fillna(0)
                bb_up      = sma20 + 2 * std20
                bb_dn      = sma20 - 2 * std20
                result.bb_upper  = float(bb_up.fillna(result.current_price).iloc[-1])
                result.bb_lower  = float(bb_dn.fillna(result.current_price).iloc[-1])
                result.bb_middle = float(sma20.iloc[-1])
                band_width = result.bb_upper - result.bb_lower
                if band_width > 0:
                    pos = (result.current_price - result.bb_lower) / band_width
                    result.bb_position = round(max(0.0, min(1.0, pos)), 3)
                # Bant daralması: son 5 günün bant genişliği, 20 günlük ortalamanın %60'ından azsa
                bw_series   = (bb_up - bb_dn).fillna(0)
                recent_bw   = float(bw_series.iloc[-5:].mean())
                historic_bw = float(bw_series.iloc[-20:].mean())
                if historic_bw > 0 and recent_bw < historic_bw * 0.6:
                    result.bb_squeeze = True
            except Exception:
                pass

        # Stochastic Oscillator (%K 14, %D 3)
        if len(df) >= 14:
            try:
                low14  = low.rolling(14, min_periods=1).min()
                high14 = high.rolling(14, min_periods=1).max()
                denom  = (high14 - low14).replace(0, np.nan)
                stoch_k_series = ((close - low14) / denom * 100).fillna(50)
                stoch_d_series = stoch_k_series.rolling(3, min_periods=1).mean()
                sk = float(stoch_k_series.iloc[-1])
                sd = float(stoch_d_series.iloc[-1])
                result.stoch_k          = round(max(0.0, min(100.0, sk)), 1)
                result.stoch_d          = round(max(0.0, min(100.0, sd)), 1)
                result.stoch_oversold   = result.stoch_k < 20
                result.stoch_overbought = result.stoch_k > 80
            except Exception:
                pass

        # OBV (On-Balance Volume)
        if vol is not None and len(vol) >= 20:
            obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
            obv_sma10 = obv.rolling(10).mean()
            obv_now   = float(obv.iloc[-1])
            obv_10ago = float(obv.iloc[-10])
            price_now  = float(close.iloc[-1])
            price_10ago = float(close.iloc[-10])

            # Trend yönü: OBV son 10 günde arttı mı azaldı mı?
            if obv_now > float(obv_sma10.iloc[-1]) * 1.01:
                result.obv_trend = "yukari"
            elif obv_now < float(obv_sma10.iloc[-1]) * 0.99:
                result.obv_trend = "asagi"
            else:
                result.obv_trend = "notr"

            # Pozitif ıraksama: fiyat düşerken OBV yükseldiyse gizli birikim
            price_fell = price_now < price_10ago
            obv_rose   = obv_now > obv_10ago
            result.obv_divergence = price_fell and obv_rose

        # ATR (Average True Range, 14 gün)
        if len(df) >= 15:
            try:
                prev_close = close.shift(1)
                tr = pd.concat([
                    high - low,
                    (high - prev_close).abs(),
                    (low  - prev_close).abs(),
                ], axis=1).max(axis=1)
                # min_periods=1: ilk günlerden itibaren NaN üretmez
                atr_val = float(tr.rolling(14, min_periods=1).mean().fillna(0).iloc[-1])
                result.atr     = round(atr_val, 4)
                result.atr_pct = round(atr_val / result.current_price * 100, 2) if result.current_price > 0 else 0.0
            except Exception as exc:
                log.warning("ATR hesaplama hatası: %s | df.columns=%s", exc, list(df.columns))

        # 52 Haftalık Pozisyon
        lookback = min(len(close), 252)
        w52_high = float(close.iloc[-lookback:].max())
        w52_low  = float(close.iloc[-lookback:].min())
        result.week52_high = round(w52_high, 4)
        result.week52_low  = round(w52_low, 4)
        w52_range = w52_high - w52_low
        if w52_range > 0:
            result.week52_position = round(
                (result.current_price - w52_low) / w52_range, 3
            )

        # ADX (Average Directional Index, 14 gün)
        if len(df) >= 28:
            try:
                result.adx = TechnicalEngine._adx(high, low, close, 14)
                result.adx_strong = result.adx > 25
                if result.adx == 0.0:
                    log.warning("ADX=0 döndü | ticker veri uzunluğu=%d | cols=%s", len(df), list(df.columns))
            except Exception as exc:
                log.warning("ADX hesaplama hatası: %s", exc)

        result.score = TechnicalEngine._compute_score(result, style)
        return result

    # Yardımcı Hesaplamalar

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """
    """
        try:
            prev_high  = high.shift(1)
            prev_low   = low.shift(1)
            prev_close = close.shift(1)

            # Ham yön hareketi (negatif değerleri 0 yap)
            dm_pos_raw = (high - prev_high).clip(lower=0)
            dm_neg_raw = (prev_low - low).clip(lower=0)

            # BUG FIX: orijinal seriler üzerinden karşılaştır,
            # birinin maskelenmesi diğerini etkilemez
            dm_pos = dm_pos_raw.where(dm_pos_raw > dm_neg_raw, 0.0)
            dm_neg = dm_neg_raw.where(dm_neg_raw > dm_pos_raw, 0.0)

            # True Range
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)

            # Wilder düzleştirme (EWM alpha = 1/period)
            alpha = 1.0 / period
            atr_s  = tr.ewm(alpha=alpha, adjust=False).mean()
            dip_s  = dm_pos.ewm(alpha=alpha, adjust=False).mean()
            din_s  = dm_neg.ewm(alpha=alpha, adjust=False).mean()

            di_pos = (dip_s / atr_s.replace(0, np.nan)) * 100
            di_neg = (din_s / atr_s.replace(0, np.nan)) * 100

            dx_denom = (di_pos + di_neg).replace(0, np.nan)
            dx  = ((di_pos - di_neg).abs() / dx_denom) * 100
            adx = dx.ewm(alpha=alpha, adjust=False).mean()

            val = float(adx.fillna(0).iloc[-1])
            # Sağlıklı aralık kontrolü
            return round(max(0.0, min(100.0, val)), 1)
        except Exception:
            return 0.0

    @staticmethod
    def _compute_score(r: TechnicalResult, style: str = "dengeli") -> float:
        """0-100 teknik skor.

        style="dengeli"  : Orijinal karma mantık — aşırı satım (düşük RSI/BB/52w)
                           ödüllendirilir. Kalibrasyon (weight_calibration.py, 2026-07)
                           bunun BIST-30 mega-cap'lerde çalıştığını gösterdi.
        style="momentum" : Güç ödüllendirilir — yüksek RSI/BB/52w pozisyonu puan alır.
                           Kalibrasyon geniş BIST evreninde bu rejimi doğruladı
                           (52w IC +0.06-0.08, t>3.5, iki alt dönemde de tutarlı).
        Trend/MACD/hacim blokları iki stilde de aynıdır (yönleri zaten momentum).
        """
        momentum = (style == "momentum")
        score = 0.0

        # 1. TREND GRUBU (max 20)
        # SMA gap oranına göre kademeli golden cross puanı
        # Çok yakın golden cross (gap < %1) → zayıf, olası geri kırılma riski
        if r.golden_cross:
            gap = abs(r.sma_gap_pct)
            if gap >= 5:
                score += 20          # Güçlü golden cross
            elif gap >= 2:
                score += 15
            elif gap >= 0.5:
                score += 10
            else:
                score += 5           # Çok zayıf, kırılma riski var
        else:
            # Death cross — ceza
            gap = abs(r.sma_gap_pct)
            if gap >= 5:
                score -= 10          # Güçlü death cross
            elif gap >= 2:
                score -= 5

        # Fiyat / SMA pozisyonu (her biri bağımsız)
        if r.price_above_sma50:
            score += 6
        else:
            score -= 3
        if r.price_above_sma200:
            score += 6
        else:
            score -= 4

        # 2. MOMENTUM GRUBU — RSI + Stochastic (max 20)
        rsi = r.rsi
        if momentum:
            # Momentum rejimi: güçlü RSI kazandırır (kalibrasyon: RSI IC +0.05, t=3.1)
            if rsi >= 80:
                score += 8           # Çok güçlü ama aşırı ısınmış — temkinli ödül
            elif rsi >= 70:
                score += 18          # Güçlü momentum → en iyi bölge
            elif rsi >= 60:
                score += 14
            elif rsi >= 50:
                score += 9
            elif rsi >= 40:
                score += 4
            elif rsi >= 30:
                score -= 6           # Zayıf hisse → ceza
            else:
                score -= 12          # Çok zayıf — düşen bıçak
        else:
            if rsi < 30:
                score += 20          # Aşırı satım → en iyi alım fırsatı
            elif rsi < 40:
                score += 15          # Ucuz bölge
            elif rsi < 50:
                score += 10          # Nötr-hafif ucuz
            elif rsi < 60:
                score += 7           # Nötr
            elif rsi < 70:
                score += 3           # Hafif pahalı
            elif rsi < 80:
                score -= 8           # Aşırı alım → ceza
            else:
                score -= 15          # Tehlikeli aşırı alım → ağır ceza

        # Stochastic — RSI'yı destekler ya da çelişir
        if momentum:
            if r.stoch_overbought and rsi > 65:
                score += 5           # İki gösterge de güç diyor → güvenilir
            elif r.stoch_oversold and rsi < 40:
                score -= 5           # İki gösterge de zayıflık diyor
            elif r.stoch_overbought:
                score += 2
            elif r.stoch_oversold:
                score -= 2
        elif r.stoch_oversold and rsi < 40:
            score += 5               # İki gösterge de aynı fikirde → güvenilir
        elif r.stoch_overbought and rsi > 65:
            score -= 5               # İki gösterge de aşırı alım diyor
        elif r.stoch_oversold:
            score += 2               # Tek başına hafif sinyal
        elif r.stoch_overbought:
            score -= 2

        # 3. MACD GRUBU (max 15)
        # Histogram büyüklüğü momentum gücünü gösterir
        if r.macd_histogram != 0 and r.current_price > 0:
            hist_pct = abs(r.macd_histogram) / r.current_price * 100
            if r.macd_bullish:
                if hist_pct >= 1.0:
                    score += 15      # Güçlü yukarı momentum
                elif hist_pct >= 0.3:
                    score += 10
                else:
                    score += 5       # Zayıf ama pozitif
            else:
                if hist_pct >= 1.0:
                    score -= 10      # Güçlü aşağı momentum
                elif hist_pct >= 0.3:
                    score -= 5
                else:
                    score -= 2
        else:
            score += (7 if r.macd_bullish else -3)

        # 4. HACİM GRUBU — Volume + OBV (max 15)
        if r.volume_breakout:
            score += 10              # Hacimli kırılım — güçlü onay

        if r.obv_trend == "yukari":
            score += 5               # Akıllı para giriyor
        elif r.obv_trend == "asagi":
            score -= 3               # Akıllı para çıkıyor

        if r.obv_divergence:
            score += 5               # Gizli birikim — güçlü erken sinyal

        # 5. BOLLINGER GRUBU (max 10)
        pos = r.bb_position          # 0=alt bant, 1=üst bant
        if momentum:
            if pos >= 0.85:
                score += 8           # Üst bantta güçlü seyir → momentum onayı
            elif pos >= 0.70:
                score += 6
            elif pos >= 0.50:
                score += 3
            elif pos >= 0.30:
                score += 0
            elif pos >= 0.15:
                score -= 4
            else:
                score -= 8           # Alt bantta sürünüyor → zayıflık
        elif pos <= 0.15:
            score += 10              # Alt bantın çok altında → aşırı satım
        elif pos <= 0.30:
            score += 7
        elif pos <= 0.50:
            score += 4               # Ortanın altı — makul
        elif pos <= 0.70:
            score += 1               # Nötr
        elif pos <= 0.85:
            score -= 4               # Üst banta yakın → dikkat
        else:
            score -= 8               # Üst bantın üstü → aşırı alım cezası

        if r.bb_squeeze:
            score += 3               # Bant daralması → yakında büyük hareket

        # 6. 52 HAFTALIK POZİSYON (max 10)
        w = r.week52_position        # 0=dip, 1=zirve
        if momentum:
            # Kalibrasyondaki en güçlü sinyal: zirveye yakınlık kazandırır
            # (IC +0.06-0.08, iki alt dönemde de t>3.5)
            if w >= 0.80:
                score += 10          # Zirveye yakın → güç devam ediyor
            elif w >= 0.60:
                score += 7
            elif w >= 0.40:
                score += 3
            elif w >= 0.20:
                score -= 2
            else:
                score -= 6           # Yıllık dipte → dipte kalma eğilimi
        elif w <= 0.20:
            score += 10              # Yıllık dibe yakın → potansiyel yüksek
        elif w <= 0.40:
            score += 7
        elif w <= 0.60:
            score += 3               # Yıllık ortasında
        elif w <= 0.80:
            score -= 2
        else:
            score -= 6               # Yıllık zirveye yakın → riskli

        # 7. ADX FİLTRESİ (çarpan)
        # Trend yoksa (yatay piyasa) trend sinyallerinin güvenilirliği düşer.
        # ADX < 20: skorun trend kısmını %70'e indir (trend sinyalleri anlamsız).
        # ADX > 25: trend güçlü, skor olduğu gibi kalır.
        # Aradaki değerler: doğrusal geçiş.
        if r.adx > 0:
            if r.adx < 20:
                adx_multiplier = 0.70
            elif r.adx < 25:
                adx_multiplier = 0.70 + (r.adx - 20) / 5 * 0.30  # 0.70 → 1.0
            else:
                adx_multiplier = 1.0
            # Sadece trend kaynaklı puanları (SMA + Golden Cross kısmı) ölçekle
            # Toplam puan ortalaması üzerinden yaklaşık uygula
            if adx_multiplier < 1.0:
                score = score * (0.85 + 0.15 * adx_multiplier)

        return round(min(100.0, max(0.0, score)), 1)


# 4. VALUATION ENGINE

@dataclass
class ValuationResult:
    target_price: Optional[float] = None
    prim_pct: Optional[float]     = None
    sector_fk: float              = 10.0
    stock_fk: Optional[float]     = None
    stock_pb: Optional[float]     = None
    stock_roe: Optional[float]    = None
    fk_discount: Optional[float]  = None
    pb_discount: Optional[float]  = None
    roe_premium: Optional[float]  = None
    prim_score: float             = 50.0
    deger_score: float            = 50.0

@st.cache_data(ttl=14400, show_spinner=False)
def fetch_tv_analyst_targets() -> dict:
    """TradingView screener'dan tüm BIST analist hedef fiyatları — TEK sorgu, 4 saat cache.

    Dönüş: {ticker: {"target", "count", "high", "low"}}
    En az 3 analist şartı: tek analistli hedefler gürültülü/güvenilmez.
    Paket borsapy ile birlikte geliyor (tradingview-screener).
    """
    try:
        from tradingview_screener import Query
        _n, df = (Query().set_markets("turkey")
                  .select("name", "price_target_average", "price_target_high",
                          "price_target_low", "recommendation_total")
                  .limit(700).get_scanner_data())
        out = {}
        for _, r in df.iterrows():
            t   = str(r.get("name") or "")
            tgt = r.get("price_target_average")
            cnt = r.get("recommendation_total")
            if t and pd.notna(tgt) and pd.notna(cnt) and float(cnt) >= 3:
                out[t] = {
                    "target": float(tgt),
                    "count":  int(cnt),
                    "high":   float(r["price_target_high"]) if pd.notna(r.get("price_target_high")) else None,
                    "low":    float(r["price_target_low"])  if pd.notna(r.get("price_target_low"))  else None,
                }
        log.info("TradingView hedef fiyat: %d hisse için konsensüs alındı", len(out))
        return out
    except Exception as exc:
        log.warning("TradingView hedef fiyat çekilemedi (prim bileşeni ağırlık dağıtır): %s", exc)
        return {}


class ValuationEngine:
    @staticmethod
    def compute(stock: StockData, analyst_target: Optional[float] = None) -> ValuationResult:
        result           = ValuationResult()
        result.stock_fk  = stock.pe_ratio
        result.stock_pb  = stock.pb_ratio
        result.stock_roe = stock.roe
        result.sector_fk = SECTOR_FK_AVERAGES.get(stock.sector, 10.0)

        # 1. Prim Puanı (Analyst Target)
        if analyst_target and analyst_target > 0 and stock.current_price and stock.current_price > 0:
            result.target_price = analyst_target
            result.prim_pct     = round((analyst_target / stock.current_price - 1) * 100, 1)
            result.prim_score   = round(max(0, min(100, (result.prim_pct + 50) / 100 * 100)), 1)
        else:
            result.prim_score = 50.0

        # 2. Gelişmiş Değerleme Puanı (F/K, PD/DD, ROE Kombinasyonu)
        sub_scores = []
        
        # F/K Puanı
        if result.stock_fk is not None and result.sector_fk is not None:
            result.fk_discount = round(float((result.sector_fk - result.stock_fk) / result.sector_fk * 100), 1)
            fk_score = max(0.0, min(100.0, 50.0 + result.fk_discount))
            sub_scores.append(fk_score)
            
        # PD/DD Puanı
        if result.stock_pb is not None:
            pb_score = 100.0 - (float(result.stock_pb) / 1.5 * 50.0)
            result.pb_discount = round(pb_score, 1)
            sub_scores.append(max(0.0, min(100.0, pb_score)))
            
        # ROE Puanı (Özsermaye Karlılığı)
        if result.stock_roe is not None:
            roe_pct = float(result.stock_roe) * 100.0
            result.roe_premium = round(roe_pct, 1)
            roe_score = max(0.0, min(100.0, roe_pct * 4.0)) # %25 roe -> 100 score
            sub_scores.append(roe_score)
            
        if sub_scores:
            result.deger_score = round(sum(sub_scores) / len(sub_scores), 1)
        else:
            result.deger_score = 50.0

        return result


# 5. MAIN SCORING FUNCTION  ◄──── CORE MODULE ────────────

@dataclass
class BISTScore:
    """Aggregates all analysis components into one score object."""
    ticker: str
    total_score: float  = 0.0
    signal: str         = "NOTR"
    signal_color: str   = "#6b6357"

    teknik_score: float    = 0.0
    sentiment_score: float = 0.0
    prim_score: float      = 0.0
    deger_score: float     = 0.0

    technical: TechnicalResult = field(default_factory=TechnicalResult)
    sentiment: SentimentResult = field(default_factory=SentimentResult)
    valuation: ValuationResult = field(default_factory=ValuationResult)
    # FIX: use lambda so StockData() is not called without required 'ticker' arg
    stock: StockData           = field(default_factory=lambda: StockData(ticker=""))


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_benchmark_cached(index_symbol: str) -> pd.DataFrame:
    """Benchmark endeks verisini 15 dakika cache'le — her analiz için tekrar çekmeyi önler."""
    try:
        df = yf.download(index_symbol, period="2mo", progress=False)
        if df.empty:
            log.warning("Benchmark verisi boş döndü: %s", index_symbol)
        return df
    except Exception as exc:
        log.warning("Benchmark indirme hatası (%s): %s", index_symbol, exc)
        return pd.DataFrame()


def _tech_style_for(ticker: str, market: str = "BIST") -> str:
    """Teknik skor stilini piyasaya göre seçer.

    Kalibrasyon bulgusu (weight_calibration.py, Temmuz 2026):
    - BIST momentum rejiminde: saf momentum skoru tüm evrende IC +0.045 (t=2.8),
      iki alt dönemde de pozitif. BIST-30'da eski stille berabere → kayıp yok.
    - Segment-bazlı karma stil DENENDİ ve REDDEDİLDİ: farklı cetvellerle verilen
      puanlar tek sıralamada karşılaştırılınca kesitsel IC bozuluyor (+0.012'ye
      düşüyor). Tüm BIST tek stil kullanmalı.
    - US için kanıt yok → mevcut davranış ("dengeli") korunur.
    """
    return "momentum" if market == "BIST" else "dengeli"


def compute_bist_score(
    ticker: str,
    analyst_target: Optional[float] = None,
    period: str = "1y",
    language: str = "TR",
    market: str = "BIST",
) -> BISTScore:
    """
    Weighted composite score (bkz. WEIGHTS — kalibrasyonla güncellendi, 2026-07):
        Total = (Technical x 0.45) + (Sentiment x 0.25)
              + (Premium   x 0.20) + (Valuation x 0.10)
    Eksik veri durumunda ağırlıklar mevcut bileşenlere dağıtılır (aşağıda).
    """
    result = BISTScore(ticker=ticker.upper())

    # Fetch Data
    stock        = DataFetcher.fetch(ticker, period=period, market=market)
    result.stock = stock
    if stock.error:
        result.signal = "HATA"
        return result

    # Technical Analysis — stil hisse segmentine göre (bkz. _tech_style_for)
    technical           = TechnicalEngine.compute(stock.df, style=_tech_style_for(ticker, market))

    # .5: Relative Strength vs benchmark index (cached)
    _bench_idx = _default_index(market)
    try:
        if len(stock.df) >= 14:
            xu_df = _fetch_benchmark_cached(_bench_idx)
            if xu_df is not None and not xu_df.empty and "Close" in xu_df:
                xu_close = xu_df["Close"].dropna()
                st_close = stock.df["Close"].dropna()
                if len(xu_close) >= 14 and len(st_close) >= 14:
                    base_st = float(st_close.iloc[-14])
                    base_xu = float(xu_close.iloc[-14])
                    if base_st > 0 and base_xu > 0:  # Division by zero koruması
                        s_ret = (float(st_close.iloc[-1]) - base_st) / base_st
                        x_ret = (float(xu_close.iloc[-1]) - base_xu) / base_xu
                        rs_val = (s_ret - x_ret) * 100.0
                        technical.relative_strength = round(float(rs_val), 1)

                        if rs_val > 5.0:
                            technical.score = min(100.0, float(technical.score + 10))
                        elif rs_val > 0.0:
                            technical.score = min(100.0, float(technical.score + 5))
    except Exception as exc:
        log.warning("Relative strength hesaplama hatası: %s", exc)

    result.technical    = technical
    result.teknik_score = technical.score

    # News Sentiment — real data only, no fake fallbacks
    if NEWS_ENGINE_AVAILABLE:
        news_result = analyze_news(ticker, days=30, language=language)
        sentiment   = SentimentResult(
            score          = news_result.score,
            positive_count = news_result.positive_count,
            negative_count = news_result.negative_count,
            headlines      = news_result.headlines,
        )
        result._kap_disclosures  = news_result.kap_disclosures
        result._material_events  = news_result.material_events
        result._news_result      = news_result
    elif FAZ2_AVAILABLE:
        _cache     = DataCache()
        kap_result = enhanced_news_analysis(ticker, _cache)
        sentiment  = SentimentResult(
            score          = kap_result["score"],
            positive_count = kap_result["positive_count"],
            negative_count = kap_result["negative_count"],
            headlines      = kap_result["headlines"],
        )
        result._kap_disclosures = kap_result.get("kap_disclosures", [])
        result._material_events = kap_result.get("material_events", [])
        result._news_result     = None
    else:
        sentiment = SentimentResult(score=50.0, headlines=[])
        result._kap_disclosures = []
        result._material_events = []
        result._news_result     = None

    result.sentiment       = sentiment
    result.sentiment_score = sentiment.score

    # Valuation & Premium
    _auto_target = analyst_target
    # TradingView analist konsensüsü — elle hedef girilmediyse otomatik kaynak
    # (tek toplu sorgu 4 saat cache'lenir; min 3 analist şartı fetch içinde)
    if _auto_target is None and market == "BIST":
        _tv = fetch_tv_analyst_targets().get(ticker.upper().replace(".IS", ""))
        if _tv:
            _auto_target = _tv["target"]
            result._tv_target = _tv  # UI'da "N analist" göstermek için
    if FAZ2_AVAILABLE and _auto_target is None:
        _cache    = DataCache()
        fetcher   = TargetPriceFetcher(_cache)
        consensus = fetcher.fetch(ticker, stock.current_price)
        result._target_consensus = consensus
        if consensus.analyst_count > 0:
            _auto_target = consensus.consensus_target
    else:
        result._target_consensus = None

    valuation          = ValuationEngine.compute(stock, _auto_target)
    result.valuation   = valuation
    result.prim_score  = valuation.prim_score
    result.deger_score = valuation.deger_score

    # Dinamik Ağırlıklı Toplam Skor
    # Problem: Hedef fiyat yoksa prim_score=50 (nötr) → 43-56 aralığına yığılma.
    # Çözüm: Veri yoksa o bileşenin ağırlığını mevcut bileşenlere orantılı dağıt.
    has_target    = valuation.target_price is not None
    has_fk        = stock.pe_ratio is not None

    # Temel ağırlıklar
    w_teknik    = WEIGHTS["teknik"]     # 45
    w_sentiment = WEIGHTS["sentiment"]  # 25
    w_prim      = WEIGHTS["prim"]       # 20
    w_deger     = WEIGHTS["deger"]      # 10

    # Hedef fiyat yoksa prim ağırlığını teknik + sentimente eşit böl
    if not has_target:
        extra = w_prim
        w_teknik    += extra // 2         # +10 → 55
        w_sentiment += extra - extra // 2  # +10 → 35
        w_prim       = 0

    # F/K yoksa değer ağırlığını tekniğe ver
    if not has_fk:
        w_teknik += w_deger  # +10
        w_deger   = 0

    total_w = w_teknik + w_sentiment + w_prim + w_deger  # 100 veya değişken

    result.total_score = round(
        (
            result.teknik_score    * w_teknik
            + result.sentiment_score * w_sentiment
            + result.prim_score      * w_prim
            + result.deger_score     * w_deger
        ) / max(total_w, 1),
        1,
    )

    # Signal
    result.signal, result.signal_color = _score_to_signal(result.total_score)
    return result


def _score_to_signal(score: float) -> tuple:
    """
    Skor → Sinyal eşikleri
    Dağılım genelde 30-80 arasında → eşikler buna göre ayarlandı.
    """
    if   score >= 72: return "GUCLU AL",  "#1d6f4e"   # Güçlü Al  ≥ 72
    elif score >= 57: return "AL",         "#3e8e6c"   # Al        57–71
    elif score >= 43: return "NOTR",       "#6b6357"   # Nötr      43–56
    elif score >= 30: return "SAT",        "#b45309"   # Sat       30–42
    else:             return "GUCLU SAT",  "#9e2b25"   # Güçlü Sat < 30


# 6. RISK ENGINE

@dataclass
class RiskLevels:
    stop_loss_tight: float   = 0.0
    stop_loss_normal: float  = 0.0
    stop_loss_wide: float    = 0.0
    take_profit_1: float     = 0.0
    take_profit_2: float     = 0.0
    risk_reward_ratio: float = 0.0
    saturation_warning: bool = False
    saturation_message: str  = ""


class RiskEngine:
    """
    """
    @staticmethod
    def compute(score: BISTScore) -> RiskLevels:
        price = score.stock.current_price
        r     = RiskLevels()
        if price <= 0:
            return r

        t   = score.technical
        atr = t.atr  # ATR hesaplanmışsa gerçek değer, yoksa 0.0

        # ATR yoksa (veri az, yeni hisse vb.) yüzde tabanlı fallback
        if atr <= 0 or price <= 0:
            atr = price * 0.02  # %2'yi 1 günlük ATR gibi say (muhafazakâr)

        # Stop-Loss seviyeleri (ATR çarpanı ile)
        # Tight  : 1.5× ATR — scalp / kısa vadeli pozisyonlar
        # Normal : 2.5× ATR — swing trade (1-4 hafta)
        # Wide   : SMA200 tabanlı — pozisyon trade / uzun vade
        r.stop_loss_tight  = round(max(price - 1.5 * atr, price * 0.90), 2)
        r.stop_loss_normal = round(max(price - 2.5 * atr, price * 0.82), 2)

        # Wide stop: SMA200'ün %2 altı, ama fiyatın %20'sinden aşağı inmez
        sma200 = t.sma200 if t.sma200 > 0 else price * 0.85
        sma200_based = sma200 * 0.98
        floor_15pct  = price * 0.80   # maksimum kayıp sınırı %20
        if sma200_based >= price:
            # SMA200 zaten fiyatın üstünde (düşüş trendi) → %20 sınırını kullan
            r.stop_loss_wide = round(floor_15pct, 2)
        else:
            r.stop_loss_wide = round(max(sma200_based, floor_15pct), 2)

        # Take-profit hedefleri
        # Target 1: minimum 1.5:1 R/R oranını garanti et
        #   risk = price - stop_normal
        #   reward = risk * 2.0  →  hedef = price + 2.0 * risk
        risk_amount = price - r.stop_loss_normal
        if risk_amount > 0:
            tp1_rr_based = price + 2.0 * risk_amount   # 2:1 R/R garantisi
        else:
            tp1_rr_based = price * 1.12

        # RSI yüksekse zaten pahalı, hedefi biraz yaklaştır
        rsi = t.rsi
        if rsi > 75:
            tp1_rr_based = min(tp1_rr_based, price * 1.08)
        elif rsi > 65:
            tp1_rr_based = min(tp1_rr_based, price * 1.12)

        r.take_profit_1 = round(tp1_rr_based, 2)

        # Target 2: önce analist hedefi, sonra 52H yüksek (erişilebilirse),
        #           son olarak 3.5× ATR bazlı fallback
        analyst_tp = score.valuation.target_price
        w52_high   = t.week52_high

        if analyst_tp and analyst_tp > price * 1.03:
            r.take_profit_2 = round(analyst_tp, 2)
        elif w52_high and w52_high > price * 1.05 and t.week52_position < 0.85:
            # 52 haftalık zirveye %85'ten yakın değilse hedef olarak kullan
            r.take_profit_2 = round(w52_high, 2)
        else:
            r.take_profit_2 = round(price + 3.5 * risk_amount if risk_amount > 0 else price * 1.20, 2)

        # Risk / Reward Oranı
        # Normal stop ile Target 1 arasındaki oran
        # Profesyonel eşik: R/R > 2 çok iyi, 1.5-2 kabul edilebilir, < 1.5 kötü
        stop_dist   = price - r.stop_loss_normal
        reward_dist = r.take_profit_1 - price
        r.risk_reward_ratio = round(reward_dist / stop_dist, 2) if stop_dist > 0 else 0.0

        # Uyarı Koşulları
        warnings = []
        if rsi > 70:
            warnings.append(
                f"⚠️ RSI {rsi:.1f} → Aşırı alım! Yeni pozisyon yerine kâr alımı değerlendirilebilir."
            )
        if t.adx > 0 and t.adx < 20:
            warnings.append(
                f"⚠️ ADX {t.adx:.1f} → Yatay piyasa. Kırılım beklemeden işlem açmak riskli."
            )
        if not t.price_above_sma200:
            warnings.append(
                "⚠️ Fiyat SMA200 altında → Uzun vadeli düşüş trendi. Stop-loss disiplini kritik."
            )
        if t.atr_pct > 4.0:
            warnings.append(
                f"⚠️ Yüksek volatilite (ATR %{t.atr_pct:.1f}) → Stop seviyeleri geniş, pozisyon büyüklüğünü küçük tut."
            )
        if r.risk_reward_ratio > 0 and r.risk_reward_ratio < 1.5:
            warnings.append(
                f"⚠️ R/R oranı {r.risk_reward_ratio:.2f} → 1.5 altında işlem açmak iyi pratik değil."
            )

        if warnings:
            r.saturation_warning = True
            r.saturation_message = "  \n".join(warnings)

        return r


# 7. AI REPORTER (REMOVED)
# AI Summary was removed to avoid API costs as per user request.


# 7B. BACKTEST ENGINE

# ABD Piyasası Popüler Hisseler
US_POPULAR_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "COST",
    "NFLX", "CRM", "AMD", "INTC", "DIS", "PYPL", "BA", "NKE", "SBUX",
    "KO", "PEP", "MCD", "ABBV", "MRK", "PFE", "LLY", "TMO", "AVGO",
    "ORCL", "ADBE", "QCOM", "CSCO", "TXN", "AMAT", "LRCX", "MU",
]

US_SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Semiconductors",
    "GOOGL": "Technology", "AMZN": "Consumer", "META": "Technology",
    "TSLA": "Automotive", "BRK-B": "Finance", "JPM": "Finance",
    "V": "Finance", "UNH": "Healthcare", "XOM": "Energy",
    "JNJ": "Healthcare", "WMT": "Retail", "MA": "Finance",
    "PG": "Consumer", "HD": "Retail", "COST": "Retail",
    "NFLX": "Technology", "CRM": "Technology", "AMD": "Semiconductors",
    "INTC": "Semiconductors", "DIS": "Media", "PYPL": "Finance",
    "BA": "Aerospace", "NKE": "Consumer", "SBUX": "Consumer",
    "KO": "Consumer", "PEP": "Consumer", "MCD": "Consumer",
    "ABBV": "Healthcare", "MRK": "Healthcare", "PFE": "Healthcare",
    "LLY": "Healthcare", "TMO": "Healthcare", "AVGO": "Semiconductors",
    "ORCL": "Technology", "ADBE": "Technology", "QCOM": "Semiconductors",
    "CSCO": "Technology", "TXN": "Semiconductors", "AMAT": "Semiconductors",
    "LRCX": "Semiconductors", "MU": "Semiconductors",
}

US_INDEX_MAP = {
    "Technology": "^IXIC", "Semiconductors": "^SOX",
    "Finance": "^BKX", "Healthcare": "^IXHC",
    "Consumer": "^GSPC", "Retail": "^GSPC",
    "Energy": "^GSPE", "Automotive": "^GSPC",
    "Aerospace": "^GSPC", "Media": "^GSPC",
}

# Varsayılan 5 hisse — yüksek likidite, uzun geçmiş veri, BIST30 üyeleri
BACKTEST_TICKERS = ["THYAO", "GARAN", "EREGL", "SISE", "TUPRS"]

# Genişletilmiş liste (kullanıcı ekleyebilir)
BACKTEST_TICKERS_EXTENDED = [
    "THYAO", "GARAN", "AKBNK", "EREGL", "SISE",
    "TUPRS", "BIMAS", "KCHOL", "TCELL", "ASELS",
    "FROTO", "PETKM", "ISCTR", "VAKBN", "TOASO",
]

# Sektör haritası — portföy çeşitlendirmesi için (aynı sektörden max 2 hisse)
SECTOR_MAP: dict = {
    # Bankacılık
    "AKBNK": "Bankacılık", "GARAN": "Bankacılık", "ISCTR": "Bankacılık",
    "VAKBN": "Bankacılık", "YKBNK": "Bankacılık", "HALKB": "Bankacılık",
    "FINBN": "Bankacılık", "QNBFB": "Bankacılık",
    # Sigorta & Emeklilik
    "ANSGR": "Sigorta", "AGESA": "Sigorta",
    # Havacılık & Ulaşım
    "THYAO": "Havacılık", "PGSUS": "Havacılık", "TAVHL": "Havacılık",
    # Demir-Çelik & Metal
    "EREGL": "Demir-Çelik", "KRDMD": "Demir-Çelik", "ISDMR": "Demir-Çelik",
    # Enerji & Rafineri
    "TUPRS": "Enerji", "AKSEN": "Enerji", "ENKAI": "Enerji",
    "ENJSA": "Enerji", "IPEKE": "Enerji", "ENERY": "Enerji",
    # Perakende & Gıda
    "BIMAS": "Perakende", "MGROS": "Perakende", "SOKM": "Perakende",
    "BANVT": "Gıda", "GOLTS": "Gıda", "AEFES": "Gıda",
    # Otomotiv
    "FROTO": "Otomotiv", "TOASO": "Otomotiv",
    # Telekom
    "TCELL": "Telekom", "TTKOM": "Telekom",
    # Cam & Kimya & Plastik
    "SISE": "Cam-Kimya", "PETKM": "Cam-Kimya", "HEKTS": "Cam-Kimya",
    "ALKIM": "Cam-Kimya", "GUBRF": "Cam-Kimya", "SASA": "Cam-Kimya",
    # Holding & Yatırım
    "KCHOL": "Holding", "SAHOL": "Holding", "DOHOL": "Holding",
    "GLYHO": "Holding",
    # Savunma & Teknoloji
    "ASELS": "Savunma", "INDES": "Teknoloji",
    # Gayrimenkul (GYO)
    "EKGYO": "GYO", "ISGYO": "GYO", "EMLAK": "GYO",
    # Madencilik & Altın
    "KOZAA": "Madencilik", "KOZAL": "Madencilik",
    # Diğer Sanayi
    "BRISA": "Lastik", "CIMSA": "Çimento", "AYGAZ": "Gaz-Dağıtım",
    "ARCLK": "Beyaz Eşya", "GENIL": "Sanayi", "ALARK": "Sanayi",
    "ALFAS": "Sanayi", "BAGFS": "Gübre", "DOAS": "Otomotiv-Satış",
    "EGEEN": "Sağlık", "FENER": "Spor", "ISGYO": "GYO",
}

# Sektörel endeks eşleşmesi (Yahoo Finance ticker)
SECTOR_INDEX_MAP = {
    "Bankacılık":    "XBANK.IS",
    "Sigorta":       "XBANK.IS",
    "Havacılık":     "XU100.IS",
    "Demir-Çelik":   "XUSIN.IS",
    "Enerji":        "XUSIN.IS",
    "Perakende":     "XTCRT.IS",
    "Gıda":          "XGIDA.IS",
    "Otomotiv":      "XUSIN.IS",
    "Telekom":       "XILTM.IS",
    "Cam-Kimya":     "XUSIN.IS",
    "Holding":       "XHOLD.IS",
    "Savunma":       "XUTEK.IS",
    "Teknoloji":     "XUTEK.IS",
    "GYO":           "XGMYO.IS",
    "Madencilik":    "XUSIN.IS",
    "Beyaz Eşya":    "XUSIN.IS",
    "Sanayi":        "XUSIN.IS",
    "Gübre":         "XUSIN.IS",
    "Lastik":        "XUSIN.IS",
    "Çimento":       "XUSIN.IS",
    "Gaz-Dağıtım":  "XUSIN.IS",
    "Otomotiv-Satış":"XTCRT.IS",
    "Sağlık":        "XU100.IS",
    "Spor":          "XU100.IS",
    "Diğer":         "XU100.IS",
}

# Tüm taranabilir BIST hisseleri (PortfolioScanner için)
BIST_SCAN_UNIVERSE = [
    # BIST30
    "AKBNK","ARCLK","ASELS","BIMAS","EKGYO","ENKAI","EREGL","FROTO","GARAN",
    "GUBRF","HEKTS","ISCTR","KCHOL","KOZAA","KOZAL","KRDMD","PETKM","PGSUS",
    "SAHOL","SASA","SISE","TAVHL","TCELL","THYAO","TOASO","TTKOM","TUPRS",
    "VAKBN","YKBNK","AKSEN",
    # BIST50 ek hisseler
    "AEFES","AGESA","ALARK","ALFAS","ALKIM","ANSGR","AYGAZ","BAGFS","BANVT",
    "BRISA","CIMSA","DOHOL","EGEEN","EMLAK","ENERY","ENJSA","FENER","FINBN",
    "GENIL","GLYHO","GOLTS","HALKB","INDES","IPEKE","ISDMR","ISGYO","ISGYO",
    "KARSN","KATMR","KONTR","KONYA","LOGO","MAVI","MGROS","MPARK","NETAS",
    "ODAS","OTKAR","OYAKC","PARSN","PRKAB","RBOAK","SELEC","SILVR","SKBNK",
    "SMART","SOKM","TATGD","TRGYO","TRILC","TURSG","UFUK","ULKER","VESBE",
    "VESTL","ZOREN","BRSAN","CEMTS","DOAS","KERVT","METUR","NUHCM","TSKB",
    "TTRAK","YUNSA",
]
# Tekrar edenleri temizle
BIST_SCAN_UNIVERSE = list(dict.fromkeys(BIST_SCAN_UNIVERSE))

# DİNAMİK EVREN (2026-07): el yapımı liste yerine TradingView'dan likidite
# filtreli TÜM BIST. 30 günlük ort. hacim >= 20M TL → ~500/600 hisse geçer;
# yalnızca gerçekten işlem görmeyen mikro/kabuk hisseler elenir.
# Eski statik liste yedek olarak kalır (TV erişilemezse tarama durmasın).

UNIVERSE_MIN_TL_VOLUME = 20_000_000   # günlük ort. işlem hacmi eşiği (TL)

_TV_SECTOR_TR = {
    "Electronic Technology": "Teknoloji", "Technology Services": "Teknoloji",
    "Finance": "Finans", "Communications": "İletişim",
    "Energy Minerals": "Enerji", "Utilities": "Enerji Dağıtım",
    "Non-Energy Minerals": "Madencilik & Metal", "Process Industries": "Proses Sanayi",
    "Producer Manufacturing": "Sanayi", "Industrial Services": "Sanayi Hizmetleri",
    "Consumer Non-Durables": "Temel Tüketim", "Consumer Durables": "Dayanıklı Tüketim",
    "Consumer Services": "Tüketici Hizmetleri", "Retail Trade": "Perakende",
    "Transportation": "Ulaştırma", "Health Technology": "Sağlık",
    "Health Services": "Sağlık", "Commercial Services": "Ticari Hizmetler",
    "Distribution Services": "Dağıtım", "Miscellaneous": "Diğer",
}

_UNIVERSE_CACHE = {"date": None, "tickers": None, "sectors": {}}


def get_scan_universe() -> list:
    """Likidite filtreli dinamik BIST evreni (günde 1 kez TV'den çekilir).

    'Eksik kalmasın' ilkesi: hacim eşiğini geçen HER hisse + eski el listesi
    + aktif portföylerdeki pozisyonlar (izlenen hisse asla evren dışı kalmaz).
    TV erişilemezse statik listeye düşer.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if _UNIVERSE_CACHE["date"] == today and _UNIVERSE_CACHE["tickers"]:
        return _UNIVERSE_CACHE["tickers"]
    try:
        from tradingview_screener import Query
        _n, df = (Query().set_markets("turkey")
                  .select("name", "close", "average_volume_30d_calc", "sector")
                  .limit(700).get_scanner_data())
        df["tl_hacim"] = df["close"] * df["average_volume_30d_calc"]
        likit = df[df["tl_hacim"] >= UNIVERSE_MIN_TL_VOLUME]
        tickers = [str(t) for t in likit["name"] if t and str(t).isalpha()]
        sectors = {}
        for _, r in df.iterrows():
            t, sec = str(r.get("name") or ""), r.get("sector")
            if t and isinstance(sec, str):
                sectors[t] = _TV_SECTOR_TR.get(sec, sec)
        # Eski liste + portföy pozisyonları her koşulda dahil
        merged = list(dict.fromkeys(tickers + BIST_SCAN_UNIVERSE))
        try:
            ports = PortfolioManager.active_portfolios()
            pos = PortfolioManager.positions_all([p["id"] for p in ports])
            held = {x["ticker"] for xs in pos.values() for x in xs}
            merged = list(dict.fromkeys(merged + sorted(held)))
        except Exception:
            pass
        _UNIVERSE_CACHE.update({"date": today, "tickers": merged, "sectors": sectors})
        log.info("Dinamik evren: %d hisse (esik %dM TL)", len(merged), UNIVERSE_MIN_TL_VOLUME // 10**6)
        return merged
    except Exception as exc:
        log.warning("Dinamik evren alınamadı, statik listeye düşülüyor: %s", exc)
        return BIST_SCAN_UNIVERSE

# 7A. PORTFOLIO SCANNER & SMART PORTFOLIO BUILDER

@dataclass
class StockScanResult:
    ticker:        str   = ""
    score:         float = 0.0
    rsi:           float = 50.0
    adx:           float = 0.0
    atr_pct:       float = 0.0
    bb_position:   float = 0.5
    week52_pos:    float = 0.5
    obv_trend:     str   = "notr"
    golden_cross:  bool  = False
    price_above_sma200: bool = False
    current_price: float = 0.0
    volume_ok:     bool  = False   # Yeterli hacim var mı
    data_rows:     int   = 0
    error:         str   = ""
    momentum_1m:   float = 0.0   # 21 günlük fiyat değişimi %
    momentum_3m:   float = 0.0   # 63 günlük fiyat değişimi %
    price_above_sma50: bool = False
    macd_bullish:  bool = False   # MACD histogram > 0
    adx_strong:    bool = False   # ADX > 22
    volume_ratio:  float = 1.0   # 5-gün hacim / 20-gün ort. hacim
    sma_gap_pct:   float = 0.0   # (SMA50-SMA200)/SMA200*100
    bb_squeeze:    bool = False   # Bollinger squeeze
    stoch_oversold: bool = False
    comp_score:    float = 0.0   # Bileşik sıralama skoru


class PortfolioScanner:
    """
    BIST_SCAN_UNIVERSE'deki tüm hisseleri toplu olarak tarar,
    her biri için teknik skor + filtre metrikleri hesaplar.
    Sonuçlar SQLite'a kaydedilir (günlük cache).
    """
    CACHE_HRS = 6   # 6 saatten yeni scan sonucu varsa yeniden çekme

    @staticmethod
    def _init_table():
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_scan (
                    ticker TEXT PRIMARY KEY,
                    scan_date TEXT,
                    score REAL,
                    rsi REAL,
                    adx REAL,
                    atr_pct REAL,
                    bb_position REAL,
                    week52_pos REAL,
                    obv_trend TEXT,
                    golden_cross INTEGER,
                    price_above_sma200 INTEGER,
                    current_price REAL,
                    volume_ok INTEGER,
                    data_rows INTEGER,
                    error TEXT,
                    momentum_1m REAL,
                    momentum_3m REAL,
                    price_above_sma50 INTEGER,
                    macd_bullish INTEGER,
                    adx_strong INTEGER,
                    volume_ratio REAL,
                    sma_gap_pct REAL,
                    bb_squeeze INTEGER,
                    stoch_oversold INTEGER,
                    comp_score REAL
                )
            """)
            conn.commit()

    @staticmethod
    def scan_all(force: bool = False, progress_cb=None) -> list:
        """
        Tüm BIST_SCAN_UNIVERSE hisselerini tara.
        progress_cb(ticker, idx, total) → UI progress güncellemesi için
        Sonuç: List[StockScanResult]
        """
        PortfolioScanner._init_table()

        # Cache kontrolü — tablo varsa ve verisi tazeyse direkt dön
        if not force:
            cached = PortfolioScanner._load_cache()
            if cached:
                return cached

        # Yeni tarama yapılacak — eski veriyi temizle
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM portfolio_scan")
                conn.commit()
        except Exception:
            pass

        universe = get_scan_universe()   # dinamik likidite-filtreli evren (~500 hisse)
        results = []
        total   = len(universe)

        # Toplu veri indirme (yfinance batch download — çok daha hızlı)
        # ~500 sembol tek istekte sorun çıkarabilir → 100'lük partiler
        symbols = [_yf_symbol(t, "BIST") for t in universe]
        bulk_parts = []
        try:
            for i in range(0, len(symbols), 100):
                part = yf.download(
                    symbols[i:i+100], period="1y",
                    auto_adjust=True, progress=False,
                    group_by="ticker",
                )
                bulk_parts.append(part)
                if progress_cb:
                    # İndirme aşamasını ilerlemenin ilk %30'u say
                    done = min(i + 100, len(symbols))
                    progress_cb(f"veri indiriliyor {done}/{len(symbols)}",
                                int(done / len(symbols) * total * 0.3), total)
            bulk = pd.concat(bulk_parts, axis=1) if bulk_parts else None
        except Exception as exc:
            log.error("Bulk download hatası: %s", exc)
            bulk = bulk_parts[0] if bulk_parts else None

        for idx, ticker in enumerate(universe):
            if progress_cb:
                progress_cb(ticker, int(total * 0.3 + idx * 0.7), total)

            result = StockScanResult(ticker=ticker)
            try:
                sym = _yf_symbol(ticker, "BIST")

                # Toplu indirmeden al, başarısız olursa tekil çek
                df = PortfolioScanner._extract_df(bulk, sym, ticker)
                if df is None or df.empty or len(df) < 60:
                    # Fallback: tekil indir
                    df = yf.Ticker(sym).history(period="1y", auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                if df.empty or len(df) < 60:
                    result.error = "Veri yetersiz"
                    results.append(result)
                    continue

                # Kolon normalize
                df.index = pd.to_datetime(df.index).tz_localize(None)
                col_rn = {c: c.strip().title() for c in df.columns
                          if c.strip().title() in ("Open","High","Low","Close","Volume")}
                if col_rn:
                    df = df.rename(columns=col_rn)
                if "High"   not in df.columns: df["High"]   = df["Close"]
                if "Low"    not in df.columns: df["Low"]    = df["Close"]
                if "Volume" not in df.columns: df["Volume"] = 0.0

                # TechnicalEngine ile tam analiz — stil hisse segmentine göre
                tech = TechnicalEngine.compute(df, style=_tech_style_for(ticker, "BIST"))

                result.score             = tech.score
                result.rsi               = tech.rsi
                result.adx               = tech.adx
                result.atr_pct           = tech.atr_pct
                result.bb_position       = tech.bb_position
                result.week52_pos        = tech.week52_position
                result.obv_trend         = tech.obv_trend
                result.golden_cross      = tech.golden_cross
                result.price_above_sma200 = tech.price_above_sma200
                result.current_price     = tech.current_price
                result.data_rows         = len(df)

                # Yeni alanlar — TechnicalResult'tan al
                result.price_above_sma50 = tech.price_above_sma50
                result.macd_bullish      = tech.macd_bullish
                result.adx_strong        = tech.adx > 22
                result.sma_gap_pct       = tech.sma_gap_pct
                result.bb_squeeze        = tech.bb_squeeze
                result.stoch_oversold    = tech.stoch_oversold

                # Momentum hesapla
                close = df["Close"]
                if len(close) >= 22:
                    result.momentum_1m = float(
                        round((close.iloc[-1] / close.iloc[-22] - 1) * 100, 2)
                    )
                if len(close) >= 64:
                    result.momentum_3m = float(
                        round((close.iloc[-1] / close.iloc[-64] - 1) * 100, 2)
                    )

                # Volume ratio (son 5 gün / 20 gün ort)
                if "Volume" in df.columns and len(df) >= 20:
                    vol_20 = float(df["Volume"].rolling(20).mean().iloc[-1])
                    vol_5  = float(df["Volume"].rolling(5).mean().iloc[-1])
                    result.volume_ratio = round(vol_5 / vol_20, 2) if vol_20 > 0 else 1.0

                # Hacim kontrolü: ortalama günlük hacim > 1M TL
                if "Volume" in df.columns:
                    avg_vol_tl = float((df["Volume"] * df["Close"]).rolling(20).mean().iloc[-1])
                    result.volume_ok = avg_vol_tl > 1_000_000

            except Exception as exc:
                result.error = str(exc)[:100]
                log.warning("PortfolioScanner hata (%s): %s", ticker, exc)

            results.append(result)

        # SQLite'a kaydet
        PortfolioScanner._save_cache(results)
        return results

    @staticmethod
    def _extract_df(bulk, sym: str, ticker: str):
        """Bulk download'dan tek hisse DataFrame çıkarır."""
        if bulk is None or bulk.empty:
            return None
        try:
            cols = bulk.columns
            if not isinstance(cols, pd.MultiIndex):
                return bulk  # Tek hisse indirilmişse
            lvl0 = cols.get_level_values(0)
            lvl1 = cols.get_level_values(1)
            # Yeni format: (Price, Ticker)
            if "Close" in lvl0:
                price_cols = [c for c in ("Open","High","Low","Close","Volume") if c in lvl0]
                data = {}
                for pc in price_cols:
                    if sym in bulk[pc].columns:
                        data[pc] = bulk[pc][sym]
                return pd.DataFrame(data).dropna(how="all") if data else None
            # Eski format: (Ticker, Price)
            if sym in lvl0:
                return bulk[sym].dropna(how="all")
        except Exception:
            pass
        return None

    @staticmethod
    def _save_cache(results: list):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with sqlite3.connect(DB_PATH) as conn:
            for r in results:
                conn.execute("""
                    INSERT OR REPLACE INTO portfolio_scan
                        (ticker, scan_date, score, rsi, adx, atr_pct, bb_position,
                         week52_pos, obv_trend, golden_cross, price_above_sma200,
                         current_price, volume_ok, data_rows, error, momentum_1m,
                         momentum_3m, price_above_sma50, macd_bullish, adx_strong,
                         volume_ratio, sma_gap_pct, bb_squeeze, stoch_oversold, comp_score)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    r.ticker, now, r.score, r.rsi, r.adx, r.atr_pct,
                    r.bb_position, r.week52_pos, r.obv_trend,
                    int(r.golden_cross), int(r.price_above_sma200),
                    r.current_price, int(r.volume_ok), r.data_rows, r.error,
                    r.momentum_1m, r.momentum_3m, int(r.price_above_sma50),
                    int(r.macd_bullish), int(r.adx_strong), r.volume_ratio,
                    r.sma_gap_pct, int(r.bb_squeeze), int(r.stoch_oversold),
                    r.comp_score,
                ))
            conn.commit()

    @staticmethod
    def _load_cache() -> list:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT * FROM portfolio_scan ORDER BY score DESC
                """).fetchall()
            if not rows:
                return []
            scan_date_str = rows[0][1]
            try:
                scan_dt = datetime.strptime(scan_date_str, "%Y-%m-%d %H:%M")
                age_hrs = (datetime.now() - scan_dt).total_seconds() / 3600
                if age_hrs > PortfolioScanner.CACHE_HRS:
                    return []
            except Exception:
                return []
            results = []
            for row in rows:
                r = StockScanResult(
                    ticker=row[0], score=row[2] or 0, rsi=row[3] or 50,
                    adx=row[4] or 0, atr_pct=row[5] or 0, bb_position=row[6] or 0.5,
                    week52_pos=row[7] or 0.5, obv_trend=row[8] or "notr",
                    golden_cross=bool(row[9]), price_above_sma200=bool(row[10]),
                    current_price=row[11] or 0, volume_ok=bool(row[12]),
                    data_rows=row[13] or 0, error=row[14] or "",
                    momentum_1m=row[15] or 0, momentum_3m=row[16] or 0,
                    price_above_sma50=bool(row[17]), macd_bullish=bool(row[18]),
                    adx_strong=bool(row[19]), volume_ratio=row[20] or 1.0,
                    sma_gap_pct=row[21] or 0, bb_squeeze=bool(row[22]),
                    stoch_oversold=bool(row[23]), comp_score=row[24] or 0,
                )
                results.append(r)
            return results
        except Exception:
            return []


class SmartPortfolioBuilder:
    """
    """

    @staticmethod
    def build(scan_results: list) -> dict:
        valid = [r for r in scan_results
                 if not r.error and r.data_rows >= 60 and r.current_price > 0]

        # Önce agresif seç
        aggressive = SmartPortfolioBuilder._build_aggressive(valid)
        agg_tickers = {s["ticker"] for s in aggressive}

        # Defansiften agresif hisseleri çıkar
        defensive_pool = [r for r in valid if r.ticker not in agg_tickers]
        defensive = SmartPortfolioBuilder._build_defensive(defensive_pool)

        return {"aggressive": aggressive, "defensive": defensive}

    @staticmethod
    def _build_aggressive(stocks: list) -> list:
        """
    """
        candidates = []
        for r in stocks:
            if r.score < 43:              continue   # Biraz daha seçici ama ulaşılabilir
            if r.adx < 20:                continue   # Trend yok, yatay piyasa
            if not r.price_above_sma200:  continue   # Uzun vade kırık — kesin filtre
            # SMA50 için toleranslı: üstünde VEYA momentum_1m pozitif (toparlanıyor)
            if not r.price_above_sma50 and r.momentum_1m <= 0:  continue
            if not (35 <= r.rsi <= 72):   continue   # Genişletilmiş bölge
            if r.momentum_3m < -5:        continue   # Sadece sert düşüşleri ele
            # volume_ratio zorunlu kritere gerek yok (sıralama bonusu yeterli)

            comp = r.score
            comp += min(r.adx - 22, 20) * 0.4
            comp += max(0, r.momentum_1m) * 0.6
            comp += max(0, r.momentum_3m) * 0.3
            comp += (r.volume_ratio - 1.0) * 6
            if r.obv_trend == "yukari":  comp += 5
            if r.macd_bullish:           comp += 4
            if r.golden_cross:           comp += 3
            if r.sma_gap_pct > 1:        comp += 2
            if r.bb_squeeze:             comp += 3
            r.comp_score = round(comp, 2)

            reasons = [f"Skor {r.score:.0f}", f"ADX {r.adx:.0f}", f"RSI {r.rsi:.0f}"]
            if r.golden_cross:          reasons.append("Golden Cross")
            if r.obv_trend == "yukari": reasons.append("OBV↑")
            if r.macd_bullish:          reasons.append("MACD↑")
            if r.momentum_1m > 0:       reasons.append(f"1A%{r.momentum_1m:+.1f}")
            if r.volume_ratio >= 1.2:   reasons.append(f"Hacim×{r.volume_ratio:.1f}")
            if r.bb_squeeze:            reasons.append("Sıkışma!")

            candidates.append({
                "ticker":       r.ticker,
                "score":        round(r.score, 1),
                "comp_score":   round(r.comp_score, 1),
                "rsi":          round(r.rsi, 1),
                "adx":          round(r.adx, 1),
                "atr_pct":      round(r.atr_pct, 2),
                "week52_pos":   round(r.week52_pos * 100, 1),
                "momentum_1m":  round(r.momentum_1m, 1),
                "momentum_3m":  round(r.momentum_3m, 1),
                "obv_trend":    r.obv_trend,
                "golden_cross": r.golden_cross,
                "macd_bullish": r.macd_bullish,
                "price":        round(r.current_price, 2),
                "sector":       SECTOR_MAP.get(r.ticker, "Diğer"),
                "reason":       " · ".join(reasons),
                "risk_level":   "Agresif",
            })

        candidates.sort(key=lambda x: x["comp_score"], reverse=True)

        # Sektör çeşitlendirmesi: aynı sektörden max 2 hisse
        sector_count: dict = {}
        selected = []
        for c in candidates:
            sec = c["sector"]
            if sector_count.get(sec, 0) < 2:
                selected.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected) >= 7:
                break
        return selected

    @staticmethod
    def _build_defensive(stocks: list) -> list:
        """
    """
        candidates = []
        for r in stocks:
            if r.score < 38:               continue
            if not r.price_above_sma200:   continue
            if r.atr_pct > 4.5:            continue
            if not (30 <= r.rsi <= 62):    continue
            if r.momentum_3m < -8:         continue   # Sert düşüş var — geç

            comp = r.score
            if r.golden_cross:            comp += 5
            if r.obv_trend in ("yukari","notr"):  comp += 3
            comp += max(0, r.momentum_3m) * 0.3
            if r.week52_pos <= 0.70:       comp += 3
            if r.atr_pct < 2.5:            comp += 4   # Düşük volatilite bonus
            if r.macd_bullish:             comp += 2
            r.comp_score = round(comp, 2)

            reasons = [f"Skor {r.score:.0f}", f"RSI {r.rsi:.0f}", f"ATR%{r.atr_pct:.1f}"]
            if r.golden_cross:             reasons.append("Golden Cross")
            if r.obv_trend == "yukari":    reasons.append("OBV↑")
            if r.momentum_3m > 0:          reasons.append(f"3A%{r.momentum_3m:+.1f}")
            if r.atr_pct < 2.5:            reasons.append("Stabil")

            candidates.append({
                "ticker":       r.ticker,
                "score":        round(r.score, 1),
                "comp_score":   round(r.comp_score, 1),
                "rsi":          round(r.rsi, 1),
                "adx":          round(r.adx, 1),
                "atr_pct":      round(r.atr_pct, 2),
                "week52_pos":   round(r.week52_pos * 100, 1),
                "momentum_1m":  round(r.momentum_1m, 1),
                "momentum_3m":  round(r.momentum_3m, 1),
                "obv_trend":    r.obv_trend,
                "golden_cross": r.golden_cross,
                "macd_bullish": r.macd_bullish,
                "price":        round(r.current_price, 2),
                "sector":       SECTOR_MAP.get(r.ticker, "Diğer"),
                "reason":       " · ".join(reasons),
                "risk_level":   "Defansif",
            })

        candidates.sort(key=lambda x: x["comp_score"], reverse=True)

        # Sektör çeşitlendirmesi: aynı sektörden max 2 hisse
        sector_count: dict = {}
        selected = []
        for c in candidates:
            sec = c["sector"]
            if sector_count.get(sec, 0) < 2:
                selected.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected) >= 7:
                break
        return selected


# 7B. STRATEJI ZAMAN MAKİNESİ (TIME MACHINE ENGINE)

# PORTFÖY YÖNETİCİSİ v2 — ENAG reel getiri + piyasa rejimi + vade/profil portföyleri

# ENAG E-TÜFE aylık artışları (%). Kaynak: enagrup.org / basın duyuruları.
# "tahmini" işaretliler: 2025 yıllığı %56.14'ten türetildi (ay verisi bulunamadı).
# Yeni aylar UI'daki "ENAG Verisi" bölümünden eklenir — tablo DB'de tutulur.
ENAG_MOM_DEFAULTS = {
    # 2020-2024: ENAG yıllık değerlerinden türetilmiş düz aylık eşdeğerler
    # (ay bazlı seri yayında yok; yıllıklar: 2020 %36.7, 2021 %82.8,
    #  2022 %137.5, 2023 %127.2, 2024 %83.4). Tarihsel simülasyon için.
    **{f"2020-{m:02d}": (2.65, "tahmini-yillik") for m in range(1, 13)},
    **{f"2021-{m:02d}": (5.16, "tahmini-yillik") for m in range(1, 13)},
    **{f"2022-{m:02d}": (7.47, "tahmini-yillik") for m in range(1, 13)},
    **{f"2023-{m:02d}": (7.08, "tahmini-yillik") for m in range(1, 13)},
    **{f"2024-{m:02d}": (5.18, "tahmini-yillik") for m in range(1, 13)},
    # Wayback arşivinden hasat edilen GERÇEK aylık değerler (2026-07 taraması;
    # site kopyalarındaki resmi duyurulardan) — düz tahminlerin üzerine yazar:
    "2020-11": (4.03, "resmi-arsiv"), "2020-12": (4.08, "resmi-arsiv"),
    "2021-01": (2.99, "resmi-arsiv"), "2021-03": (3.36, "resmi-arsiv"),
    "2021-04": (2.62, "resmi-arsiv"), "2021-05": (3.94, "resmi-arsiv"),
    "2022-07": (5.03, "resmi-arsiv"), "2022-09": (5.30, "resmi-arsiv"),
    "2023-07": (13.18, "resmi-arsiv"), "2023-10": (5.09, "resmi-arsiv"),
    "2023-11": (5.58, "resmi-arsiv"), "2024-01": (9.38, "resmi-arsiv"),
    "2024-02": (4.32, "resmi-arsiv"), "2024-05": (5.66, "resmi-arsiv"),
    "2024-06": (4.27, "resmi-arsiv"), "2024-12": (2.34, "resmi-arsiv"),
    "2025-02": (3.37, "resmi-arsiv"),
    "2025-01": (5.10, "tahmini"), "2025-02": (5.10, "tahmini"),
    "2025-03": (5.10, "tahmini"), "2025-04": (4.46, "resmi"),
    "2025-05": (3.66, "resmi"),   "2025-06": (1.94, "resmi"),
    "2025-07": (5.10, "tahmini"), "2025-08": (3.23, "resmi"),
    "2025-09": (3.79, "resmi"),   "2025-10": (3.74, "resmi"),
    "2025-11": (2.13, "resmi"),   "2025-12": (2.11, "resmi"),
    "2026-01": (6.32, "resmi"),   "2026-02": (4.01, "resmi"),
    "2026-03": (4.10, "resmi"),   "2026-04": (5.07, "resmi"),
    "2026-05": (2.16, "resmi"),   "2026-06": (1.94, "resmi"),
}


class _PMDB:
    """Portföy Yöneticisi tabloları (pm_*, enag_monthly) için depolama katmanı.

    TURSO_DATABASE_URL + TURSO_AUTH_TOKEN tanımlıysa Turso HTTP API kullanılır
    (Streamlit Cloud'da /tmp silinse bile veri kalıcı olur). Tanımlı değilse
    veya Turso'ya ulaşılamazsa lokal SQLite'a düşer — mevcut davranış korunur.
    Saf HTTP (requests) — libsql paketi gerekmez, her platformda çalışır.
    """
    _cfg_checked = False
    _url = None
    _token = None
    _http = None   # kalıcı bağlantı (TLS bir kez kurulur — sorgu başına ~600ms → ~100ms)

    @staticmethod
    def _session():
        if _PMDB._http is None:
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json"})
            _PMDB._http = s
        return _PMDB._http

    @staticmethod
    def _cfg():
        if not _PMDB._cfg_checked:
            url = _get_secret("TURSO_DATABASE_URL")
            tok = _get_secret("TURSO_AUTH_TOKEN")
            if url and tok:
                if url.startswith("libsql://"):
                    url = "https://" + url[len("libsql://"):]
                _PMDB._url = url.rstrip("/") + "/v2/pipeline"
                _PMDB._token = tok
            _PMDB._cfg_checked = True
        return _PMDB._url, _PMDB._token

    @staticmethod
    def is_cloud() -> bool:
        url, tok = _PMDB._cfg()
        return bool(url and tok)

    @staticmethod
    def _arg(v):
        if v is None:
            return {"type": "null", "value": None}
        if isinstance(v, bool):
            return {"type": "integer", "value": str(int(v))}
        if isinstance(v, (int, np.integer)):
            return {"type": "integer", "value": str(int(v))}
        if isinstance(v, (float, np.floating)):
            return {"type": "float", "value": float(v)}
        return {"type": "text", "value": str(v)}

    @staticmethod
    def _parse_rows(rr: dict) -> list:
        cols = [c["name"] for c in rr.get("cols", [])]
        rows = []
        for raw in rr.get("rows", []):
            d = {}
            for c, cell in zip(cols, raw):
                v, t = cell.get("value"), cell.get("type")
                if v is not None and t == "integer":
                    v = int(v)
                elif v is not None and t == "float":
                    v = float(v)
                d[c] = v
            rows.append(d)
        return rows

    @staticmethod
    def execute_batch(stmts: list) -> list:
        """[(sql, args), ...] — hepsi tek istekte/transaction'da.
        Dönüş: her stmt için {"rows": [...], "lastrowid": int|None} listesi."""
        url, tok = _PMDB._cfg()
        if url:
            try:
                reqs = [{"type": "execute",
                         "stmt": {"sql": s, "args": [_PMDB._arg(a) for a in args]}}
                        for s, args in stmts]
                reqs.append({"type": "close"})
                r = _PMDB._session().post(url, json={"requests": reqs},
                                          headers={"Authorization": f"Bearer {tok}"}, timeout=20)
                r.raise_for_status()
                out = []
                for res in r.json()["results"][:len(stmts)]:
                    if res.get("type") == "error":
                        # SQL hatası: Turso'ya ULAŞILDI, statement geçersiz.
                        # Lokale düşme (split-brain yazma riski) — hatayı yükselt.
                        raise sqlite3.OperationalError(
                            res.get("error", {}).get("message", "turso error"))
                    rr = res["response"]["result"]
                    lid = rr.get("last_insert_rowid")
                    out.append({"rows": _PMDB._parse_rows(rr),
                                "lastrowid": int(lid) if lid else None})
                return out
            except sqlite3.OperationalError:
                raise
            except Exception as exc:
                log.warning("Turso erişim hatası — lokal SQLite'a düşülüyor: %s", exc)
        # Lokal SQLite yolu
        out = []
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            for s, args in stmts:
                cur = conn.execute(s, tuple(args))
                out.append({"rows": [dict(r) for r in cur.fetchall()],
                            "lastrowid": cur.lastrowid})
            conn.commit()
        return out

    @staticmethod
    def execute(sql: str, args: tuple = ()) -> dict:
        return _PMDB.execute_batch([(sql, tuple(args))])[0]


class InflationEngine:
    """ENAG bazlı reel getiri hesaplayıcı.

    Nominal getiri yanıltıcıdır: %40 kazanç, ENAG yıllık %55 iken kayıptır.
    deflator(): dönem boyunca kümülatif enflasyon çarpanı (gün bazında oranlanır).
    Eksik aylar için son 12 bilinen ayın ortalaması kullanılır (yaklaşık).
    """

    _initialized = False

    @staticmethod
    def _init_table():
        if InflationEngine._initialized:
            return
        stmts = [("""
            CREATE TABLE IF NOT EXISTS enag_monthly (
                ym TEXT PRIMARY KEY,
                mom_pct REAL,
                kaynak TEXT DEFAULT 'resmi'
            )
        """, ())]
        stmts += [("INSERT OR IGNORE INTO enag_monthly (ym, mom_pct, kaynak) VALUES (?,?,?)",
                   (ym, pct, src)) for ym, (pct, src) in ENAG_MOM_DEFAULTS.items()]
        _PMDB.execute_batch(stmts)
        InflationEngine._initialized = True

    @staticmethod
    def rates() -> dict:
        InflationEngine._init_table()
        try:
            rows = _PMDB.execute("SELECT ym, mom_pct, kaynak FROM enag_monthly ORDER BY ym")["rows"]
            return {r["ym"]: (r["mom_pct"], r["kaynak"]) for r in rows}
        except Exception:
            return {k: v for k, v in ENAG_MOM_DEFAULTS.items()}

    @staticmethod
    def set_rate(ym: str, pct: float, kaynak: str = "resmi"):
        # kaynak: elle UI girişi = 'resmi' (insan doğruladı). Robotun otomatik
        # web-kazıması 'otomatik-dogrulanmadi' geçmeli — kazınmış değer resmi
        # veriyle karışmasın (dış kaynak bozulursa reel-getiri metriği çürür).
        InflationEngine._init_table()
        _PMDB.execute("INSERT OR REPLACE INTO enag_monthly (ym, mom_pct, kaynak) VALUES (?,?,?)",
                      (ym, pct, kaynak))

    @staticmethod
    def _monthly_rate(ym: str, table: dict) -> float:
        if ym in table:
            return table[ym][0]
        known = sorted(table.keys())
        last12 = [table[k][0] for k in known[-12:]] if known else [3.0]
        return float(np.mean(last12))

    @staticmethod
    def deflator(start, end) -> float:
        """start→end arası kümülatif ENAG enflasyon çarpanı (>= 1.0)."""
        if isinstance(start, str): start = datetime.strptime(start[:10], "%Y-%m-%d")
        if isinstance(end, str):   end   = datetime.strptime(end[:10], "%Y-%m-%d")
        if end <= start:
            return 1.0
        table  = InflationEngine.rates()
        factor = 1.0
        cur    = start
        while cur < end:
            # Ay sonu
            if cur.month == 12:
                month_end = datetime(cur.year + 1, 1, 1)
            else:
                month_end = datetime(cur.year, cur.month + 1, 1)
            seg_end   = min(end, month_end)
            days_used = (seg_end - cur).days
            days_in_m = (month_end - datetime(cur.year, cur.month, 1)).days
            r = InflationEngine._monthly_rate(cur.strftime("%Y-%m"), table) / 100.0
            factor *= (1.0 + r) ** (days_used / max(days_in_m, 1))
            cur = seg_end
        return factor

    @staticmethod
    def real_return(nominal_pct: float, start, end) -> float:
        """Nominal % getiriyi ENAG ile reel %'ye çevirir."""
        d = InflationEngine.deflator(start, end)
        return round(((1 + nominal_pct / 100.0) / d - 1) * 100, 2)


def _sector_of(ticker: str) -> str:
    """Sektör: önce el yapımı kategoriler, sonra TradingView sektörü, sonra 'Diğer'.

    Dinamik evren (~500 hisse) el listesinden çok daha geniş — TV sektörleri
    olmadan hepsi 'Diğer'e düşer ve sektör yoğunlaşma limiti anlamsızlaşırdı.
    """
    for cat, ts in BIST_STOCKS.items():
        if cat == "BIST 30":
            continue
        if ticker in ts:
            return cat
    if not _UNIVERSE_CACHE.get("sectors"):
        try:
            get_scan_universe()   # sektör haritasını doldurur (günde 1 TV çağrısı)
        except Exception:
            pass
    tv_sec = _UNIVERSE_CACHE.get("sectors", {}).get(ticker.upper().replace(".IS", ""))
    return tv_sec or "Diğer"


@st.cache_data(ttl=1800, show_spinner=False)
def compute_market_regime() -> dict:
    """Piyasa rejimi: XU100 trendi + piyasa genişliği + kur baskısı → 0-7 puan.

    Amaç 'gereksiz para kaybetmemek': Ayı rejiminde kısa vadeli alımlar frenlenir.
    """
    out = {"score": 0, "regime": "Belirsiz", "detay": [], "color": _theme()["muted"]}
    pts = 0
    try:
        xu = yf.download("XU100.IS", period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(xu.columns, pd.MultiIndex):
            xu.columns = xu.columns.get_level_values(0)
        c = xu["Close"].dropna()
        if len(c) >= 200:
            sma50  = float(c.rolling(50).mean().iloc[-1])
            sma200 = float(c.rolling(200).mean().iloc[-1])
            last   = float(c.iloc[-1])
            mom1m  = (last / float(c.iloc[-21]) - 1) * 100 if len(c) >= 21 else 0
            if last > sma200: pts += 2; out["detay"].append("XU100 SMA200 üstünde ✓")
            else:             out["detay"].append("XU100 SMA200 ALTINDA ✗")
            if sma50 > sma200: pts += 1; out["detay"].append("Golden cross aktif ✓")
            if mom1m > 0:      pts += 1; out["detay"].append(f"1 aylık momentum %{mom1m:+.1f} ✓")
            else:              out["detay"].append(f"1 aylık momentum %{mom1m:+.1f} ✗")
            out["xu100_mom1m"] = round(mom1m, 1)
    except Exception as exc:
        out["detay"].append(f"XU100 verisi alınamadı: {exc}")

    # Piyasa genişliği — BIST30 içinde SMA200 üstündeki hisse oranı
    try:
        b30 = [t + ".IS" for t in BIST_STOCKS["BIST 30"][:30]]
        bulk = yf.download(b30, period="1y", interval="1d",
                           auto_adjust=True, progress=False, group_by="ticker")
        above = total = 0
        for t in b30:
            try:
                cl = bulk[t]["Close"].dropna() if isinstance(bulk.columns, pd.MultiIndex) else bulk["Close"].dropna()
                if len(cl) >= 200:
                    total += 1
                    if float(cl.iloc[-1]) > float(cl.rolling(200).mean().iloc[-1]):
                        above += 1
            except Exception:
                continue
        if total >= 10:
            breadth = above / total * 100
            out["breadth"] = round(breadth, 0)
            if breadth >= 60: pts += 2; out["detay"].append(f"Genişlik: hisselerin %{breadth:.0f}'i SMA200 üstünde ✓")
            elif breadth >= 40: pts += 1; out["detay"].append(f"Genişlik: %{breadth:.0f} (nötr)")
            else: out["detay"].append(f"Genişlik ZAYIF: sadece %{breadth:.0f} SMA200 üstünde ✗")
    except Exception:
        pass

    # Kur baskısı — USDTRY 1 aylık değişim
    try:
        fx = yf.download("TRY=X", period="3mo", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        fc = fx["Close"].dropna()
        if len(fc) >= 21:
            fx1m = (float(fc.iloc[-1]) / float(fc.iloc[-21]) - 1) * 100
            out["usdtry_1m"] = round(fx1m, 1)
            if fx1m < 4.0: pts += 1; out["detay"].append(f"USDTRY sakin (%{fx1m:+.1f}/ay) ✓")
            else: out["detay"].append(f"USDTRY BASKILI (%{fx1m:+.1f}/ay) ✗")
    except Exception:
        pass

    out["score"] = pts
    if pts >= 5:   out["regime"], out["color"] = "Boğa",  "#1d6f4e"
    elif pts >= 3: out["regime"], out["color"] = "Nötr",  "#a2701d"
    else:          out["regime"], out["color"] = "Ayı",   "#9e2b25"
    return out


# Portföy kimlikleri — karakterli isim + ikon + strateji açıklaması
PORTFOLIO_META = {
    ("kisa", "Temkinli"): {"ad": "Nöbetçi", "ikon": "🛡️",
        "desc": "Kısa vadede yalnızca düşük oynaklıklı, trendi net hisselere girer; az işlem, sıkı stop. "
                "Sermaye korumayı önceleyen, ekrana az bakan yatırımcıya uygun."},
    ("kisa", "Dengeli"): {"ad": "Sörfçü", "ikon": "🏄",
        "desc": "Momentum dalgasını yakalayıp 1-3 haftada kâr almayı hedefler. Orta risk; "
                "piyasayı haftada birkaç kez kontrol edebilene uygun."},
    ("kisa", "Agresif"): {"ad": "Şimşek", "ikon": "⚡",
        "desc": "Yüksek momentum, hızlı al-sat, geniş evren. Sert dalgalanmaya dayanıklı, "
                "aktif takip yapabilen deneyimli yatırımcı için."},
    ("orta", "Temkinli"): {"ad": "Kaptan", "ikon": "⚓",
        "desc": "1-3 aylık ufukta düşük oynaklıklı, uzun vadeli trendi sağlam hisselerle sakin seyir. "
                "'Az ama öz' diyen yatırımcıya uygun."},
    ("orta", "Dengeli"): {"ad": "Pusula", "ikon": "🧭",
        "desc": "Sistemin ana stratejisi: SMA200 üstü + pozitif momentum + sektör çeşitlemesi, 1-3 ay pozisyon. "
                "Çoğu yatırımcı için dengeli başlangıç noktası."},
    ("orta", "Agresif"): {"ad": "Rüzgar", "ikon": "🌪️",
        "desc": "Güçlü 3 aylık momentumun peşinde geniş evrende avlanır. Getiri potansiyeli de "
                "dalgalanması da yüksek."},
    ("uzun", "Temkinli"): {"ad": "Çınar", "ikon": "🌳",
        "desc": "6-12 ay ufkunda golden cross + düşük oynaklık: yavaş ama istikrarlı büyüme. "
                "Enflasyona karşı sabırlı birikim yapana uygun."},
    ("uzun", "Dengeli"): {"ad": "Maraton", "ikon": "🏃",
        "desc": "Uzun soluklu trend takibi: 52 hafta gücü + hacim onayı. Pozisyonu aylarca "
                "taşıyabilen yatırımcı için."},
    ("uzun", "Agresif"): {"ad": "Zirve Avcısı", "ikon": "🏔️",
        "desc": "Yıllık zirvesine yakın seyreden liderleri tutar — kalibrasyonun en güçlü sinyali. "
                "Momentum kırılırsa sert düşüş riskini kabul edene göre."},
    ("sektor", "*"): {"ad": "Sektör Odak", "ikon": "🎯",
        "desc": "Tek sektöre konsantre portföy — sektör rotasyonunda güçlenen tarafa oynar. "
                "Çeşitlendirme azdır, sektör görüşü olan yatırımcıya uygun."},
    ("trend", "*"): {"ad": "Trend Avcısı", "ikon": "🚀",
        "desc": "Rotasyon haritasının 'Lider' çeyreğindeki sektörlerden en güçlü hisseleri toplar — "
                "paranın şu an aktığı yere biner. Rejim dönüşlerinde hızlı çıkış gerekir."},
    ("kontrol", "Rastgele"): {"ad": "Kontrol: Kura", "ikon": "🎲",
        "desc": "BİLİMSEL KIYAS — evrenden kurayla 6 hisse, eşit ağırlık, hiçbir sinyal yok. "
                "Sistem bunu yenemiyorsa skorların değeri yoktur."},
    ("kontrol", "BIST30"): {"ad": "Kontrol: Pasif BIST-30", "ikon": "📇",
        "desc": "BİLİMSEL KIYAS — BIST-30'un tamamı eşit ağırlık; 'hiç uğraşmasaydım' alternatifi. "
                "Sistemin pasif yatırıma karşı katma değer ölçütü."},
}


def _pm_meta(horizon: str, profile: str) -> dict:
    key = (horizon.split(":")[0], profile)
    m = PORTFOLIO_META.get(key) or PORTFOLIO_META.get((key[0], "*"))
    return m or {"ad": f"{horizon}/{profile}", "ikon": "📁", "desc": ""}


# Hisse logoları — bilinen şirketlerde gerçek favicon, kalanında renkli monogram rozet
BIST_LOGO_DOMAINS = {
    "THYAO": "turkishairlines.com",  "GARAN": "garantibbva.com.tr", "AKBNK": "akbank.com",
    "ISCTR": "isbank.com.tr",        "YKBNK": "yapikredi.com.tr",   "VAKBN": "vakifbank.com.tr",
    "HALKB": "halkbank.com.tr",      "QNBFB": "qnb.com.tr",         "SKBNK": "sekerbank.com.tr",
    "ALBRK": "albaraka.com.tr",      "SISE": "sisecam.com.tr",      "KCHOL": "koc.com.tr",
    "SAHOL": "sabanci.com",          "TCELL": "turkcell.com.tr",    "TTKOM": "turktelekom.com.tr",
    "BIMAS": "bim.com.tr",           "MGROS": "migros.com.tr",      "SOKM": "sokmarket.com.tr",
    "ASELS": "aselsan.com",          "FROTO": "fordotosan.com.tr",  "TOASO": "tofas.com.tr",
    "ARCLK": "arcelik.com.tr",       "VESTL": "vestel.com.tr",      "VESBE": "vestel.com.tr",
    "TUPRS": "tupras.com.tr",        "PETKM": "petkim.com.tr",      "EREGL": "erdemir.com.tr",
    "KRDMD": "kardemir.com",         "PGSUS": "flypgs.com",         "TAVHL": "tav.aero",
    "ENKAI": "enka.com",             "EKGYO": "emlakkonut.com.tr",  "SASA": "sasa.com.tr",
    "GUBRF": "gubretas.com.tr",      "AKSEN": "aksa.com.tr",        "ENJSA": "enerjisa.com.tr",
    "ULKER": "ulker.com.tr",         "LOGO": "logo.com.tr",         "OTKAR": "otokar.com.tr",
    "TTRAK": "turktraktor.com.tr",   "CLEBI": "celebiaviation.com", "DOHOL": "doganholding.com.tr",
    "TKFEN": "tekfen.com.tr",        "KOZAL": "kozaaltin.com.tr",   "ODAS": "odasenerji.com.tr",
    "ZOREN": "zorluenerji.com.tr",   "AYGAZ": "aygaz.com.tr",       "MAVI": "mavi.com",
    "LCWGR": "lcw.com",              "NETAS": "netas.com.tr",       "INDES": "indeks.com.tr",
    "KAREL": "karel.com.tr",         "CEMTS": "cemtas.com.tr",      "BIMAS2": "bim.com.tr",
    "TURSG": "turkiyesigorta.com.tr","THY": "turkishairlines.com",
}


def _logo_url(ticker: str) -> str:
    """Hisse logosu: bilinen domain'de Google favicon servisi, yoksa monogram SVG rozet."""
    t = ticker.upper().replace(".IS", "")
    dom = BIST_LOGO_DOMAINS.get(t)
    if dom:
        return f"https://www.google.com/s2/favicons?domain={dom}&sz=64"
    # Deterministik renkli monogram (harici servis yok, her zaman çalışır)
    hue = sum(ord(c) * 37 for c in t) % 360
    from urllib.parse import quote as _q
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'>"
           f"<rect width='64' height='64' rx='14' fill='hsl({hue},55%,40%)'/>"
           f"<text x='32' y='41' font-size='21' font-family='Arial' font-weight='bold' "
           f"fill='white' text-anchor='middle'>{t[:3]}</text></svg>")
    return "data:image/svg+xml;utf8," + _q(svg)


def _with_logo_col(df: pd.DataFrame, ticker_col: str = "Hisse") -> pd.DataFrame:
    """DataFrame'in başına Logo kolonu ekler (st.column_config.ImageColumn ile kullan)."""
    if df.empty or ticker_col not in df.columns:
        return df
    df = df.copy()
    df.insert(0, "Logo", df[ticker_col].map(_logo_url))
    return df


_LOGO_COL_CFG = {"Logo": st.column_config.ImageColumn("", width="small")}


@st.cache_data(ttl=300, show_spinner=False)
def _price_snapshot(tickers: tuple) -> dict:
    """Pozisyon tabloları için güncel fiyat + günlük/aylık değişim (5 dk cache).

    Dönüş: {ticker: {"son": float, "gun_pct": float, "ay_pct": float}}
    """
    out = {}
    if not tickers:
        return out
    try:
        syms = [t + ".IS" for t in tickers]
        bulk = yf.download(syms, period="2mo", interval="1d",
                           auto_adjust=True, progress=False, group_by="ticker")
        for t in tickers:
            try:
                s = t + ".IS"
                cl = bulk[s]["Close"].dropna() if isinstance(bulk.columns, pd.MultiIndex) else bulk["Close"].dropna()
                if len(cl) < 2:
                    continue
                son  = float(cl.iloc[-1])
                gun  = (son / float(cl.iloc[-2]) - 1) * 100
                ay_i = max(0, len(cl) - 22)
                ay   = (son / float(cl.iloc[ay_i]) - 1) * 100
                out[t] = {"son": round(son, 2), "gun_pct": round(gun, 2), "ay_pct": round(ay, 2)}
            except Exception:
                continue
    except Exception as exc:
        log.warning("Fiyat anlık görüntüsü hatası: %s", exc)
    return out


def _render_position_table(pos: list):
    """Pozisyon tablosu: logo + güncel fiyat + günlük/aylık/girişten değişim (renkli)."""
    if not pos:
        st.info("Pozisyon kaydı yok.")
        return
    t = _theme()
    snap = _price_snapshot(tuple(sorted({x["ticker"] for x in pos})))
    rows = []
    for x in pos:
        s = snap.get(x["ticker"], {})
        son = s.get("son")
        giris = x["entry_price"]
        rows.append({
            "Logo": _logo_url(x["ticker"]),
            "Hisse": x["ticker"],
            "Sektör": _sector_of(x["ticker"]),
            "Giriş": round(giris, 2),
            "Güncel": son,
            "Girişten %": round((son / giris - 1) * 100, 2) if son and giris else None,
            "Günlük %": s.get("gun_pct"),
            "Aylık %": s.get("ay_pct"),
            "Ağırlık %": x["weight"],
            "Stop": x["stop_price"],
            "Hedef": x["target_price"],
        })
    df = pd.DataFrame(rows)

    def _renk(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return f"color: {t['up']}" if v > 0 else (f"color: {t['down']}" if v < 0 else "")

    styled = df.style.map(_renk, subset=["Girişten %", "Günlük %", "Aylık %"]) \
                     .format({c: "{:+.2f}" for c in ["Girişten %", "Günlük %", "Aylık %"]},
                             na_rep="—") \
                     .format({c: "{:.2f}" for c in ["Giriş", "Güncel", "Stop", "Hedef"]}, na_rep="—")
    st.dataframe(styled, use_container_width=True, hide_index=True,
                 column_config=_LOGO_COL_CFG)


# Strateji rozetleri — emoji yerine hisse logolarıyla aynı dilde SVG rozet
_HORIZON_HUE = {"kisa": "#b45309", "orta": "#27509e", "uzun": "#1d6f4e",
                "sektor": "#6d3f8e", "trend": "#a2701d", "kontrol": "#6b6357"}


def _strategy_badge_uri(horizon: str, profile: str) -> str:
    """Vade rengi + profil harfli yuvarlak köşeli SVG rozet (data URI)."""
    h = horizon.split(":")[0]
    renk = _HORIZON_HUE.get(h, "#6b6357")
    kisaltma = {"kisa": "K", "orta": "O", "uzun": "U", "sektor": "S", "trend": "T"}.get(h, "?")
    harf = (profile or "?")[0].upper()
    from urllib.parse import quote as _q
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'>"
           f"<rect width='64' height='64' rx='16' fill='{renk}'/>"
           f"<text x='32' y='42' font-size='24' font-family='Georgia,serif' font-weight='bold' "
           f"fill='white' text-anchor='middle'>{kisaltma}{harf}</text></svg>")
    return "data:image/svg+xml;utf8," + _q(svg)


class PortfolioManager:
    """Vade × yatırımcı profili matrisinden portföy önerir, kaydeder, izler.

    Performans defteri: her kayıtlı portföyün NAV'ı (100 bazlı) günlük yazılır;
    nominal, ENAG-reel ve XU100-göreli getiri buradan ölçülür.
    """

    PROFILES = {
        "Temkinli": {"max_pos": 5, "max_atr": 3.0, "sector_cap": 1,
                     "aciklama": "Düşük oynaklık, az sayıda büyük hisse, sıkı stop"},
        "Dengeli":  {"max_pos": 6, "max_atr": 4.5, "sector_cap": 2,
                     "aciklama": "Orta oynaklık, sektör çeşitliliği"},
        "Agresif":  {"max_pos": 8, "max_atr": 99.0, "sector_cap": 2,
                     "aciklama": "Yüksek momentum toleransı, geniş evren"},
    }
    HORIZONS = {
        "Kısa (1-3 hafta)":  "kisa",
        "Orta (1-3 ay)":     "orta",
        "Uzun (6-12 ay)":    "uzun",
    }
    # 2026-07 kalibrasyonu (analyze-stops) — değişiklikler git geçmişinde,
    # her aylık koşuda calib_history'ye de yazılır (geri dönüş garantisi)
    STOP_MULTS   = {"kisa": 2.5, "orta": 3.5, "uzun": 3.5}
    TARGET_MULTS = {"kisa": 3.5, "orta": 6.0, "uzun": 9.0}

    _initialized = False

    @staticmethod
    def _init_tables():
        # _initialized sınıf özniteliği her Streamlit rerun'ında sıfırlanır
        # (script baştan çalışır) — session_state rerun'lar arası kalıcıdır,
        # yoksa her tıklamada Turso'ya CREATE TABLE turu atılır.
        try:
            if st.session_state.get("_pm_db_init"):
                return
        except Exception:
            pass   # Streamlit dışı bağlam (robot) — sınıf bayrağıyla devam
        if PortfolioManager._initialized:
            return
        _PMDB.execute_batch([
            ("""
                CREATE TABLE IF NOT EXISTS pm_portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT, horizon TEXT, profile TEXT,
                    created_at TEXT, status TEXT DEFAULT 'aktif',
                    regime_at_start TEXT,
                    kind TEXT DEFAULT 'kullanici'
                )
            """, ()),
            ("""
                CREATE TABLE IF NOT EXISTS pm_positions (
                    pid INTEGER, ticker TEXT, entry_price REAL, weight REAL,
                    stop_price REAL, target_price REAL,
                    PRIMARY KEY (pid, ticker)
                )
            """, ()),
            ("""
                CREATE TABLE IF NOT EXISTS pm_nav (
                    pid INTEGER, date TEXT, nav REAL, xu100_close REAL,
                    PRIMARY KEY (pid, date)
                )
            """, ()),
        ])
        # Eski tabloya kind kolonu ekle — idempotent: önce var mı diye bak
        try:
            cols = _PMDB.execute("SELECT name FROM pragma_table_info('pm_portfolios')")["rows"]
            if not any(c.get("name") == "kind" for c in cols):
                _PMDB.execute("ALTER TABLE pm_portfolios ADD COLUMN kind TEXT DEFAULT 'kullanici'")
        except Exception as exc:
            log.warning("pm_portfolios kind migration atlandı: %s", exc)
        PortfolioManager._initialized = True
        try:
            st.session_state["_pm_db_init"] = True
        except Exception:
            pass

    # PORTFÖY ÖNERİSİ

    @staticmethod
    def propose(scan_results: list, horizon: str, profile_name: str, regime: dict) -> tuple:
        """Returns (picks: list[dict], warning: str)."""
        prof = PortfolioManager.PROFILES[profile_name]
        valid = [r for r in scan_results
                 if not r.error and r.data_rows >= 200 and r.current_price > 0
                 and r.volume_ok and r.atr_pct <= prof["max_atr"]]

        warning = ""
        if regime.get("regime") == "Ayı":
            if horizon == "kisa":
                return [], ("Piyasa AYI rejiminde — kısa vadeli alım önerilmez. "
                            "Sermayeyi korumak da bir pozisyondur; rejim dönene kadar nakit/kısa vadeli mevduat.")
            warning = "Piyasa AYI rejiminde: pozisyon boyutlarını yarıya indirmek ve stopları sıkmak önerilir."

        # Vadeye göre filtre + sıralama anahtarı
        if horizon == "kisa":
            pool = [r for r in valid if r.adx >= 25 and r.macd_bullish
                    and r.momentum_1m > 0 and r.score >= 57]
            keyf = lambda r: r.score + r.momentum_1m * 0.8 + (r.volume_ratio - 1) * 8
        elif horizon == "orta":
            pool = [r for r in valid if r.price_above_sma200 and r.score >= 50
                    and r.momentum_3m > -3]
            keyf = lambda r: r.score + r.momentum_3m * 0.4 + (10 if r.golden_cross else 0)
        else:  # uzun
            pool = [r for r in valid if r.golden_cross and r.price_above_sma200
                    and r.week52_pos >= 0.45]
            keyf = lambda r: r.score + r.week52_pos * 20 + (5 if r.obv_trend == "yukari" else 0) - r.atr_pct * 2

        # Katsayılar 2026-07 kalibrasyonuyla güncellendi (weight_calibration.py
        # analyze-stops, 2730 giriş olayı): dar stoplar gürültüye takılıyordu
        # (eski kısa 1.5×ATR tüm kombinasyonlar içinde SONUNCUydu), hedefte
        # otomatik satış momentum kazananlarını erken kesiyordu. Geniş stop =
        # felaket sigortası kalır; hedef artık bilgi seviyesidir (alarm önerir,
        # otomatik çıkış yok). Ayı piyasası deneyimi henüz veri setinde yok —
        # rejim kapısı + portföy freni bu açığı kapatan diğer katmanlar.
        stop_mult   = PortfolioManager.STOP_MULTS[horizon]
        target_mult = PortfolioManager.TARGET_MULTS[horizon]
        picks = PortfolioManager._format_picks(pool, keyf, prof["max_pos"],
                                               prof["sector_cap"], stop_mult, target_mult)
        # Minimum çeşitleme koruması: filtrelerden 3'ten az hisse geçtiyse
        # portföy KURULMAZ. Tek hisseye %100 ağırlık "portföy" değildir —
        # W29 kisa/Temkinli gölgesi tek hisse TUPRS ile kaydolmuştu; o hafta
        # +%16 yazdı ama bu temkinlilik değil konsantrasyon kumarıdır.
        MIN_POS = 3
        if 0 < len(picks) < MIN_POS:
            adlar = ", ".join(p["ticker"] for p in picks)
            return [], (f"Filtrelerden yalnız {len(picks)} hisse geçti ({adlar}) — "
                        f"{MIN_POS} hisseden az portföy önerilmez (tek hisseye yığılma riski). "
                        "Bu vade/profil kombinasyonu için bu hafta beklemek daha sağlıklı.")
        return picks, warning

    # Günlük kapanış serisi cache'i (korelasyon filtresi için — günde 1 indirme)
    _closes_cache = {"date": None, "series": {}}

    @staticmethod
    def _fetch_closes(tickers: list, period: str = "6mo") -> dict:
        """Korelasyon hesabı için kapanış serileri (gün içi cache'li, toplu indirme)."""
        today = datetime.now().strftime("%Y-%m-%d")
        cache = PortfolioManager._closes_cache
        if cache["date"] != today:
            PortfolioManager._closes_cache = {"date": today, "series": {}}
            cache = PortfolioManager._closes_cache
        missing = [t for t in tickers if t not in cache["series"]]
        if missing:
            try:
                syms = [t + ".IS" for t in missing]
                bulk = yf.download(syms, period=period, interval="1d",
                                   auto_adjust=True, progress=False, group_by="ticker")
                for t in missing:
                    try:
                        s = t + ".IS"
                        cl = bulk[s]["Close"].dropna() if isinstance(bulk.columns, pd.MultiIndex) else bulk["Close"].dropna()
                        if len(cl) >= 60:
                            cache["series"][t] = cl.pct_change().dropna()
                    except Exception:
                        continue
            except Exception as exc:
                log.warning("Korelasyon serisi indirilemedi: %s", exc)
        return {t: cache["series"][t] for t in tickers if t in cache["series"]}

    CORR_LIMIT = 0.85   # bu eşiğin üstü "aynı bahis" sayılır, çeşitleme sahtedir

    @staticmethod
    def _format_picks(pool: list, keyf, max_pos: int, sector_cap: int,
                      stop_mult: float, target_mult: float) -> list:
        """Havuzdan sıralı seçim + risk katmanı:
        - Sektör limiti (mevcut)
        - KORELASYON eleme: seçilmiş bir hisseyle günlük getiri korelasyonu
          > 0.85 olan aday atlanır (üç banka almak çeşitleme değildir)
        - ATR-ters-orantılı ağırlık: oynak hisseye az, sakin hisseye çok
          (eşit ağırlık yerine risk dengeli dağılım; tek pozisyon max %30)
        """
        pool = sorted(pool, key=keyf, reverse=True)
        # Korelasyon serilerini aday havuzunun başı için hazırla (tek toplu indirme)
        returns = PortfolioManager._fetch_closes([r.ticker for r in pool[:max_pos * 3]])

        picks, sector_count, kept_returns = [], {}, {}
        for r in pool:
            sec = _sector_of(r.ticker)
            if sector_cap > 0 and sector_count.get(sec, 0) >= sector_cap:
                continue
            # Korelasyon kontrolü
            cand = returns.get(r.ticker)
            too_correlated = False
            if cand is not None:
                for kt, ks in kept_returns.items():
                    try:
                        c = cand.corr(ks)
                        if pd.notna(c) and c > PortfolioManager.CORR_LIMIT:
                            too_correlated = True
                            log.info("Korelasyon eleme: %s ~ %s (r=%.2f)", r.ticker, kt, c)
                            break
                    except Exception:
                        continue
            if too_correlated:
                continue

            atr_abs = r.current_price * r.atr_pct / 100.0
            picks.append({
                "ticker": r.ticker, "sektor": sec,
                "fiyat": round(r.current_price, 2),
                "skor": round(r.score, 1),
                "atr_pct": round(r.atr_pct, 2),
                "stop": round(r.current_price - stop_mult * atr_abs, 2),
                "hedef": round(r.current_price + target_mult * atr_abs, 2),
                "gerekce": f"Skor {r.score:.0f} | ADX {r.adx:.0f} | 3A %{r.momentum_3m:+.0f} | 52H %{r.week52_pos*100:.0f}",
            })
            sector_count[sec] = sector_count.get(sec, 0) + 1
            if cand is not None:
                kept_returns[r.ticker] = cand
            if len(picks) >= max_pos:
                break

        # ATR-ters-orantılı ağırlık (risk paritesi hafif versiyonu)
        if picks:
            inv = [1.0 / max(p["atr_pct"], 0.5) for p in picks]
            tot = sum(inv)
            ws = [x / tot * 100.0 for x in inv]
            # Tek pozisyon tavanı %30 — fazlası diğerlerine orantılı dağıtılır
            for _ in range(3):
                over = sum(max(0.0, w - 30.0) for w in ws)
                if over <= 0.01:
                    break
                ws = [min(w, 30.0) for w in ws]
                under_ix = [i for i, w in enumerate(ws) if w < 30.0]
                under_tot = sum(ws[i] for i in under_ix)
                for i in under_ix:
                    ws[i] += over * (ws[i] / under_tot) if under_tot > 0 else 0
            for p, w in zip(picks, ws):
                p["agirlik"] = round(w, 1)
        return picks

    @staticmethod
    def propose_sector(scan_results: list, sector: str, profile_name: str) -> tuple:
        """Tek sektöre odaklı portföy — sektör görüşü olan yatırımcı için."""
        prof = PortfolioManager.PROFILES[profile_name]
        pool = [r for r in scan_results
                if not r.error and r.data_rows >= 200 and r.current_price > 0
                and r.volume_ok and _sector_of(r.ticker) == sector
                and r.score >= 50 and r.price_above_sma200
                and r.atr_pct <= prof["max_atr"] + 1.0]  # tek sektörde biraz tolerans
        picks = PortfolioManager._format_picks(
            pool, lambda r: r.score + r.momentum_3m * 0.3,
            max_pos=prof["max_pos"], sector_cap=0,  # sektör limiti yok — zaten tek sektör
            stop_mult=3.5, target_mult=6.0)  # kalibrasyon 2026-07: orta vade ayarları
        warn = ("Tek sektör portföyü: çeşitlendirme düşük, sektör haberlerine duyarlılık yüksek. "
                "Toplam varlığın küçük bir bölümüyle sınırlamak sağlıklıdır.")
        return picks, warn

    @staticmethod
    def propose_trend(scan_results: list, profile_name: str) -> tuple:
        """Güncel trend portföyü: rotasyonda 'Lider' çeyrekteki sektörlerin en güçlüleri."""
        prof = PortfolioManager.PROFILES[profile_name]
        valid = [r for r in scan_results
                 if not r.error and r.data_rows >= 200 and r.current_price > 0 and r.volume_ok]
        # Sektör momentumları
        sec_groups = {}
        for r in valid:
            sec_groups.setdefault(_sector_of(r.ticker), []).append(r)
        leaders = [sec for sec, rs in sec_groups.items()
                   if len(rs) >= 2
                   and np.mean([x.momentum_1m for x in rs]) > 0
                   and np.mean([x.momentum_3m for x in rs]) > 0]
        if not leaders:
            return [], ("Şu an rotasyonda 'Lider' çeyrekte sektör yok — trend portföyü için "
                        "uygun zaman değil. Sektörler toparlanınca tekrar deneyin.")
        pool = [r for r in valid
                if _sector_of(r.ticker) in leaders
                and r.week52_pos >= 0.55 and r.momentum_3m > 0 and r.score >= 55]
        picks = PortfolioManager._format_picks(
            pool, lambda r: r.score + r.momentum_1m * 0.5 + r.week52_pos * 15,
            max_pos=prof["max_pos"], sector_cap=max(prof["sector_cap"], 2),
            stop_mult=2.5, target_mult=4.5)  # kalibrasyon 2026-07: trend döner, kısa-orta arası
        warn = f"Lider sektörler: {', '.join(leaders)}. Trend döndüğünde hızlı çıkış gerekir — stoplara uy."
        return picks, warn

    # KAYIT & NAV TAKİBİ

    @staticmethod
    def save_portfolio(name: str, horizon: str, profile: str, picks: list, regime: str,
                       kind: str = "kullanici") -> int:
        PortfolioManager._init_tables()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        res = _PMDB.execute("""
            INSERT INTO pm_portfolios (name, horizon, profile, created_at, regime_at_start, kind)
            VALUES (?,?,?,?,?,?)
        """, (name, horizon, profile, now, regime, kind))
        pid = res["lastrowid"]
        stmts = [("""
            INSERT OR REPLACE INTO pm_positions (pid, ticker, entry_price, weight, stop_price, target_price)
            VALUES (?,?,?,?,?,?)
        """, (pid, p["ticker"], p["fiyat"], p["agirlik"], p["stop"], p["hedef"])) for p in picks]
        # Gün-0 NAV kaydı
        # Gün-0 NAV 99.8: %0.2 giriş maliyeti (tek yön) — defter sürtünmesiz olmasın.
        # Kontroller dahil herkese eşit uygulanır; karşılaştırma adil kalır.
        stmts.append(("INSERT OR REPLACE INTO pm_nav (pid, date, nav, xu100_close) VALUES (?,?,?,?)",
                      (pid, datetime.now().strftime("%Y-%m-%d"), 99.8,
                       PortfolioManager._xu100_close())))
        _PMDB.execute_batch(stmts)
        return pid

    @staticmethod
    def _xu100_close() -> float:
        try:
            xu = yf.download("XU100.IS", period="5d", interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(xu.columns, pd.MultiIndex):
                xu.columns = xu.columns.get_level_values(0)
            return round(float(xu["Close"].dropna().iloc[-1]), 2)
        except Exception:
            return 0.0

    @staticmethod
    def active_portfolios() -> list:
        PortfolioManager._init_tables()
        return _PMDB.execute(
            "SELECT * FROM pm_portfolios WHERE status='aktif' ORDER BY created_at DESC")["rows"]

    @staticmethod
    def all_portfolios() -> list:
        PortfolioManager._init_tables()
        return _PMDB.execute("SELECT * FROM pm_portfolios ORDER BY created_at DESC")["rows"]

    @staticmethod
    def positions(pid: int) -> list:
        return _PMDB.execute("SELECT * FROM pm_positions WHERE pid=?", (pid,))["rows"]

    @staticmethod
    def update_navs():
        """Aktif portföylerin bugünkü NAV'ını yazar (oturum başına 1 kez çağrılır)."""
        PortfolioManager._init_tables()
        ports = PortfolioManager.active_portfolios()
        if not ports:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        pos_map = PortfolioManager.positions_all([p["id"] for p in ports])
        all_tickers = {x["ticker"] for pos in pos_map.values() for x in pos}
        if not all_tickers:
            return
        try:
            syms = [t + ".IS" for t in all_tickers]
            bulk = yf.download(syms, period="5d", interval="1d",
                               auto_adjust=True, progress=False, group_by="ticker")
            price = {}
            for t in all_tickers:
                try:
                    s = t + ".IS"
                    cl = bulk[s]["Close"].dropna() if isinstance(bulk.columns, pd.MultiIndex) else bulk["Close"].dropna()
                    if len(cl):
                        price[t] = float(cl.iloc[-1])
                except Exception:
                    continue
            xu = PortfolioManager._xu100_close()
            stmts = []
            for p in ports:
                pos = pos_map[p["id"]]
                nav, w_used = 0.0, 0.0
                for x in pos:
                    if x["ticker"] in price and x["entry_price"] > 0:
                        nav += x["weight"] * (price[x["ticker"]] / x["entry_price"])
                        w_used += x["weight"]
                if w_used > 0:
                    nav = nav / w_used * 100.0
                    stmts.append(("INSERT OR REPLACE INTO pm_nav (pid, date, nav, xu100_close) VALUES (?,?,?,?)",
                                  (p["id"], today, round(nav, 2), xu)))
            if stmts:
                _PMDB.execute_batch(stmts)
        except Exception as exc:
            log.warning("PortfolioManager.update_navs hatası: %s", exc)

    @staticmethod
    def nav_history(pid: int) -> pd.DataFrame:
        rows = _PMDB.execute(
            "SELECT date, nav, xu100_close FROM pm_nav WHERE pid=? ORDER BY date", (pid,))["rows"]
        return pd.DataFrame(rows, columns=["date", "nav", "xu100_close"])

    # TOPLU OKUMALAR — sayfa yüklemesinde portföy başına ayrı sorgu yerine
    # tek yolculuk (Turso'da her sorgu bir HTTP turu: 14 tur ~25 sn sürüyordu).
    # Veri yine her seferinde taze çekilir — önbellek yok, tazelik kaybı yok.

    @staticmethod
    def nav_histories(pids: list) -> dict:
        """Tüm portföylerin NAV geçmişi TEK sorguda: {pid: DataFrame}."""
        if not pids:
            return {}
        ph = ",".join("?" * len(pids))
        rows = _PMDB.execute(
            f"SELECT pid, date, nav, xu100_close FROM pm_nav WHERE pid IN ({ph}) ORDER BY pid, date",
            tuple(pids))["rows"]
        out = {pid: [] for pid in pids}
        for r in rows:
            out[r["pid"]].append(r)
        return {pid: pd.DataFrame(rs, columns=["pid", "date", "nav", "xu100_close"])
                     .drop(columns=["pid"]) if rs else
                     pd.DataFrame(columns=["date", "nav", "xu100_close"])
                for pid, rs in out.items()}

    @staticmethod
    def positions_all(pids: list) -> dict:
        """Tüm portföylerin pozisyonları TEK sorguda: {pid: [pozisyon, ...]}."""
        if not pids:
            return {}
        ph = ",".join("?" * len(pids))
        rows = _PMDB.execute(
            f"SELECT * FROM pm_positions WHERE pid IN ({ph})", tuple(pids))["rows"]
        out = {pid: [] for pid in pids}
        for r in rows:
            out[r["pid"]].append(r)
        return out

    @staticmethod
    def _perf_from_nav(p: dict, df: pd.DataFrame) -> dict:
        """performance() ile aynı hesap — hazır NAV verisiyle (ek sorgu yok)."""
        out = {"nominal": 0.0, "reel": 0.0, "xu100_rel": 0.0, "gun": 0}
        if len(df) < 1:
            return out
        start = p["created_at"][:10]
        end   = df["date"].iloc[-1]
        nominal = float(df["nav"].iloc[-1]) - 100.0
        out["nominal"] = round(nominal, 2)
        out["reel"]    = InflationEngine.real_return(nominal, start, end)
        out["gun"]     = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
        xu0 = float(df["xu100_close"].iloc[0]) if len(df) and float(df["xu100_close"].iloc[0] or 0) > 0 else 0
        xu1 = float(df["xu100_close"].iloc[-1] or 0)
        if xu0 > 0 and xu1 > 0:
            out["xu100_rel"] = round(nominal - (xu1 / xu0 - 1) * 100, 2)
        return out

    @staticmethod
    def intraday_navs(ports: list, poss: dict, navs: dict, days: int = 8) -> tuple:
        """Saatlik NAV yeniden inşası — grafikler 'dümdüz' kalmasın diye.

        pm_nav günde 1 nokta tutar; burada pozisyonlardan (giriş fiyatı +
        ağırlık) son N günün SAATLİK fiyatlarıyla NAV'ı yeniden hesaplarız.
        Baz, portföyün gün-0 NAV'ına oturtulur (giriş maliyeti korunur).
        Dönüş: ({pid: pd.Series}, xu100_series)
        """
        tickers = {x["ticker"] for pid in poss for x in poss.get(pid, [])}
        if not tickers:
            return {}, pd.Series(dtype=float)
        try:
            syms = [t + ".IS" for t in tickers] + ["XU100.IS"]
            bulk = yf.download(syms, period=f"{days}d", interval="60m",
                               auto_adjust=True, progress=False, group_by="ticker")
            price = {}
            for s in syms:
                try:
                    cl = bulk[s]["Close"].dropna() if isinstance(bulk.columns, pd.MultiIndex) else bulk["Close"].dropna()
                    if len(cl):
                        price[s[:-3] if s.endswith(".IS") else s] = cl
                except Exception:
                    continue
            xu = price.pop("XU100", pd.Series(dtype=float))
            out = {}
            for p in ports:
                pos = poss.get(p["id"], [])
                if not pos:
                    continue
                created = pd.Timestamp(p["created_at"][:10])
                nav0_rows = navs.get(p["id"])
                base = float(nav0_rows["nav"].iloc[0]) if nav0_rows is not None and len(nav0_rows) else 100.0
                seri, w_top = None, 0.0
                for x in pos:
                    cl = price.get(x["ticker"])
                    if cl is None or x["entry_price"] <= 0:
                        continue
                    katki = x["weight"] * (cl / x["entry_price"])
                    seri = katki if seri is None else seri.add(katki, fill_value=None)
                    w_top += x["weight"]
                if seri is None or w_top <= 0:
                    continue
                seri = (seri / w_top * base).dropna()
                seri = seri[seri.index.tz_localize(None) >= created] if seri.index.tz else seri[seri.index >= created]
                if len(seri) >= 2:
                    out[p["id"]] = seri
            return out, xu
        except Exception as exc:
            log.warning("Saatlik NAV inşası hatası: %s", exc)
            return {}, pd.Series(dtype=float)

    @staticmethod
    def risk_metrics(nav_df: pd.DataFrame) -> dict:
        """Kurumsal risk metrikleri (Roadmap-C). En az 10 NAV noktası ister.

        Sharpe/Sortino: günlük getirilerden yıllıklandırılmış (rf=0 varsayımı —
        TL risksiz getiri tartışmalı; sunumda belirtilir).
        IR (Information Ratio): XU100'e göre fazla getirinin tutarlılığı.
        """
        out = {"sharpe": None, "sortino": None, "maxdd": None, "ir": None,
               "n_gun": len(nav_df)}
        if len(nav_df) < 10:
            return out
        nav = nav_df["nav"].astype(float)
        r = nav.pct_change().dropna()
        if r.std() > 0:
            out["sharpe"] = round(float(r.mean() / r.std() * np.sqrt(252)), 2)
        downside = r[r < 0]
        if len(downside) > 1 and downside.std() > 0:
            out["sortino"] = round(float(r.mean() / downside.std() * np.sqrt(252)), 2)
        out["maxdd"] = round(float(((nav / nav.cummax()) - 1).min() * 100), 2)
        xu = nav_df["xu100_close"].astype(float)
        if (xu > 0).all():
            aktif = r - xu.pct_change().dropna()
            aktif = aktif.dropna()
            if len(aktif) > 2 and aktif.std() > 0:
                out["ir"] = round(float(aktif.mean() / aktif.std() * np.sqrt(252)), 2)
        return out

    @staticmethod
    def performances(ports: list) -> dict:
        """Tüm portföylerin performansı tek NAV sorgusuyla: {pid: perf}."""
        navs = PortfolioManager.nav_histories([p["id"] for p in ports])
        return {p["id"]: PortfolioManager._perf_from_nav(p, navs.get(p["id"], pd.DataFrame()))
                for p in ports}

    @staticmethod
    def performance(p: dict) -> dict:
        """Nominal, ENAG-reel ve XU100-göreli getiri."""
        df = PortfolioManager.nav_history(p["id"])
        out = {"nominal": 0.0, "reel": 0.0, "xu100_rel": 0.0, "gun": 0}
        if len(df) < 1:
            return out
        start = p["created_at"][:10]
        end   = df["date"].iloc[-1]
        nominal = float(df["nav"].iloc[-1]) - 100.0
        out["nominal"] = round(nominal, 2)
        out["reel"]    = InflationEngine.real_return(nominal, start, end)
        out["gun"]     = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
        xu0 = float(df["xu100_close"].iloc[0]) if float(df["xu100_close"].iloc[0]) > 0 else 0
        xu1 = float(df["xu100_close"].iloc[-1])
        if xu0 > 0 and xu1 > 0:
            out["xu100_rel"] = round(nominal - (xu1 / xu0 - 1) * 100, 2)
        return out

    @staticmethod
    def archive(pid: int):
        _PMDB.execute("UPDATE pm_portfolios SET status='arsiv' WHERE id=?", (pid,))

    # RİSK ALARMLARI — stop kırılımı, hedef, portföy freni
    # Günlük robot (daily_robot.py) ve PM sayfası tarafından paylaşılır.

    DRAWDOWN_LIMITS = {"Temkinli": 7.0, "Dengeli": 10.0, "Agresif": 15.0}  # % tepe-düşüş freni

    @staticmethod
    def _init_alerts():
        _PMDB.execute("""
            CREATE TABLE IF NOT EXISTS pm_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT, pid INTEGER, ticker TEXT,
                tip TEXT, mesaj TEXT, okundu INTEGER DEFAULT 0
            )
        """)

    @staticmethod
    def _recent_alert_exists(pid: int, ticker: str, tip: str, days: int = 7) -> bool:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = _PMDB.execute(
            "SELECT COUNT(*) AS c FROM pm_alerts WHERE pid=? AND ticker=? AND tip=? AND created_at >= ?",
            (pid, ticker, tip, cutoff))["rows"]
        return bool(rows and rows[0].get("c", 0) > 0)

    @staticmethod
    def check_risk_alerts() -> list:
        """Aktif portföyleri tarar: stop kırılımı / hedefe ulaşma / portföy freni.
        Yeni alarmları pm_alerts'e yazar (7 gün tekrar-yazma koruması). Dönüş: yeni alarmlar."""
        PortfolioManager._init_alerts()
        ports = PortfolioManager.active_portfolios()
        if not ports:
            return []
        pos_map = PortfolioManager.positions_all([p["id"] for p in ports])
        all_tickers = {x["ticker"] for pos in pos_map.values() for x in pos}
        # Son 7 günün alarmları TEK sorguda (pozisyon başına ayrı sorgu yerine)
        _cutoff7 = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        _recent = {(r["pid"], r["ticker"], r["tip"])
                   for r in _PMDB.execute(
                       "SELECT pid, ticker, tip FROM pm_alerts WHERE created_at >= ?",
                       (_cutoff7,))["rows"]}
        nav_map = PortfolioManager.nav_histories([p["id"] for p in ports])
        # Güncel fiyatlar (tek toplu indirme)
        price = {}
        try:
            syms = [t + ".IS" for t in all_tickers]
            bulk = yf.download(syms, period="5d", interval="1d",
                               auto_adjust=True, progress=False, group_by="ticker")
            for t in all_tickers:
                try:
                    s = t + ".IS"
                    cl = bulk[s]["Close"].dropna() if isinstance(bulk.columns, pd.MultiIndex) else bulk["Close"].dropna()
                    if len(cl):
                        price[t] = float(cl.iloc[-1])
                except Exception:
                    continue
        except Exception as exc:
            log.warning("Alarm fiyat indirme hatası: %s", exc)
            return []

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_alerts = []
        for p in ports:
            golge = " (gölge)" if p.get("kind") == "golge" else ""
            # 1) Pozisyon bazlı: stop / hedef
            for x in pos_map[p["id"]]:
                t, cur = x["ticker"], price.get(x["ticker"])
                if not cur:
                    continue
                if x["stop_price"] and cur <= x["stop_price"] and (p["id"], t, "STOP") not in _recent:
                    new_alerts.append((now, p["id"], t, "STOP",
                        f"🔴 {t}: stop kesildi ({cur:.2f} ≤ {x['stop_price']:.2f}) — '{p['name']}'{golge}. Zararı büyütmemek için çıkış değerlendirilmeli."))
                elif x["target_price"] and cur >= x["target_price"] and (p["id"], t, "HEDEF") not in _recent:
                    new_alerts.append((now, p["id"], t, "HEDEF",
                        f"🟡 {t}: hedef fiyata ulaşıldı ({cur:.2f} ≥ {x['target_price']:.2f}) — '{p['name']}'{golge}. Kısmi kâr alma düşünülebilir."))
            # 2) Portföy freni: tepe NAV'dan düşüş profil limitini aşarsa
            nav = nav_map.get(p["id"], pd.DataFrame())
            if len(nav) >= 2:
                peak = float(nav["nav"].max())
                last = float(nav["nav"].iloc[-1])
                dd = (peak - last) / peak * 100 if peak > 0 else 0
                limit = PortfolioManager.DRAWDOWN_LIMITS.get(p["profile"], 10.0)
                if dd >= limit and (p["id"], "*", "FREN") not in _recent:
                    new_alerts.append((now, p["id"], "*", "FREN",
                        f"⛔ '{p['name']}'{golge}: tepe değerden %{dd:.1f} düşüş ({p['profile']} limiti %{limit:.0f}). "
                        f"Tüm pozisyonları gözden geçir — rejim değişmiş olabilir."))

        if new_alerts:
            _PMDB.execute_batch([
                ("INSERT INTO pm_alerts (created_at, pid, ticker, tip, mesaj) VALUES (?,?,?,?,?)", a)
                for a in new_alerts])
            log.info("Risk alarmı: %d yeni", len(new_alerts))
        return [a[4] for a in new_alerts]

    @staticmethod
    def unread_alerts() -> list:
        PortfolioManager._init_alerts()
        return _PMDB.execute(
            "SELECT * FROM pm_alerts WHERE okundu=0 ORDER BY created_at DESC LIMIT 20")["rows"]

    @staticmethod
    def mark_alerts_read():
        _PMDB.execute("UPDATE pm_alerts SET okundu=1 WHERE okundu=0")

    # GÖLGE PORTFÖYLER — "Ne olurdu?" panosu
    # Sistem her hafta 9 kombinasyonun (3 vade × 3 profil) önerisini otomatik
    # kaydeder; kullanıcı hiç kaydetmese bile "sistemin önerileri gerçekte ne
    # kazandırdı?" sorusu tarafsız veriyle cevaplanır.

    SHADOW_LIMITS = {"kisa": 21, "orta": 90, "uzun": 365}  # gün — vade dolunca arşiv

    @staticmethod
    def ensure_shadow_batch(regime: dict, scan_results: list) -> int:
        """Bu haftanın gölge seti yoksa oluşturur. Dönüş: yeni kayıt sayısı."""
        PortfolioManager._init_tables()
        week = datetime.now().strftime("%G-W%V")
        rows = _PMDB.execute(
            "SELECT COUNT(*) AS c FROM pm_portfolios WHERE kind='golge' AND horizon != 'kontrol' AND name LIKE ?",
            (f"%{week}%",))["rows"]
        if rows and rows[0].get("c", 0) > 0:
            return 0
        created = 0
        for horizon in ("kisa", "orta", "uzun"):
            for profile in ("Temkinli", "Dengeli", "Agresif"):
                try:
                    picks, _warn = PortfolioManager.propose(scan_results, horizon, profile, regime)
                    if not picks:
                        continue  # rejim engeli / kriter dışı — o kombinasyon bu hafta boş
                    PortfolioManager.save_portfolio(
                        f"Gölge {horizon}/{profile} {week}",
                        horizon, profile, picks, regime.get("regime", "?"), kind="golge")
                    created += 1
                except Exception as exc:
                    log.warning("Gölge portföy oluşturma hatası (%s/%s): %s", horizon, profile, exc)
        return created

    @staticmethod
    def snapshot_tv_targets() -> int:
        """Haftalık analist hedef fiyat fotoğrafı → tv_target_history (Turso'da kalıcı).

        Amaç: prim bileşeninin IC testi için point-in-time veri arşivi kurmak.
        Tarihsel analist hedefi ücretsiz hiçbir kaynakta yok (Wayback dahil
        tarandı, 2026-07) — tek yol bugünden itibaren kendi arşivimizi
        biriktirmek. ~10-12 haftada ilk kaba IC okuması mümkün olur.
        """
        _PMDB.execute("""
            CREATE TABLE IF NOT EXISTS tv_target_history (
                date TEXT, ticker TEXT, target REAL, analyst_count INTEGER, close REAL,
                PRIMARY KEY (date, ticker)
            )
        """)
        week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")
        rows = _PMDB.execute("SELECT COUNT(*) AS c FROM tv_target_history WHERE date >= ?",
                             (week_start,))["rows"]
        if rows and rows[0].get("c", 0) > 0:
            return 0  # bu hafta çekilmiş
        try:
            from tradingview_screener import Query
            _n, df = (Query().set_markets("turkey")
                      .select("name", "close", "price_target_average", "recommendation_total")
                      .limit(700).get_scanner_data())
        except Exception as exc:
            log.warning("TV hedef fotoğrafı alınamadı: %s", exc)
            return 0
        today = datetime.now().strftime("%Y-%m-%d")
        stmts = []
        for _, r in df.iterrows():
            t, tgt, cnt = str(r.get("name") or ""), r.get("price_target_average"), r.get("recommendation_total")
            if t and pd.notna(tgt) and pd.notna(cnt) and float(cnt) >= 3:
                cl = float(r["close"]) if pd.notna(r.get("close")) else None
                stmts.append(("INSERT OR REPLACE INTO tv_target_history (date, ticker, target, analyst_count, close) VALUES (?,?,?,?,?)",
                              (today, t, float(tgt), int(cnt), cl)))
        if stmts:
            for i in range(0, len(stmts), 40):
                _PMDB.execute_batch(stmts[i:i + 40])
        log.info("TV hedef fotoğrafı: %d hisse kaydedildi (%s)", len(stmts), today)
        return len(stmts)

    @staticmethod
    def ensure_control_batch(scan_results: list) -> int:
        """Haftalık BİLİMSEL KONTROL portföyleri (Roadmap-B):
        - Kura: evrenden rastgele 6 hisse (hafta tohumlu, tekrarlanabilir)
        - Pasif BIST-30: eşit ağırlık
        Sinyalsiz kıyaslar — sistem bunları yenemiyorsa alfa iddiası boştur.
        Stop/hedef = 0 (kontroller yönetilmez, alarm üretmezler)."""
        PortfolioManager._init_tables()
        week = datetime.now().strftime("%G-W%V")
        rows = _PMDB.execute(
            "SELECT COUNT(*) AS c FROM pm_portfolios WHERE kind='golge' AND horizon='kontrol' AND name LIKE ?",
            (f"%{week}%",))["rows"]
        if rows and rows[0].get("c", 0) > 0:
            return 0
        created = 0
        fiyat = {r.ticker: r.current_price for r in scan_results
                 if not r.error and r.current_price > 0}
        # 1) Kura — hafta numarası tohum: aynı hafta yeniden koşulsa aynı seçim
        havuz = sorted(r.ticker for r in scan_results
                       if not r.error and r.data_rows >= 200 and r.volume_ok
                       and r.current_price > 0)
        if len(havuz) >= 6:
            rng = np.random.default_rng(int(week.replace("-W", "")))
            secim = list(rng.choice(havuz, 6, replace=False))
            picks = [{"ticker": t, "fiyat": fiyat[t], "agirlik": round(100/6, 1),
                      "stop": 0.0, "hedef": 0.0} for t in secim]
            PortfolioManager.save_portfolio(f"Kontrol Kura {week}", "kontrol",
                                            "Rastgele", picks, "-", kind="golge")
            created += 1
        # 2) Pasif BIST-30
        b30 = [t for t in BIST_STOCKS["BIST 30"] if t in fiyat]
        if len(b30) >= 20:
            w = round(100 / len(b30), 2)
            picks = [{"ticker": t, "fiyat": fiyat[t], "agirlik": w,
                      "stop": 0.0, "hedef": 0.0} for t in b30]
            PortfolioManager.save_portfolio(f"Kontrol BIST30 {week}", "kontrol",
                                            "BIST30", picks, "-", kind="golge")
            created += 1
        if created:
            log.info("Kontrol portföyleri oluşturuldu: %d (%s)", created, week)
        return created

    @staticmethod
    def auto_archive_shadows() -> int:
        """Vadesi dolan gölge portföyleri arşivler — aktif liste şişmesin."""
        archived = 0
        now = datetime.now()
        for p in PortfolioManager.active_portfolios():
            if p.get("kind") != "golge":
                continue
            try:
                age = (now - datetime.strptime(p["created_at"][:10], "%Y-%m-%d")).days
                if age > PortfolioManager.SHADOW_LIMITS.get(p["horizon"], 90):
                    PortfolioManager.archive(p["id"])
                    archived += 1
            except Exception:
                continue
        return archived

    @staticmethod
    def shadow_scoreboard() -> pd.DataFrame:
        """Kombinasyon karnesi: tüm gölge portföylerin (aktif+arşiv) ortalama
        nominal/reel/XU100-göreli getirisi, vade×profil bazında."""
        rows = []
        _golge = [p for p in PortfolioManager.all_portfolios() if p.get("kind") == "golge"]
        _perfs = PortfolioManager.performances(_golge)
        _navs  = PortfolioManager.nav_histories([p["id"] for p in _golge])
        for p in _golge:
            perf = _perfs[p["id"]]
            if perf["gun"] < 1:
                continue
            _rm = PortfolioManager.risk_metrics(_navs.get(p["id"], pd.DataFrame()))
            rows.append({"Vade": p["horizon"], "Profil": p["profile"],
                         "nominal": perf["nominal"], "reel": perf["reel"],
                         "xu": perf["xu100_rel"], "gun": perf["gun"],
                         "sharpe": _rm["sharpe"], "maxdd": _rm["maxdd"]})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        agg = (df.groupby(["Vade", "Profil"])
                 .agg(Portfoy=("nominal", "size"),
                      Ort_Nominal=("nominal", "mean"),
                      Ort_Reel=("reel", "mean"),
                      Ort_XU100_Rel=("xu", "mean"),
                      Ort_Sharpe=("sharpe", "mean"),
                      Ort_MaxDD=("maxdd", "mean"),
                      Ort_Gun=("gun", "mean"))
                 .round(2).reset_index())
        return agg.sort_values("Ort_Reel", ascending=False)

    # AKTİF REBALANS

    @staticmethod
    def rebalance_suggestions(p: dict, scan_results: list) -> list:
        """Skoru bozulan/stop kesen pozisyonlar için ÇIKAR, güçlü adaylar için EKLE önerir."""
        pos = PortfolioManager.positions(p["id"])
        held = {x["ticker"] for x in pos}
        by_ticker = {r.ticker: r for r in scan_results if not r.error}
        suggestions = []
        for x in pos:
            r = by_ticker.get(x["ticker"])
            if r is None:
                continue
            if r.current_price <= x["stop_price"]:
                suggestions.append({"tip": "ÇIKAR", "ticker": x["ticker"],
                                    "neden": f"Stop seviyesi kesildi ({r.current_price:.2f} ≤ {x['stop_price']:.2f}) — zararı büyütme"})
            elif r.score < 43:
                suggestions.append({"tip": "ÇIKAR", "ticker": x["ticker"],
                                    "neden": f"Skor {r.score:.0f}'a düştü (SAT bölgesi)"})
            elif x["target_price"] > 0 and r.current_price >= x["target_price"]:
                suggestions.append({"tip": "KÂR AL", "ticker": x["ticker"],
                                    "neden": f"Hedef fiyata ulaşıldı ({r.current_price:.2f} ≥ {x['target_price']:.2f}) — kısmi satış düşünülebilir"})
        # Ekleme adayları: portföyde olmayan en güçlü 3
        cands = sorted([r for r in scan_results
                        if not r.error and r.ticker not in held
                        and r.score >= 60 and r.price_above_sma200],
                       key=lambda r: r.score, reverse=True)[:3]
        for r in cands:
            suggestions.append({"tip": "EKLE (aday)", "ticker": r.ticker,
                                "neden": f"Skor {r.score:.0f}, SMA200 üstünde — çıkarılanın yerine değerlendirilebilir"})
        return suggestions


class TimeMachineEngine:
    """
    """

    TABLE_RUNS   = "tm_runs"
    TABLE_PICKS  = "tm_picks"
    TABLE_DAILY  = "tm_daily"

    # DB Tabloları
    @staticmethod
    def _init_tables():
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TimeMachineEngine.TABLE_RUNS} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE,
                    run_date TEXT,
                    pit_date TEXT,
                    market TEXT,
                    style TEXT,
                    years_back INTEGER,
                    num_picks INTEGER,
                    avg_score REAL,
                    avg_return REAL,
                    bench_return REAL,
                    alpha REAL,
                    grade TEXT
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TimeMachineEngine.TABLE_PICKS} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    ticker TEXT,
                    score REAL,
                    price REAL,
                    current_price REAL,
                    return_pct REAL,
                    rsi REAL,
                    adx REAL,
                    signal TEXT,
                    style TEXT
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TimeMachineEngine.TABLE_DAILY} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date TEXT,
                    run_id TEXT,
                    ticker TEXT,
                    price REAL,
                    daily_chg REAL,
                    cum_return REAL
                )
            """)
            conn.commit()

    # PIT Teknik Analiz (3 yıl önceki verilerle)
    @staticmethod
    def _compute_pit_scores(tickers: list, pit_date: datetime,
                            market: str = "BIST") -> list:
        """
        Belirtilen tarihteki teknik verileri kullanarak skor hesaplar.
        4 yıllık veri indirir, pit_date'e kadar keser → lookahead yok.
        """
        results = []
        download_start = (pit_date - timedelta(days=400)).strftime("%Y-%m-%d")
        download_end   = (pit_date + timedelta(days=1)).strftime("%Y-%m-%d")

        symbols = [_yf_symbol(t, market) for t in tickers]

        # Toplu indirme
        try:
            bulk = yf.download(
                symbols, start=download_start, end=download_end,
                auto_adjust=True, progress=False, group_by="ticker",
            )
        except Exception:
            bulk = None

        for ticker in tickers:
            sym = _yf_symbol(ticker, market)
            try:
                # Bulk'tan çıkar veya tekil indir
                df = PortfolioScanner._extract_df(bulk, sym, ticker)
                if df is None or df.empty or len(df) < 60:
                    df = yf.Ticker(sym).history(
                        start=download_start, end=download_end,
                        auto_adjust=True
                    )
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                if df.empty or len(df) < 60:
                    continue

                # Normalize
                df.index = pd.to_datetime(df.index).tz_localize(None)
                col_rn = {c: c.strip().title() for c in df.columns
                          if c.strip().title() in ("Open","High","Low","Close","Volume")}
                if col_rn:
                    df = df.rename(columns=col_rn)
                if "High"   not in df.columns: df["High"]   = df["Close"]
                if "Low"    not in df.columns: df["Low"]    = df["Close"]
                if "Volume" not in df.columns: df["Volume"] = 0.0

                # PIT tarihine kadar kes (lookahead yok)
                df = df[df.index <= pd.Timestamp(pit_date)]
                if len(df) < 60:
                    continue

                # Teknik analiz (BacktestEngine vektörize skor) — canlı skorlamayla aynı stil
                scores, atr, rsi = BacktestEngine._vectorized_scores(
                    df, style=_tech_style_for(ticker, market))

                pit_score = float(scores.iloc[-1])
                pit_rsi   = float(rsi.iloc[-1])
                pit_price = float(df["Close"].iloc[-1])

                # SMA hesapla
                close = df["Close"]
                sma50  = close.rolling(50, min_periods=20).mean()
                sma200 = close.rolling(200, min_periods=50).mean()
                price_above_sma200 = float(close.iloc[-1]) > float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else False
                price_above_sma50  = float(close.iloc[-1]) > float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else False

                # ADX hesapla
                high_s  = df["High"]
                low_s   = df["Low"]
                prev_c  = close.shift(1)
                tr      = pd.concat([high_s - low_s,
                                     (high_s - prev_c).abs(),
                                     (low_s  - prev_c).abs()], axis=1).max(axis=1)
                atr14   = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                atr_pct = float((atr14.iloc[-1] / close.iloc[-1]) * 100) if close.iloc[-1] > 0 else 0

                plus_dm  = (high_s - high_s.shift(1)).clip(lower=0)
                minus_dm = (low_s.shift(1) - low_s).clip(lower=0)
                plus_dm  = pd.Series(np.where(plus_dm > minus_dm, plus_dm, 0), index=df.index)
                minus_dm = pd.Series(np.where(minus_dm > plus_dm, minus_dm, 0), index=df.index)
                plus_di  = 100 * plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
                minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
                dx       = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
                adx_val  = float(dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1])

                # Momentum
                momentum_1m = float((close.iloc[-1] / close.iloc[-22] - 1) * 100) if len(close) >= 22 else 0
                momentum_3m = float((close.iloc[-1] / close.iloc[-64] - 1) * 100) if len(close) >= 64 else 0

                signal, _ = _score_to_signal(pit_score)

                results.append({
                    "ticker": ticker,
                    "score": round(pit_score, 1),
                    "rsi": round(pit_rsi, 1),
                    "adx": round(adx_val, 1),
                    "atr_pct": round(atr_pct, 2),
                    "price": round(pit_price, 2),
                    "signal": signal,
                    "price_above_sma200": price_above_sma200,
                    "price_above_sma50": price_above_sma50,
                    "momentum_1m": round(momentum_1m, 1),
                    "momentum_3m": round(momentum_3m, 1),
                })

            except Exception as exc:
                log.debug("TimeMachine PIT hata (%s): %s", ticker, exc)
                continue

        return results

    # Portföy Filtresi
    # Portföy stili tanımları
    PORTFOLIO_STYLES = {
        "aggressive": {
            "label": "🚀 Agresif",
            "desc": "Yüksek momentum, güçlü trend, aktif piyasa — yüksek risk/ödül",
            "color": "#b45309",
        },
        "defensive": {
            "label": "🛡️ Defansif",
            "desc": "Düşük volatilite, stabil trend, SMA200 üstü — düşük risk",
            "color": "#27509e",
        },
        "momentum": {
            "label": "⚡ Momentum",
            "desc": "Son 1-3 ay en çok yükselenler, hacim artışı, RSI 50-70",
            "color": "#6d3f8e",
        },
        "value": {
            "label": "💎 Değer (Value)",
            "desc": "Düşük RSI, 52 haftalık dibin yakınında, SMA200 üstü — dipten dönüş",
            "color": "#0e7a6e",
        },
        "stable": {
            "label": "🏦 Stabil/Temettü",
            "desc": "En düşük ATR, en stabil fiyat, uzun vadeli yukarı trend",
            "color": "#a2701d",
        },
        "custom": {
            "label": "🎯 Özel Portföy",
            "desc": "Kullanıcının kendi seçtiği hisseler ile test",
            "color": "#a43d63",
        },
    }

    @staticmethod
    def _filter_portfolio(pit_results: list, style: str = "aggressive") -> list:
        """PIT skorlarına göre portföy filtresi uygular."""
        if style == "custom":
            # Özel portföy filtreleme yapılmaz, dışarıdan gelir
            return pit_results[:15]

        selected = []
        for r in pit_results:
            comp = 0
            if style == "aggressive":
                if r["score"] < 43:              continue
                if r["adx"] < 20:                continue
                if not r["price_above_sma200"]:  continue
                if not r["price_above_sma50"] and r["momentum_1m"] <= 0: continue
                if not (35 <= r["rsi"] <= 72):   continue
                if r["momentum_3m"] < -5:        continue
                comp = r["score"] + min(r["adx"] - 22, 20) * 0.4
                comp += max(0, r["momentum_1m"]) * 0.6

            elif style == "defensive":
                if r["score"] < 38:              continue
                if not r["price_above_sma200"]:  continue
                if r["atr_pct"] > 4.5:           continue
                if not (30 <= r["rsi"] <= 62):   continue
                if r["momentum_3m"] < -8:        continue
                comp = r["score"]

            elif style == "momentum":
                # Momentum: son 1-3 ay güçlü yükseliş, hacim artışı
                if r["momentum_1m"] < 3:         continue   # Son 1 ayda en az %3 yükseliş
                if r["momentum_3m"] < 5:          continue   # Son 3 ayda en az %5
                if not (45 <= r["rsi"] <= 75):    continue   # Güçlü bölge, aşırı alım yakını
                if r["score"] < 35:               continue   # Minimum skor
                comp = r["momentum_1m"] * 2 + r["momentum_3m"] * 1.5
                comp += r["score"] * 0.3
                if r["adx"] > 22: comp += 10       # Trend güçlüyse bonus

            elif style == "value":
                # Value: düşük RSI, 52 haftalık dip yakını, SMA200 üstü
                if r["rsi"] > 45:                 continue   # Düşük RSI (potansiyel dip)
                if not r["price_above_sma200"]:   continue   # Uzun vade trend sağlam
                if r["score"] < 30:               continue   # Çok düşük skoru da alma
                comp = (50 - r["rsi"]) * 2        # RSI ne kadar düşükse o kadar iyi
                comp += r["score"] * 0.5
                if r["momentum_3m"] < -5:  comp += 8   # Düşüş = value fırsatı
                if r["atr_pct"] < 3.5:     comp += 5   # Düşük volatilite bonus

            elif style == "stable":
                # Stabil/Temettü: en düşük ATR, stabil fiyat, uzun vadeli trend
                if r["atr_pct"] > 3.5:            continue   # Çok düşük volatilite
                if not r["price_above_sma200"]:   continue   # Uzun vade yukarı
                if r["rsi"] > 70:                 continue   # Aşırı alım olmasın
                if r["score"] < 35:               continue   # Minimum skor
                comp = (5 - r["atr_pct"]) * 15   # ATR ne kadar düşükse o kadar iyi
                comp += r["score"] * 0.4
                if r["price_above_sma50"]:  comp += 5
                if r["momentum_3m"] > 0:    comp += max(0, r["momentum_3m"]) * 0.3

            r["comp_score"] = round(comp, 2)
            selected.append(r)

        selected.sort(key=lambda x: x.get("comp_score", 0), reverse=True)
        return selected[:10]  # Max 10 hisse

    # Gerçek Performans Ölçümü
    @staticmethod
    def _measure_performance(picks: list, pit_date: datetime,
                             market: str = "BIST") -> list:
        """
        PIT tarihindeki seçimlerinin bugüne kadar gerçek performansını ölçer.
        """
        if not picks:
            return []

        today_str = datetime.now().strftime("%Y-%m-%d")
        pit_str   = pit_date.strftime("%Y-%m-%d")

        symbols = [_yf_symbol(p["ticker"], market) for p in picks]
        try:
            bulk = yf.download(
                symbols, start=pit_str, end=today_str,
                auto_adjust=True, progress=False, group_by="ticker",
            )
        except Exception:
            bulk = None

        for pick in picks:
            sym = _yf_symbol(pick["ticker"], market)
            try:
                df = PortfolioScanner._extract_df(bulk, sym, pick["ticker"])
                if df is None or df.empty:
                    df = yf.Ticker(sym).history(start=pit_str, end=today_str, auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                if df.empty or "Close" not in df.columns:
                    pick["current_price"] = pick["price"]
                    pick["return_pct"] = 0
                    continue

                df.index = pd.to_datetime(df.index).tz_localize(None)
                col_rn = {c: c.strip().title() for c in df.columns
                          if c.strip().title() in ("Close",)}
                if col_rn:
                    df = df.rename(columns=col_rn)

                current_price = float(df["Close"].iloc[-1])
                pit_price     = pick["price"]
                ret_pct       = ((current_price / pit_price) - 1) * 100 if pit_price > 0 else 0

                pick["current_price"] = round(current_price, 2)
                pick["return_pct"]    = round(ret_pct, 1)

            except Exception:
                pick["current_price"] = pick["price"]
                pick["return_pct"] = 0

        return picks

    # Benchmark Performansı
    @staticmethod
    def _benchmark_return(pit_date: datetime, market: str = "BIST") -> float:
        """PIT tarihinden bugüne endeks getirisi."""
        idx = _default_index(market)
        try:
            pit_str   = pit_date.strftime("%Y-%m-%d")
            today_str = datetime.now().strftime("%Y-%m-%d")
            df = yf.download(idx, start=pit_str, end=today_str, progress=False)
            if df.empty or "Close" not in df.columns:
                return 0.0
            df.index = pd.to_datetime(df.index).tz_localize(None)
            col_rn = {c: c.strip().title() for c in df.columns
                      if c.strip().title() in ("Close",)}
            if col_rn:
                df = df.rename(columns=col_rn)
            start_p = float(df["Close"].iloc[0])
            end_p   = float(df["Close"].iloc[-1])
            return round(((end_p / start_p) - 1) * 100, 1) if start_p > 0 else 0.0
        except Exception:
            return 0.0

    # Strateji Notu Hesaplama (0-100)
    @staticmethod
    def _compute_grade(picks: list, bench_ret: float) -> float:
        """
    """
        if not picks:
            return 0.0

        returns = [p.get("return_pct", 0) for p in picks]
        avg_ret = sum(returns) / len(returns) if returns else 0
        win_rate = len([r for r in returns if r > 0]) / len(returns) * 100 if returns else 0

        # Alfa puanı (max 40)
        alpha = avg_ret - bench_ret
        alpha_score = min(40, max(0, (alpha + 20) * (40 / 60)))  # -20%..+40% arası → 0..40

        # Win rate puanı (max 30)
        wr_score = min(30, win_rate * 0.3)

        # Getiri büyüklüğü (max 30)
        ret_score = min(30, max(0, (avg_ret + 10) * (30 / 80)))  # -10%..+70% → 0..30

        grade = round(alpha_score + wr_score + ret_score, 1)
        return min(100, max(0, grade))

    # Tam PIT Analizi Çalıştır
    @staticmethod
    def run_full_pit(years_back: int = 3, market: str = "BIST",
                     style: str = "aggressive",
                     progress_cb=None) -> dict:
        """
    """
        TimeMachineEngine._init_tables()

        pit_date = datetime.now() - timedelta(days=years_back * 365)
        tickers  = BIST_SCAN_UNIVERSE if market == "BIST" else US_POPULAR_TICKERS

        run_id   = f"tm_{market}_{style}_{pit_date.strftime('%Y%m%d')}"
        run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        if progress_cb:
            progress_cb("PIT skorları hesaplanıyor...", 0.1)

        # 1. PIT skorlar
        pit_results = TimeMachineEngine._compute_pit_scores(tickers, pit_date, market)

        if progress_cb:
            progress_cb("Portföy filtresi uygulanıyor...", 0.4)

        # 2. Portföy filtresi
        picks = TimeMachineEngine._filter_portfolio(pit_results, style)

        if progress_cb:
            progress_cb("Gerçek performans ölçülüyor...", 0.6)

        # 3. Performans ölçümü
        picks = TimeMachineEngine._measure_performance(picks, pit_date, market)

        if progress_cb:
            progress_cb("Benchmark karşılaştırılıyor...", 0.8)

        # 4. Benchmark
        bench_ret = TimeMachineEngine._benchmark_return(pit_date, market)

        # 5. Strateji notu
        grade = TimeMachineEngine._compute_grade(picks, bench_ret)

        # 6. Özet hesapla
        returns = [p.get("return_pct", 0) for p in picks]
        avg_ret = sum(returns) / len(returns) if returns else 0
        alpha   = round(avg_ret - bench_ret, 1)

        # 7. DB kaydet
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(f"""
                    INSERT OR REPLACE INTO {TimeMachineEngine.TABLE_RUNS}
                        (run_id, run_date, pit_date, market, style, years_back,
                         num_picks, avg_score, avg_return, bench_return, alpha, grade)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    run_id, run_date, pit_date.strftime("%Y-%m-%d"),
                    market, style, years_back, len(picks),
                    round(sum(p["score"] for p in picks) / max(1, len(picks)), 1),
                    round(avg_ret, 1), bench_ret, alpha, grade,
                ))

                for p in picks:
                    conn.execute(f"""
                        INSERT INTO {TimeMachineEngine.TABLE_PICKS}
                            (run_id, ticker, score, price, current_price, return_pct,
                             rsi, adx, signal, style)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (
                        run_id, p["ticker"], p["score"], p["price"],
                        p.get("current_price", 0), p.get("return_pct", 0),
                        p["rsi"], p["adx"], p["signal"], style,
                    ))
                conn.commit()
        except Exception as exc:
            log.warning("TimeMachine DB kayıt hatası: %s", exc)

        if progress_cb:
            progress_cb("Tamamlandı!", 1.0)

        return {
            "run_id":      run_id,
            "pit_date":    pit_date.strftime("%Y-%m-%d"),
            "picks":       picks,
            "avg_return":  round(avg_ret, 1),
            "bench_return": bench_ret,
            "alpha":       alpha,
            "grade":       grade,
            "style":       style,
            "market":      market,
        }

    # Günlük Snapshot Kaydet
    @staticmethod
    def save_daily_snapshot(run_id: str, market: str = "BIST"):
        """Aktif portföy hisselerinin günlük fiyatını kaydeder."""
        TimeMachineEngine._init_tables()
        today = datetime.now().strftime("%Y-%m-%d")

        try:
            with sqlite3.connect(DB_PATH) as conn:
                picks = conn.execute(
                    f"SELECT ticker, price FROM {TimeMachineEngine.TABLE_PICKS} WHERE run_id=?",
                    (run_id,)
                ).fetchall()

            if not picks:
                return

            tickers = [r[0] for r in picks]
            pit_prices = {r[0]: r[1] for r in picks}

            symbols = [_yf_symbol(t, market) for t in tickers]
            try:
                data = yf.download(symbols, period="5d", auto_adjust=True, progress=False, group_by="ticker")
            except Exception:
                return

            with sqlite3.connect(DB_PATH) as conn:
                for ticker in tickers:
                    sym = _yf_symbol(ticker, market)
                    try:
                        df = PortfolioScanner._extract_df(data, sym, ticker)
                        if df is None or df.empty:
                            continue
                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        col_rn = {c: c.strip().title() for c in df.columns
                                  if c.strip().title() in ("Close",)}
                        if col_rn:
                            df = df.rename(columns=col_rn)
                        if "Close" not in df.columns:
                            continue

                        price = float(df["Close"].iloc[-1])
                        prev_price = float(df["Close"].iloc[-2]) if len(df) >= 2 else price
                        daily_chg = ((price / prev_price) - 1) * 100 if prev_price > 0 else 0
                        cum_ret = ((price / pit_prices[ticker]) - 1) * 100 if pit_prices[ticker] > 0 else 0

                        conn.execute(f"""
                            INSERT OR REPLACE INTO {TimeMachineEngine.TABLE_DAILY}
                                (snapshot_date, run_id, ticker, price, daily_chg, cum_return)
                            VALUES (?,?,?,?,?,?)
                        """, (today, run_id, ticker, round(price, 2),
                              round(daily_chg, 2), round(cum_ret, 1)))
                    except Exception:
                        continue
                conn.commit()
        except Exception as exc:
            log.warning("TimeMachine daily snapshot hatası: %s", exc)

    # Günlük Verileri Yükle
    @staticmethod
    def load_daily_data(run_id: str) -> pd.DataFrame:
        """Kaydedilen günlük snapshot verisini döner."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql_query(
                    f"SELECT * FROM {TimeMachineEngine.TABLE_DAILY} WHERE run_id=? ORDER BY snapshot_date",
                    conn, params=(run_id,)
                )
            return df
        except Exception:
            return pd.DataFrame()

    # Önceki Çalışmaları Yükle
    @staticmethod
    def load_previous_runs(market: str = "BIST") -> list:
        """Daha önceki TimeMachine çalışmalarını listeler."""
        TimeMachineEngine._init_tables()
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute(
                    f"SELECT * FROM {TimeMachineEngine.TABLE_RUNS} WHERE market=? ORDER BY run_date DESC",
                    (market,)
                ).fetchall()
                cols = [d[0] for d in conn.execute(
                    f"SELECT * FROM {TimeMachineEngine.TABLE_RUNS} LIMIT 0"
                ).description]
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    # Çalışmanın Hisselerini Yükle
    @staticmethod
    def load_picks(run_id: str) -> list:
        """Belirli bir run_id'nin hisse seçimlerini döner."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute(
                    f"SELECT * FROM {TimeMachineEngine.TABLE_PICKS} WHERE run_id=?",
                    (run_id,)
                ).fetchall()
                cols = [d[0] for d in conn.execute(
                    f"SELECT * FROM {TimeMachineEngine.TABLE_PICKS} LIMIT 0"
                ).description]
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    # Rapor Üretici
    @staticmethod
    def generate_report(result: dict) -> dict:
        """
        Türkçe strateji raporu üretir.
        Çıktı: {grade, consistency, daily_note, risky_stocks}
        """
        picks = result.get("picks", [])
        grade = result.get("grade", 0)
        alpha = result.get("alpha", 0)
        avg_ret = result.get("avg_return", 0)
        bench = result.get("bench_return", 0)
        style = result.get("style", "aggressive")

        returns = [p.get("return_pct", 0) for p in picks]
        win_count = len([r for r in returns if r > 0])
        lose_count = len([r for r in returns if r <= 0])
        win_rate = (win_count / len(returns) * 100) if returns else 0

        # ── Geriye Dönük Skor Yorumu
        if grade >= 75:
            grade_text = (
                f"Strateji 3 yıllık geçmiş sınavından **{grade:.0f}/100** puanla geçti. "
                f"Bu güçlü bir sonuç — portföy endeksin {alpha:+.1f}% üzerinde performans gösterdi. "
                f"{win_count}/{len(returns)} hisse pozitif getiri sağladı."
            )
        elif grade >= 50:
            grade_text = (
                f"Strateji **{grade:.0f}/100** puan aldı — kabul edilebilir ancak mükemmel değil. "
                f"Portföy getirisi %{avg_ret:+.1f}, endeks %{bench:+.1f} ({alpha:+.1f}% alfa). "
                f"{win_count}/{len(returns)} hisse kazandırdı."
            )
        elif grade >= 25:
            grade_text = (
                f"Strateji **{grade:.0f}/100** — vasat. "
                f"Portföy %{avg_ret:+.1f} getiri sağlarken endeks %{bench:+.1f} yaptı. "
                f"Alfa negatif ({alpha:+.1f}%). Strateji parametreleri gözden geçirilmeli."
            )
        else:
            grade_text = (
                f"Strateji **{grade:.0f}/100** — zayıf performans. "
                f"Seçilen hisseler endeksin {abs(alpha):.1f}% gerisinde kaldı. "
                f"Sadece {win_count}/{len(returns)} hisse kazandırdı. "
                f"Kriterlerde ciddi revizyon gerekebilir."
            )

        # ── Tutarlılık Analizi
        if grade >= 65 and win_rate >= 60:
            consistency = (
                "PIT sonuçları stratejinin tutarlı çalıştığını gösteriyor. "
                "Bugünkü portföye **makul güven** duyulabilir — ancak 'aşırı güven' tuzağından kaçınılmalı. "
                "Piyasa koşulları 3 yıl öncesinden farklı olabilir."
            )
        elif grade >= 40:
            consistency = (
                "Strateji kısmen başarılı oldu ancak mükemmel değil. "
                "Bugünkü portföy için **temkinli iyimserlik** uygun. "
                "Özellikle volatilite yüksek dönemlerde strateji zorlanmış olabilir."
            )
        else:
            consistency = (
                "PIT sonuçları stratejinin geçmişte zorlandığını gösteriyor. "
                "Bugünkü portföye **dikkatli yaklaşılmalı**. "
                "Strateji parametreleri (RSI eşikleri, ADX filtresi) optimize edilmeli."
            )

        # ── Riskli Hisseler (en düşük getirili + yüksek volatilite)
        risky = []
        sorted_picks = sorted(picks, key=lambda x: x.get("return_pct", 0))
        for p in sorted_picks[:3]:
            reason = []
            if p.get("return_pct", 0) < -10:
                reason.append(f"PIT'ten bu yana %{p['return_pct']:.1f} kayıp")
            if p.get("atr_pct", 0) > 4:
                reason.append(f"yüksek volatilite (ATR%{p['atr_pct']:.1f})")
            if p.get("rsi", 50) > 70:
                reason.append("aşırı alım bölgesinde")
            if not reason:
                reason.append(f"en düşük getiri (%{p.get('return_pct', 0):+.1f})")
            risky.append({
                "ticker": p["ticker"],
                "return_pct": p.get("return_pct", 0),
                "reason": ", ".join(reason),
            })

        # ── Günlük Gözlem Notu
        daily_note = (
            "**Kritik İlke:** Bugün oluşturulan bir portföyden hemen çıkarım yapmak mantıksızdır. "
            "Strateji ancak zaman içinde kanıtlanır. Anlık değil, sürekli başarı önemlidir. "
            "Günlük değişimler gürültüdür — haftalık ve aylık trendlere odaklanın."
        )

        return {
            "grade": grade,
            "grade_text": grade_text,
            "consistency": consistency,
            "daily_note": daily_note,
            "risky_stocks": risky,
            "win_rate": round(win_rate, 1),
            "alpha": alpha,
        }


def _get_news_sentiment_cached(ticker: str, date_str: str, lookback_days: int = 7) -> tuple:
    """
    """
    _db = DB_PATH
    # Cache lookup
    try:
        with sqlite3.connect(_db) as conn:
            row = conn.execute(
                "SELECT sentiment, count FROM news_backtest_cache WHERE ticker=? AND date=?",
                (ticker, date_str)
            ).fetchone()
            if row:
                return row[0], int(row[1])
    except Exception:
        pass

    # Cache miss → çek
    sentiment, count = "notr", 0
    if NEWS_ENGINE_AVAILABLE:
        try:
            from news_engine import analyze_news_for_date
            sentiment, count = analyze_news_for_date(ticker, date_str, lookback_days)
        except Exception as exc:
            log.debug("_get_news_sentiment_cached hata: %s", exc)

    # Cache'e kaydet
    try:
        with sqlite3.connect(_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO news_backtest_cache "
                "(ticker, date, sentiment, count, fetched_at) VALUES (?,?,?,?,?)",
                (ticker, date_str, sentiment, count,
                 datetime.now().strftime("%Y-%m-%d %H:%M"))
            )
    except Exception:
        pass

    return sentiment, count


class BacktestEngine:
    """
    """

    # Varsayılan parametreler (Universal mod)
    BUY_THRESHOLD  = 48
    SELL_THRESHOLD = 22
    MAX_HOLD_DAYS  = 60
    WARMUP_DAYS    = 50
    STOP_PCT       = 0.07
    TP_PCT         = 0.14
    COMMISSION     = 0.002
    TRAIL_PCT      = 0.05   # %5 trailing stop mesafesi
    PARTIAL_TP     = 0.07   # %7'de stop breakeven'a çek
    INITIAL_CAPITAL = 100_000.0   # Başlangıç kasası (TL)
    RISK_PER_TRADE  = 0.02        # İşlem başına max risk: kasanın %2'si
    SCALE_OUT_RATIO = 0.5         # Kademeli kâr: pozisyonun %50'si partial TP'de kapatılır
    SHORT_THRESHOLD = 18          # Açığa satış eşiği (skor <= bu değer ve düşen trend)

    # Strateji modları
    MODES = {
        "swing": {
            # Kısa vade: hızlı giriş-çıkış, dar stop, düşük hedef
            "label":          "⚡ Swing (Kısa Vade)",
            "desc":           "5–25 gün, dar stop, hızlı çıkış",
            "BUY_THRESHOLD":  52,
            "SELL_THRESHOLD": 20,
            "MAX_HOLD_DAYS":  25,
            "STOP_PCT":       0.05,
            "TP_PCT":         0.10,
            "TRAIL_PCT":      0.03,
            "PARTIAL_TP":     0.05,
        },
        "trend": {
            # Uzun vade: sabırlı giriş, geniş stop, büyük hedef
            "label":          "📈 Trend (Uzun Vade)",
            "desc":           "30–90 gün, geniş stop, büyük hedef",
            "BUY_THRESHOLD":  44,
            "SELL_THRESHOLD": 25,
            "MAX_HOLD_DAYS":  90,
            "STOP_PCT":       0.09,
            "TP_PCT":         0.22,
            "TRAIL_PCT":      0.07,
            "PARTIAL_TP":     0.11,
        },
        "universal": {
            # Dengeli: varsayılan mod
            "label":          "⚖️ Universal (Dengeli)",
            "desc":           "30–60 gün, dengeli risk/ödül",
            "BUY_THRESHOLD":  48,
            "SELL_THRESHOLD": 22,
            "MAX_HOLD_DAYS":  60,
            "STOP_PCT":       0.07,
            "TP_PCT":         0.14,
            "TRAIL_PCT":      0.05,
            "PARTIAL_TP":     0.07,
        },
        "investor": {
            # Yatırımcı: uzun vadeli al-tut, geniş stop, büyük hedef
            "label":          "🏦 Yatırımcı (Uzun Vade)",
            "desc":           "90–365 gün, geniş stop (%15), SMA200 bazlı çıkış",
            "BUY_THRESHOLD":  40,
            "SELL_THRESHOLD": 18,
            "MAX_HOLD_DAYS":  365,
            "STOP_PCT":       0.15,
            "TP_PCT":         0.40,
            "TRAIL_PCT":      0.10,
            "PARTIAL_TP":     0.20,
        },
        "buyhold": {
            # Al ve Unut: en uzun vade, minimum müdahale
            "label":          "💎 Al & Tut (Buy & Hold)",
            "desc":           "180–730 gün, çok geniş stop (%20), sadece SMA200 kırılımında çık",
            "BUY_THRESHOLD":  38,
            "SELL_THRESHOLD": 15,
            "MAX_HOLD_DAYS":  730,
            "STOP_PCT":       0.20,
            "TP_PCT":         0.60,
            "TRAIL_PCT":      0.12,
            "PARTIAL_TP":     0.30,
        },
    }

    # Vektörize Skor Hesaplama
    @staticmethod
    def _vectorized_scores(df: pd.DataFrame, style: str = "dengeli") -> pd.Series:
        """
        Tüm geçmiş için tek seferde teknik skor hesaplar.
        pandas rolling() geriye dönük olduğundan lookahead bias YOK.

        style: "dengeli" (orijinal, kontrarian) | "momentum" — canlı skorlamayla
        (TechnicalEngine._compute_score) aynı rejim mantığı, bkz. _tech_style_for.
        """
        momentum = (style == "momentum")
        close = df["Close"]
        high  = df.get("High",   close)
        low   = df.get("Low",    close)
        vol   = df.get("Volume", pd.Series(0, index=df.index))

        # SMA & Golden Cross
        sma50  = close.rolling(50,  min_periods=20).mean()
        sma200 = close.rolling(200, min_periods=50).mean()
        sma_gap = (sma50 - sma200) / sma200.replace(0, np.nan) * 100

        mask_gc = sma50 > sma200
        # Golden cross: gap büyüklüğüne göre kademeli +5..+20
        # Death cross: gap büyüklüğüne göre kademeli 0..-10
        gap_abs = sma_gap.abs().clip(0, 5)
        gc_score = pd.Series(np.where(
            mask_gc,
            gap_abs / 5 * 20,           # golden: +0..+20
            -(gap_abs / 5 * 10)         # death:  -0..-10
        ), index=df.index)

        # Fiyat / SMA konumu — orijinale yakın büyüklük
        pa50_score  = pd.Series(np.where(close > sma50,  12.0, -5.0), index=df.index)
        pa200_score = pd.Series(np.where(close > sma200, 12.0, -5.0), index=df.index)

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        # RSI yön tespiti: 3 günlük RSI değişimi (dönüş momentumu)
        rsi_delta  = rsi - rsi.shift(3)   # pozitif = RSI yükseliyor (dönüş)
        rsi_rising = rsi_delta.fillna(0) > 2  # en az 2 puan artmış

        rsi_score = pd.Series(0.0, index=df.index)
        if momentum:
            # Momentum rejimi: güçlü RSI kazandırır (kalibrasyon: IC +0.05, t=3.1)
            rsi_score[rsi >= 80]                          =   8  # aşırı ısınmış — temkinli
            rsi_score[(rsi >= 70) & (rsi < 80)]           =  18
            rsi_score[(rsi >= 60) & (rsi < 70)]           =  14
            rsi_score[(rsi >= 50) & (rsi < 60)]           =   9
            rsi_score[(rsi >= 40) & (rsi < 50)]           =   4
            rsi_score[(rsi >= 30) & (rsi < 40)]           =  -6
            rsi_score[rsi < 30]                           = -12
        else:
            # Mean-reversion odaklı: dipten dönüşe büyük prim
            # RSI < 30 ve yükseliyorsa → güçlü dönüş sinyali (max puan)
            rsi_score[(rsi < 30) & rsi_rising]            = 28
            rsi_score[(rsi < 30) & ~rsi_rising]           = 18   # dip ama henüz dönüş yok
            rsi_score[(rsi >= 30) & (rsi < 40) & rsi_rising]  = 22  # dönüş konfirme
            rsi_score[(rsi >= 30) & (rsi < 40) & ~rsi_rising] = 14
            rsi_score[(rsi >= 40) & (rsi < 50)]           = 10
            rsi_score[(rsi >= 50) & (rsi < 60)]           =  6
            rsi_score[(rsi >= 60) & (rsi < 70)]           =  2
            rsi_score[(rsi >= 70) & (rsi < 80)]           = -10
            rsi_score[rsi >= 80]                          = -18

        # MACD
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd_h = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
        hist_pct = macd_h.abs() / close.replace(0, np.nan) * 100

        macd_score = pd.Series(0.0, index=df.index)
        macd_score[(macd_h > 0) & (hist_pct >= 1.0)]  =  15
        macd_score[(macd_h > 0) & (hist_pct >= 0.3) & (hist_pct < 1.0)] = 10
        macd_score[(macd_h > 0) & (hist_pct < 0.3)]   =   5
        macd_score[(macd_h <= 0) & (hist_pct >= 1.0)] = -10
        macd_score[(macd_h <= 0) & (hist_pct >= 0.3) & (hist_pct < 1.0)] = -5
        macd_score[(macd_h <= 0) & (hist_pct < 0.3)]  =  -2

        # Bollinger
        sma20 = close.rolling(20, min_periods=10).mean()
        std20 = close.rolling(20, min_periods=10).std()
        bb_pos = (close - (sma20 - 2*std20)) / (4 * std20).replace(0, np.nan)

        bb_score = pd.Series(0.0, index=df.index)
        if momentum:
            bb_score[bb_pos > 0.85]                        =   8
            bb_score[(bb_pos > 0.70) & (bb_pos <= 0.85)]  =   6
            bb_score[(bb_pos > 0.50) & (bb_pos <= 0.70)]  =   3
            bb_score[(bb_pos > 0.15) & (bb_pos <= 0.30)]  =  -4
            bb_score[bb_pos <= 0.15]                       =  -8
        else:
            bb_score[bb_pos <= 0.15]                       =  10
            bb_score[(bb_pos > 0.15) & (bb_pos <= 0.30)]  =   7
            bb_score[(bb_pos > 0.30) & (bb_pos <= 0.50)]  =   4
            bb_score[(bb_pos > 0.50) & (bb_pos <= 0.70)]  =   1
            bb_score[(bb_pos > 0.70) & (bb_pos <= 0.85)]  =  -4
            bb_score[bb_pos > 0.85]                        =  -8

        # 52 Haftalık Pozisyon
        w52_hi  = close.rolling(252, min_periods=50).max()
        w52_lo  = close.rolling(252, min_periods=50).min()
        w52_pos = (close - w52_lo) / (w52_hi - w52_lo).replace(0, np.nan)

        w52_score = pd.Series(0.0, index=df.index)
        if momentum:
            # Kalibrasyondaki en güçlü sinyal: zirveye yakınlık kazandırır (t>3.5)
            w52_score[w52_pos > 0.80]                           =  10
            w52_score[(w52_pos > 0.60) & (w52_pos <= 0.80)]    =   7
            w52_score[(w52_pos > 0.40) & (w52_pos <= 0.60)]    =   3
            w52_score[(w52_pos > 0.20) & (w52_pos <= 0.40)]    =  -2
            w52_score[w52_pos <= 0.20]                          =  -6
        else:
            w52_score[w52_pos <= 0.20]                          =  10
            w52_score[(w52_pos > 0.20) & (w52_pos <= 0.40)]    =   7
            w52_score[(w52_pos > 0.40) & (w52_pos <= 0.60)]    =   3
            w52_score[(w52_pos > 0.60) & (w52_pos <= 0.80)]    =  -2
            w52_score[w52_pos > 0.80]                           =  -6

        # Hacim
        avg_vol10 = vol.rolling(10, min_periods=3).mean()
        vol_score = pd.Series(0.0, index=df.index)
        vol_score[(vol > avg_vol10 * 1.5) & (close > close.shift(1))] = 10

        # ATR (stop/hedef hesabı için)
        prev_c = close.shift(1)
        tr     = pd.concat([high - low,
                             (high - prev_c).abs(),
                             (low  - prev_c).abs()], axis=1).max(axis=1)
        atr    = tr.rolling(14, min_periods=5).mean()

        # ADX (yatay piyasa filtresi)
        # Wilder's ADX hesabı (vectorized)
        plus_dm  = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        # +DM sadece -DM'den büyükse geçerli, aksi 0
        plus_dm  = pd.Series(
            np.where(plus_dm > minus_dm, plus_dm, 0), index=df.index)
        minus_dm = pd.Series(
            np.where(minus_dm > plus_dm, minus_dm, 0), index=df.index)

        atr14    = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
        dx       = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx      = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean().fillna(0)

        # ADX < 20 → yatay/yönsüz piyasa → skor penaltisi (sahte sinyal engeli)
        # ADX 20-25 → zayıf trend → küçük penalti
        # ADX ≥ 25 → yeterli trend → penalti yok
        adx_penalty = pd.Series(0.0, index=df.index)
        adx_penalty[adx < 15] = -12   # çok yatay — güçlü penalti
        adx_penalty[(adx >= 15) & (adx < 20)] = -8
        adx_penalty[(adx >= 20) & (adx < 25)] = -4

        # Bollinger Squeeze (bant daralması) → ek penalti
        bb_width  = (4 * std20) / sma20.replace(0, np.nan) * 100
        bb_narrow = bb_width < 5   # bant genişliği %5'ten dar → sıkışma
        squeeze_penalty = pd.Series(
            np.where(bb_narrow & (adx < 22), -5.0, 0.0), index=df.index)

        total = (gc_score + pa50_score + pa200_score +
                 rsi_score + macd_score + bb_score +
                 w52_score + vol_score +
                 adx_penalty + squeeze_penalty)
        scores = total.clip(0, 100).fillna(50)
        return scores, atr, rsi

    # Tek Hisse Backtest
    @staticmethod
    def _run_single(ticker: str, period: str, run_id: str,
                    mode: str = "universal", use_news: bool = True,
                    enable_short: bool = False, enable_scaling: bool = True,
                    initial_capital: float = None, risk_per_trade: float = None,
                    optimized_params: dict = None,
                    market: str = "BIST"):
        """
    """
        # Mod / Optimize parametrelerini çöz
        if optimized_params:
            BUY_TH    = optimized_params.get("BUY_THRESHOLD", BacktestEngine.BUY_THRESHOLD)
            SELL_TH   = optimized_params.get("SELL_THRESHOLD", BacktestEngine.SELL_THRESHOLD)
            MAX_HOLD  = optimized_params.get("MAX_HOLD_DAYS",  BacktestEngine.MAX_HOLD_DAYS)
            STOP_P    = optimized_params.get("STOP_PCT",       BacktestEngine.STOP_PCT)
            TP_P      = optimized_params.get("TP_PCT",         BacktestEngine.TP_PCT)
            TRAIL_P   = optimized_params.get("TRAIL_PCT",      BacktestEngine.TRAIL_PCT)
            PARTIAL_P = optimized_params.get("PARTIAL_TP",     BacktestEngine.PARTIAL_TP)
        else:
            cfg       = BacktestEngine.MODES.get(mode, BacktestEngine.MODES["universal"])
            BUY_TH    = cfg["BUY_THRESHOLD"]
            SELL_TH   = cfg["SELL_THRESHOLD"]
            MAX_HOLD  = cfg["MAX_HOLD_DAYS"]
            STOP_P    = cfg["STOP_PCT"]
            TP_P      = cfg["TP_PCT"]
            TRAIL_P   = cfg["TRAIL_PCT"]
            PARTIAL_P = cfg["PARTIAL_TP"]

        SHORT_TH = BacktestEngine.SHORT_THRESHOLD
        COMM     = BacktestEngine.COMMISSION
        cap      = initial_capital or BacktestEngine.INITIAL_CAPITAL
        risk_pct = risk_per_trade  or BacktestEngine.RISK_PER_TRADE
        scale_r  = BacktestEngine.SCALE_OUT_RATIO if enable_scaling else 0.0

        # Veri çekme
        yt = _yf_symbol(ticker, market)
        raw = yf.Ticker(yt).history(period=period, auto_adjust=True)

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        col_rn = {c: c.strip().title() for c in raw.columns
                  if c.strip().title() in ("Open","High","Low","Close","Volume")}
        if col_rn:
            raw = raw.rename(columns=col_rn)
        if raw.empty or "Close" not in raw.columns:
            raise ValueError("Veri boş veya Close kolonu yok")
        if "Open"   not in raw.columns: raw["Open"]   = raw["Close"]
        if "High"   not in raw.columns: raw["High"]   = raw["Close"]
        if "Low"    not in raw.columns: raw["Low"]    = raw["Close"]
        if "Volume" not in raw.columns: raw["Volume"] = 0.0

        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        min_rows  = BacktestEngine.WARMUP_DAYS + 20
        if len(raw) < min_rows:
            raise ValueError(f"Yeterli veri yok: {len(raw)} < {min_rows} gün")

        scores, atr_s, rsi_s = BacktestEngine._vectorized_scores(
            raw, style=_tech_style_for(ticker, market))
        dates  = raw.index.to_list()
        opens  = raw["Open"].values
        highs  = raw["High"].values
        lows   = raw["Low"].values
        closes = raw["Close"].values
        n      = len(closes)

        # SMA200 — Yatırımcı modları için çıkış kriteri
        _is_investor_mode = mode in ("investor", "buyhold")
        _sma200 = pd.Series(closes).rolling(200, min_periods=50).mean().values

        daily_scores = [
            {"date": str(dates[j])[:10], "score": round(float(scores.iloc[j]), 1),
             "rsi":  round(float(rsi_s.iloc[j]), 1) if not np.isnan(rsi_s.iloc[j]) else 50.0,
             "price": round(float(closes[j]), 2)}
            for j in range(n)
        ]

        # Sektörel Endeks Filtresi
        _fallback_idx = _default_index(market)
        if market == "US":
            sector   = US_SECTOR_MAP.get(ticker, "Technology")
            sect_idx = US_INDEX_MAP.get(sector, "^GSPC")
        else:
            sector   = SECTOR_MAP.get(ticker, "Diğer")
            sect_idx = SECTOR_INDEX_MAP.get(sector, "XU100.IS")
        try:
            for _try_ticker in (sect_idx, _fallback_idx):
                try:
                    _idx_raw = yf.Ticker(_try_ticker).history(period=period, auto_adjust=True)
                    if isinstance(_idx_raw.columns, pd.MultiIndex):
                        _idx_raw.columns = _idx_raw.columns.get_level_values(0)
                    _idx_raw.index = pd.to_datetime(_idx_raw.index).tz_localize(None)
                    if len(_idx_raw) > 30 and "Close" in _idx_raw.columns:
                        break
                except Exception:
                    continue
            _idx_close = _idx_raw["Close"].reindex(raw.index, method="ffill").ffill().bfill()
            _idx_sma50 = _idx_close.rolling(50, min_periods=20).mean()
            index_ok      = (_idx_close > _idx_sma50).values       # boğa: endeks SMA50 üstü
            index_bearish = (_idx_close < _idx_sma50).values       # ayı: endeks SMA50 altı
        except Exception:
            index_ok      = np.ones(n, dtype=bool)
            index_bearish = np.zeros(n, dtype=bool)

        # İşlem döngüsü
        trades       = []
        capital      = cap         # Kasa takibi
        i            = BacktestEngine.WARMUP_DAYS
        in_pos       = False
        is_short     = False       # Mevcut pozisyon yönü
        entry_price  = 0.0
        entry_date   = ""
        entry_score  = 0.0
        take_profit  = 0.0
        trailing_stop    = 0.0
        breakeven_done   = False
        scaled_out       = False   # Kademeli satış yapıldı mı
        entry_idx    = 0
        shares       = 0.0        # Pozisyon adedi
        partial_shares   = 0.0    # Kalan adet (scaling sonrası)
        partial_locked   = 0.0    # Kilitlenen kâr (TL)
        atr_val      = 0.0

        while i < n:
            score = float(scores.iloc[i])

            # POZİSYON YOK
            if not in_pos:

                # LONG giriş
                if score >= BUY_TH and i + 1 < n and index_ok[i]:
                    # Haber filtresi
                    if use_news:
                        _sig_date = str(dates[i])[:10]
                        _ns, _ = _get_news_sentiment_cached(ticker, _sig_date)
                        if _ns == "olumsuz":
                            i += 1
                            continue

                    next_open    = float(opens[i + 1])
                    entry_price  = next_open
                    entry_date   = str(dates[i + 1])[:10]
                    entry_score  = score
                    entry_idx    = i + 1
                    is_short     = False

                    # ATR-bazlı stop/TP
                    atr_val   = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 0.0
                    base_stop = entry_price * (1 - STOP_P)
                    base_tp   = entry_price * (1 + TP_P)
                    if atr_val > 0:
                        atr_stop = entry_price - 2.0 * atr_val
                        atr_tp   = entry_price + 3.0 * atr_val
                        stop_loss   = (base_stop + atr_stop) / 2
                        take_profit = (base_tp   + atr_tp)   / 2
                    else:
                        stop_loss   = base_stop
                        take_profit = base_tp

                    # Hard sınırlar
                    if atr_val > 0:
                        atr_pct_e = atr_val / entry_price
                        min_sd = max(STOP_P * 0.6, 2.0 * atr_pct_e)
                        max_sd = max(STOP_P * 1.6, 3.5 * atr_pct_e)
                    else:
                        min_sd = STOP_P * 0.5
                        max_sd = STOP_P * 1.5
                    stop_loss   = max(stop_loss, entry_price * (1 - max_sd))
                    stop_loss   = min(stop_loss, entry_price * (1 - min_sd))
                    take_profit = max(take_profit, entry_price * 1.06)
                    # Yatırımcı modları için TP tavanını genişlet
                    _tp_cap = 1.70 if _is_investor_mode else 1.35
                    take_profit = min(take_profit, entry_price * _tp_cap)

                    # Pozisyon boyutlandırma (risk-bazlı)
                    risk_per_share = entry_price - stop_loss
                    if risk_per_share > 0:
                        risk_amount = capital * risk_pct
                        shares = risk_amount / risk_per_share
                        position_val = shares * entry_price
                        if position_val > capital * 0.95:  # max %95 kasa
                            shares = (capital * 0.95) / entry_price
                    else:
                        shares = (capital * 0.95) / entry_price

                    trailing_stop  = stop_loss
                    breakeven_done = False
                    scaled_out     = False
                    partial_shares = shares
                    partial_locked = 0.0
                    in_pos = True
                    i += 1
                    continue

                # SHORT giriş
                elif enable_short and score <= SHORT_TH and i + 1 < n and index_bearish[i]:
                    if use_news:
                        _sig_date = str(dates[i])[:10]
                        _ns, _ = _get_news_sentiment_cached(ticker, _sig_date)
                        if _ns == "olumlu":  # pozitif haber varken short açma
                            i += 1
                            continue

                    next_open    = float(opens[i + 1])
                    entry_price  = next_open
                    entry_date   = str(dates[i + 1])[:10]
                    entry_score  = score
                    entry_idx    = i + 1
                    is_short     = True

                    atr_val   = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 0.0
                    # Short: stop YUKARI, TP AŞAĞI
                    base_stop = entry_price * (1 + STOP_P)
                    base_tp   = entry_price * (1 - TP_P)
                    if atr_val > 0:
                        atr_stop = entry_price + 2.0 * atr_val
                        atr_tp   = entry_price - 3.0 * atr_val
                        stop_loss   = (base_stop + atr_stop) / 2
                        take_profit = (base_tp   + atr_tp)   / 2
                    else:
                        stop_loss   = base_stop
                        take_profit = base_tp
                    # Sınırlar
                    stop_loss   = min(stop_loss, entry_price * (1 + STOP_P * 1.6))
                    stop_loss   = max(stop_loss, entry_price * (1 + STOP_P * 0.5))
                    take_profit = min(take_profit, entry_price * 0.94)
                    take_profit = max(take_profit, entry_price * 0.65)

                    risk_per_share = stop_loss - entry_price
                    if risk_per_share > 0:
                        risk_amount = capital * risk_pct
                        shares = risk_amount / risk_per_share
                        position_val = shares * entry_price
                        if position_val > capital * 0.95:
                            shares = (capital * 0.95) / entry_price
                    else:
                        shares = (capital * 0.95) / entry_price

                    trailing_stop  = stop_loss
                    breakeven_done = False
                    scaled_out     = False
                    partial_shares = shares
                    partial_locked = 0.0
                    in_pos = True
                    i += 1
                    continue

            # POZİSYONDAYIZ
            else:
                hold      = i - entry_idx
                day_open  = float(opens[i])
                day_high  = float(highs[i])
                day_low   = float(lows[i])
                day_close = float(closes[i])

                exit_price  = None
                exit_reason = None

                if not is_short:
                    # LONG çıkış kontrolleri
                    if day_open <= trailing_stop:
                        exit_price  = day_open
                        exit_reason = "STOP_LOSS"
                    elif day_low <= trailing_stop:
                        exit_price  = trailing_stop
                        exit_reason = "STOP_LOSS"
                    elif day_open >= take_profit:
                        exit_price  = day_open
                        exit_reason = "TAKE_PROFIT"
                    elif day_high >= take_profit:
                        exit_price  = take_profit
                        exit_reason = "TAKE_PROFIT"
                    elif _is_investor_mode and hold >= 5 and not np.isnan(_sma200[i]):
                        # Yatırımcı modları: fiyat SMA200 altına düşerse çık
                        if day_close < _sma200[i] * 0.98:  # %2 tampon
                            exit_price  = day_close
                            exit_reason = "SMA200_KIRILIM"
                    elif score <= SELL_TH and hold >= 1:
                        exit_price  = day_close
                        exit_reason = "SAT_SINYAL"
                    elif hold >= MAX_HOLD:
                        exit_price  = day_close
                        exit_reason = "MAX_SURE"

                    # Çıkış yoksa: Scaling out + Trailing güncelle
                    if exit_price is None:
                        # Kademeli kâr alma (scaling out)
                        partial_level = entry_price * (1 + PARTIAL_P)
                        if scale_r > 0 and not scaled_out and day_high >= partial_level:
                            close_shares = shares * scale_r
                            partial_pnl  = (partial_level - entry_price) * close_shares
                            partial_pnl -= close_shares * entry_price * COMM  # komisyon
                            partial_locked += partial_pnl
                            partial_shares  = shares - close_shares
                            trailing_stop   = max(trailing_stop, entry_price * 1.003)
                            breakeven_done  = True
                            scaled_out      = True
                        elif not scaled_out and not breakeven_done and day_high >= partial_level:
                            trailing_stop   = max(trailing_stop, entry_price * 1.003)
                            breakeven_done  = True

                        # Trailing stop ratchet (kapanış bazlı → yarın geçerli)
                        if day_close > entry_price:
                            cur_atr = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else atr_val
                            if cur_atr > 0:
                                atr_trail = max(TRAIL_P, 1.8 * cur_atr / day_close)
                            else:
                                atr_trail = TRAIL_P
                            trail_cand = day_close * (1 - atr_trail)
                            trailing_stop = max(trailing_stop, trail_cand)

                else:
                    # SHORT çıkış kontrolleri
                    if day_open >= trailing_stop:
                        exit_price  = day_open
                        exit_reason = "STOP_LOSS"
                    elif day_high >= trailing_stop:
                        exit_price  = trailing_stop
                        exit_reason = "STOP_LOSS"
                    elif day_open <= take_profit:
                        exit_price  = day_open
                        exit_reason = "TAKE_PROFIT"
                    elif day_low <= take_profit:
                        exit_price  = take_profit
                        exit_reason = "TAKE_PROFIT"
                    elif score >= BUY_TH and hold >= 1:
                        exit_price  = day_close
                        exit_reason = "AL_SINYAL"
                    elif hold >= MAX_HOLD:
                        exit_price  = day_close
                        exit_reason = "MAX_SURE"

                    # Çıkış yoksa: Scaling out + Trailing güncelle
                    if exit_price is None:
                        partial_level = entry_price * (1 - PARTIAL_P)
                        if scale_r > 0 and not scaled_out and day_low <= partial_level:
                            close_shares = shares * scale_r
                            partial_pnl  = (entry_price - partial_level) * close_shares
                            partial_pnl -= close_shares * entry_price * COMM
                            partial_locked += partial_pnl
                            partial_shares  = shares - close_shares
                            trailing_stop   = min(trailing_stop, entry_price * 0.997)
                            breakeven_done  = True
                            scaled_out      = True

                        # Short trailing (aşağı hareket ettikçe stop iner)
                        if day_close < entry_price:
                            cur_atr = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else atr_val
                            if cur_atr > 0:
                                atr_trail = max(TRAIL_P, 1.8 * cur_atr / day_close)
                            else:
                                atr_trail = TRAIL_P
                            trail_cand = day_close * (1 + atr_trail)
                            trailing_stop = min(trailing_stop, trail_cand)

                # ÇIKIŞ İŞLEME
                if exit_price is not None:
                    if not is_short:
                        # LONG P&L
                        remaining_pnl = (exit_price - entry_price) * partial_shares
                    else:
                        # SHORT P&L
                        remaining_pnl = (entry_price - exit_price) * partial_shares

                    remaining_pnl -= partial_shares * entry_price * COMM  # exit komisyon
                    total_pnl      = partial_locked + remaining_pnl
                    # Entry komisyonu (tüm shares için, bir kez)
                    total_pnl     -= shares * entry_price * COMM
                    position_cost  = shares * entry_price
                    net_ret_pct    = (total_pnl / position_cost * 100) if position_cost > 0 else 0.0

                    capital += total_pnl  # Kasa güncelle

                    direction = "SHORT" if is_short else "LONG"
                    trades.append({
                        "run_id":       run_id,
                        "ticker":       ticker,
                        "entry_date":   entry_date,
                        "entry_price":  round(entry_price, 2),
                        "exit_date":    str(dates[i])[:10],
                        "exit_price":   round(exit_price, 2),
                        "exit_reason":  exit_reason,
                        "return_pct":   round(net_ret_pct, 2),
                        "hold_days":    hold,
                        "entry_score":  round(entry_score, 1),
                        "stop_loss":    round(trailing_stop, 2),
                        "take_profit":  round(take_profit, 2),
                        "breakeven":    breakeven_done,
                        "direction":    direction,
                        "shares":       round(shares, 0),
                        "position_tl":  round(shares * entry_price, 0),
                        "pnl_tl":       round(total_pnl, 0),
                        "scaled_out":   scaled_out,
                        "capital_after": round(capital, 0),
                    })
                    in_pos = False

            i += 1

        # Dönem sonunda açık pozisyon
        if in_pos:
            last_price = float(closes[-1])
            if not is_short:
                remaining_pnl = (last_price - entry_price) * partial_shares
            else:
                remaining_pnl = (entry_price - last_price) * partial_shares
            remaining_pnl -= partial_shares * entry_price * COMM
            total_pnl      = partial_locked + remaining_pnl
            total_pnl     -= shares * entry_price * COMM
            position_cost  = shares * entry_price
            net_ret_pct    = (total_pnl / position_cost * 100) if position_cost > 0 else 0.0
            capital += total_pnl

            direction = "SHORT" if is_short else "LONG"
            trades.append({
                "run_id":       run_id,
                "ticker":       ticker,
                "entry_date":   entry_date,
                "entry_price":  round(entry_price, 2),
                "exit_date":    str(dates[-1])[:10],
                "exit_price":   round(last_price, 2),
                "exit_reason":  "HALA_ACIK",
                "return_pct":   round(net_ret_pct, 2),
                "hold_days":    n - 1 - entry_idx,
                "entry_score":  round(entry_score, 1),
                "stop_loss":    round(trailing_stop, 2),
                "take_profit":  round(take_profit, 2),
                "breakeven":    breakeven_done,
                "direction":    direction,
                "shares":       round(shares, 0),
                "position_tl":  round(shares * entry_price, 0),
                "pnl_tl":       round(total_pnl, 0),
                "scaled_out":   scaled_out,
                "capital_after": round(capital, 0),
            })

        summary = BacktestEngine._summarize(trades, cap, capital)
        return trades, summary, daily_scores

    @staticmethod
    def _summarize(trades: list, initial_capital: float = 100000,
                   final_capital: float = None) -> dict:
        """İşlem listesinden özet istatistikler çıkarır."""
        if not trades:
            return {
                "total_trades": 0, "winning_trades": 0, "win_rate": 0.0,
                "avg_return_pct": 0.0, "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0, "best_trade_pct": 0.0,
                "worst_trade_pct": 0.0, "avg_hold_days": 0.0,
                "final_capital": initial_capital, "capital_return_pct": 0.0,
                "profit_factor": 0.0, "long_trades": 0, "short_trades": 0,
            }

        returns   = [t["return_pct"] for t in trades]
        wins      = [r for r in returns if r > 0]
        losses    = [r for r in returns if r < 0]
        hold_days = [t["hold_days"]  for t in trades]

        # Max drawdown (equity curve)
        equity = 100.0
        peak   = 100.0
        max_dd = 0.0
        for r in returns:
            equity = equity * (1 + r / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        total_ret = (equity - 100.0)

        # Profit factor
        gross_profit = sum(r for r in returns if r > 0) if wins else 0
        gross_loss   = abs(sum(r for r in returns if r < 0)) if losses else 0
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0

        fc = final_capital or initial_capital
        cap_ret = (fc - initial_capital) / initial_capital * 100

        long_count  = sum(1 for t in trades if t.get("direction", "LONG") == "LONG")
        short_count = sum(1 for t in trades if t.get("direction") == "SHORT")

        return {
            "total_trades":     len(trades),
            "winning_trades":   len(wins),
            "win_rate":         round(len(wins) / len(trades) * 100, 1),
            "avg_return_pct":   round(sum(returns) / len(returns), 2),
            "total_return_pct": round(total_ret, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "best_trade_pct":   round(max(returns), 2),
            "worst_trade_pct":  round(min(returns), 2),
            "avg_hold_days":    round(sum(hold_days) / len(hold_days), 1),
            "final_capital":    round(fc, 0),
            "capital_return_pct": round(cap_ret, 2),
            "profit_factor":    pf,
            "long_trades":      long_count,
            "short_trades":     short_count,
        }

    @staticmethod
    def optimize(ticker: str, period: str = "1y",
                 metric: str = "total_return_pct") -> dict:
        """
    """
        buy_grid  = [42, 44, 46, 48, 50, 52]
        stop_grid = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        tp_grid   = [0.08, 0.10, 0.14, 0.18, 0.22]

        best_score  = -999
        best_params = None

        for buy_th in buy_grid:
            for stop_pct in stop_grid:
                for tp_pct in tp_grid:
                    try:
                        # Geçici mod oluştur
                        temp_mode = {
                            "label": "opt", "desc": "optimizer",
                            "BUY_THRESHOLD": buy_th,
                            "SELL_THRESHOLD": 22,
                            "MAX_HOLD_DAYS": 60,
                            "STOP_PCT": stop_pct,
                            "TP_PCT": tp_pct,
                            "TRAIL_PCT": max(0.03, stop_pct * 0.7),
                            "PARTIAL_TP": tp_pct * 0.5,
                        }
                        # Geçici olarak MODES'a ekle
                        BacktestEngine.MODES["_opt_temp"] = temp_mode
                        trades, summary, _ = BacktestEngine._run_single(
                            ticker, period, "_opt",
                            mode="_opt_temp", use_news=False,
                            enable_short=False, enable_scaling=False,
                        )
                        val = summary.get(metric, 0)
                        # Minimum işlem filtresi (en az 3 işlem)
                        if summary.get("total_trades", 0) >= 3 and val > best_score:
                            best_score  = val
                            best_params = {
                                "BUY_THRESHOLD": buy_th,
                                "STOP_PCT":      stop_pct,
                                "TP_PCT":        tp_pct,
                                "TRAIL_PCT":     max(0.03, stop_pct * 0.7),
                                "PARTIAL_TP":    tp_pct * 0.5,
                                "score":         round(val, 2),
                                "trades":        summary["total_trades"],
                                "win_rate":      summary["win_rate"],
                            }
                    except Exception:
                        pass

        # Temizle
        BacktestEngine.MODES.pop("_opt_temp", None)

        if best_params is None:
            # Varsayılan döndür
            cfg = BacktestEngine.MODES["universal"]
            return {
                "BUY_THRESHOLD": cfg["BUY_THRESHOLD"],
                "STOP_PCT":      cfg["STOP_PCT"],
                "TP_PCT":        cfg["TP_PCT"],
                "TRAIL_PCT":     cfg["TRAIL_PCT"],
                "PARTIAL_TP":    cfg["PARTIAL_TP"],
                "score": 0, "trades": 0, "win_rate": 0,
            }
        return best_params

    @staticmethod
    def _save(db_path, run_id, run_date, ticker, period, trades, summary, daily_scores=None):
        """Backtest sonuçlarını SQLite'a kaydeder (trades + daily scores)."""
        with sqlite3.connect(db_path) as conn:
            # Trades
            for t in trades:
                conn.execute("""
                    INSERT INTO backtest_trades
                        (run_id, ticker, entry_date, entry_price, exit_date, exit_price,
                         exit_reason, return_pct, hold_days, entry_score, stop_loss, take_profit)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    t["run_id"], t["ticker"], t["entry_date"], t["entry_price"],
                    t["exit_date"], t["exit_price"], t["exit_reason"], t["return_pct"],
                    t["hold_days"], t["entry_score"], t["stop_loss"], t["take_profit"],
                ))
            # Daily scores
            if daily_scores:
                conn.executemany("""
                    INSERT OR IGNORE INTO backtest_daily (run_id, ticker, date, score, rsi, price)
                    VALUES (?,?,?,?,?,?)
                """, [(run_id, ticker, d["date"], d["score"], d["rsi"], d["price"])
                      for d in daily_scores])
            # Summary
            if summary.get("total_trades", 0) > 0:
                conn.execute("""
                    INSERT OR REPLACE INTO backtest_summary
                        (run_id, run_date, ticker, period, total_trades, winning_trades,
                         win_rate, avg_return_pct, total_return_pct, max_drawdown_pct,
                         best_trade_pct, worst_trade_pct, avg_hold_days)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    run_id, run_date, ticker, period,
                    summary["total_trades"], summary["winning_trades"],
                    summary["win_rate"], summary["avg_return_pct"],
                    summary["total_return_pct"], summary["max_drawdown_pct"],
                    summary["best_trade_pct"], summary["worst_trade_pct"],
                    summary["avg_hold_days"],
                ))
            conn.commit()

    @staticmethod
    def load_runs(db_path: str = None) -> list:
        db_path = db_path or DB_PATH
        """Kayıtlı backtest özetlerini döndürür."""
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute("""
                    SELECT run_id, run_date, ticker, period, total_trades, winning_trades,
                           win_rate, avg_return_pct, total_return_pct, max_drawdown_pct,
                           best_trade_pct, worst_trade_pct, avg_hold_days
                    FROM backtest_summary ORDER BY run_date DESC
                """).fetchall()
            cols = ["run_id","run_date","ticker","period","total_trades","winning_trades",
                    "win_rate","avg_return_pct","total_return_pct","max_drawdown_pct",
                    "best_trade_pct","worst_trade_pct","avg_hold_days"]
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    @staticmethod
    def load_daily_scores(run_id: str, ticker: str, db_path: str = None) -> pd.DataFrame:
        db_path = db_path or DB_PATH
        """Belirli run+hisse için günlük skor serisini döndürür."""
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT date, score, rsi, price FROM backtest_daily "
                    "WHERE run_id=? AND ticker=? ORDER BY date",
                    (run_id, ticker)
                ).fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=["date","score","rsi","price"])
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def load_trades(run_id: str, ticker: str = None, db_path: str = None) -> list:
        db_path = db_path or DB_PATH
        """Belirli bir backtest run'ının trade detaylarını döndürür."""
        try:
            with sqlite3.connect(db_path) as conn:
                if ticker:
                    rows = conn.execute(
                        "SELECT * FROM backtest_trades WHERE run_id=? AND ticker=? ORDER BY entry_date",
                        (run_id, ticker)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM backtest_trades WHERE run_id=? ORDER BY ticker, entry_date",
                        (run_id,)
                    ).fetchall()
                cols = [d[0] for d in conn.execute(
                    "SELECT * FROM backtest_trades LIMIT 0"
                ).description or []]
            if not cols:
                cols = ["id","run_id","ticker","entry_date","entry_price","exit_date",
                        "exit_price","exit_reason","return_pct","hold_days","entry_score",
                        "stop_loss","take_profit"]
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []


# 8. STREAMLIT DASHBOARD (Dark Mode)

def create_gauge_chart(score: float, title: str = "BIST Buy Score") -> go.Figure:
    if score >= 75:   bar_color = "#1d6f4e"
    elif score >= 55: bar_color = "#3e8e6c"
    elif score >= 40: bar_color = "#6b6357"
    elif score >= 20: bar_color = "#b45309"
    else:             bar_color = "#9e2b25"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 16, "color": "#1a1712"}},
        number={"font": {"size": 36, "color": "#1a1712"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#b8ae9a",
                     "tickfont": {"color": _theme()["muted"]}},
            "bar": {"color": bar_color},
            "bgcolor": "#efe9db",
            "steps": [
                {"range": [0, 20],   "color": "#e7c9c5"},
                {"range": [20, 40],  "color": "#ecd9c4"},
                {"range": [40, 55],  "color": "#efe9db"},
                {"range": [55, 75],  "color": "#cfe0d4"},
                {"range": [75, 100], "color": "#b9d4c2"},
            ],
            "threshold": {"line": {"color": _theme()["ink"], "width": 3},
                          "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["bg"],
        height=280, margin=dict(l=30, r=30, t=40, b=20),
    )
    return fig


def create_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=ticker,
        increasing_line_color="#1d6f4e", decreasing_line_color="#9e2b25",
    ), row=1, col=1)
    
    sma50 = df["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50",
                             line=dict(color="#a2701d", width=1.5)), row=1, col=1)
    if len(df) >= 200:
        sma200 = df["Close"].rolling(200).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA 200",
                                 line=dict(color="#27509e", width=1.5)), row=1, col=1)
                                 
    if "Volume" in df.columns:
        colors = ["#1d6f4e" if row["Close"] >= row["Open"] else "#9e2b25" 
                  for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Hacim",
            marker_color=colors, opacity=0.8
        ), row=2, col=1)
        
    fig.update_layout(
        title=f"{ticker} — Price, MAs & Volume",
        paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
        font=dict(color=_theme()["ink"]),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#efe9db"), height=500,
        margin=dict(l=30, r=30, t=40, b=20)
    )
    fig.update_yaxes(gridcolor="#d8d0c0")
    fig.update_xaxes(gridcolor="#d8d0c0")
    return fig


def render_dashboard_page(ui_lang):
    title = "Piyasa Özeti (Dashboard)" if ui_lang == "TR" else "Market Overview"
    st.markdown(f"# {title}")
    st.caption(
        "Borsa İstanbul ve makroekonomik göstergelerin anlık özeti. Veriler 10-60 dk aralıklarla güncellenir."
        if ui_lang == "TR" else
        "Real-time summary of Borsa Istanbul and macroeconomic indicators. Data refreshes every 10-60 min."
    )

    # Makroekonomik Göstergeler
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_macro_indicators():
        """USD/TL, EUR/TL, Altın (USD), BIST100 günlük değişimlerini çeker."""
        symbols = {
            "USDTRY=X": ("USD/TL", "₺"),
            "EURTRY=X": ("EUR/TL", "₺"),
            "GC=F":     ("Altın", "$"),
            "XU100.IS": ("BIST100", ""),
        }
        result = {}
        try:
            df = yf.download(
                list(symbols.keys()), period="5d",
                group_by="ticker", auto_adjust=True, progress=False,
            )
            for sym, (label, unit) in symbols.items():
                try:
                    col = df[sym]["Close"] if len(symbols) > 1 else df["Close"]
                    col = col.dropna()
                    if len(col) >= 2:
                        cur  = float(col.iloc[-1])
                        prev = float(col.iloc[-2])
                        pct  = ((cur - prev) / prev * 100) if prev != 0 else 0.0
                        result[label] = {"value": cur, "pct": pct, "unit": unit}
                except Exception as exc:
                    log.warning("Makro gösterge hatası (%s): %s", sym, exc)
                    pass
        except Exception:
            pass
        return result

    with st.spinner("Makro veriler çekiliyor..." if ui_lang == "TR" else "Loading macro data..."):
        macro = fetch_macro_indicators()

    if macro:
        st.markdown("### Makroekonomik Göstergeler" if ui_lang == "TR" else "### Macro Indicators")
        mcols = st.columns(len(macro))
        for col, (label, data) in zip(mcols, macro.items()):
            pct   = data["pct"]
            val   = data["value"]
            unit  = data["unit"]
            delta = f"{pct:+.2f}%"
            col.metric(
                label=label,
                value=f"{val:,.2f} {unit}".strip(),
                delta=delta,
                delta_color="inverse" if label in ("USD/TL", "EUR/TL") else "normal",
            )
        st.markdown("---")

    acc_rate, acc_success, acc_total = _history_db.get_accuracy_stats()
    if acc_total > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Doğruluk Oranı" if ui_lang == "TR" else "Model Accuracy", f"%{acc_rate:.1f}")
        c2.metric("Başarılı Tahmin" if ui_lang == "TR" else "Successful Calls", acc_success)
        c3.metric("Değerlendirilen" if ui_lang == "TR" else "Total Evaluated", acc_total)
        st.markdown("---")

    # Bugünün Sinyalleri (Hızlı Özet)
    try:
        _all_signals = _history_db.get_validation_report()
        if _all_signals:
            _today = datetime.now().strftime("%Y-%m-%d")
            _recent = [s for s in _all_signals
                       if s.get("signal_date", "")[:10] == _today
                       and s.get("source", "").startswith("live")]
            if _recent:
                st.markdown("### " + (
                    "Bugunun Sinyalleri" if ui_lang == "TR" else "Today's Signals"
                ))
                _al = [s for s in _recent if s["signal"] in ("AL", "GUCLU AL")]
                _sat = [s for s in _recent if s["signal"] in ("SAT", "GUCLU SAT")]

                if _al:
                    _al_items = " &nbsp;|&nbsp; ".join(
                        f"<span style='color:#1d6f4e;font-weight:600'>{s['ticker']}</span> "
                        f"<small>({s['signal']} {s['score']:.0f})</small>"
                        for s in sorted(_al, key=lambda x: -x["score"])
                    )
                    st.markdown(
                        f"**{'AL' if ui_lang == 'TR' else 'BUY'}:** {_al_items}",
                        unsafe_allow_html=True
                    )
                if _sat:
                    _sat_items = " &nbsp;|&nbsp; ".join(
                        f"<span style='color:#9e2b25;font-weight:600'>{s['ticker']}</span> "
                        f"<small>({s['signal']} {s['score']:.0f})</small>"
                        for s in sorted(_sat, key=lambda x: x["score"])
                    )
                    st.markdown(
                        f"**{'SAT' if ui_lang == 'TR' else 'SELL'}:** {_sat_items}",
                        unsafe_allow_html=True
                    )
                if not _al and not _sat:
                    st.info(
                        "Bugun NOTR disinda sinyal yok."
                        if ui_lang == "TR" else "No non-neutral signals today."
                    )
                st.markdown("---")
    except Exception:
        pass

    @st.cache_data(ttl=3600)
    def fetch_market_overview():
        bist30 = ["AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TTKOM.IS", "TUPRS.IS", "VAKBN.IS", "YKBNK.IS"]

        def _ticker_df(raw: pd.DataFrame, sym: str) -> pd.DataFrame:
            """Versiyon bağımsız tek hisse OHLCV DataFrame döndürür."""
            if not isinstance(raw.columns, pd.MultiIndex):
                return raw  # Tek hisse indirilmişse düz döner
            lvl0 = raw.columns.get_level_values(0)
            # Yeni format: (Price, Ticker)
            if "Close" in lvl0:
                price_cols = [c for c in ("Open","High","Low","Close","Volume") if c in lvl0]
                rows = {pc: raw[pc][sym] for pc in price_cols if sym in raw[pc].columns}
                return pd.DataFrame(rows) if rows else pd.DataFrame()
            # Eski format: (Ticker, Price)
            if sym in lvl0:
                return raw[sym]
            return pd.DataFrame()

        try:
            df = yf.download(bist30, period="60d", auto_adjust=True, progress=False)
            results = []
            screener_oversold = []
            screener_macd = []

            for t in bist30:
                d = _ticker_df(df, t)
                d = d.dropna(subset=["Close"]) if "Close" in d.columns else d
                if len(d) >= 30:
                    current = float(d["Close"].iloc[-1])
                    prev = float(d["Close"].iloc[-2])
                    pct = ((current - prev) / prev) * 100
                    results.append({"Hisse": t.replace(".IS", ""), "Fiyat": current, "Değişim (%)": pct})
                    
                    # Technicals for Screener
                    delta = d["Close"].diff()
                    gain = delta.where(delta > 0, 0.0)
                    loss = -delta.where(delta < 0, 0.0)
                    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
                    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
                    rs = avg_gain / (avg_loss + 1e-9)
                    rsi_series = 100 - (100 / (1 + rs))
                    current_rsi = float(rsi_series.iloc[-1])
                    
                    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
                    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
                    macd_line = ema12 - ema26
                    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                    
                    m_curr = float(macd_line.iloc[-1])
                    s_curr = float(macd_signal.iloc[-1])
                    m_prev = float(macd_line.iloc[-2])
                    s_prev = float(macd_signal.iloc[-2])
                    
                    if current_rsi < 35:
                        screener_oversold.append({"Hisse": t.replace(".IS", ""), "RSI": current_rsi, "Fiyat": current})
                        
                    if m_curr > s_curr and m_prev <= s_prev:
                        screener_macd.append({"Hisse": t.replace(".IS", ""), "Sinyal": "MACD Yukarı Kesti"})
                        
            r_df = pd.DataFrame(results).sort_values("Değişim (%)", ascending=False) if results else pd.DataFrame()
            so_df = pd.DataFrame(screener_oversold).sort_values("RSI") if screener_oversold else pd.DataFrame()
            sm_df = pd.DataFrame(screener_macd) if screener_macd else pd.DataFrame()
            return r_df, so_df, sm_df
        except Exception:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    @st.cache_data(ttl=1800)
    def fetch_economy_news():
        if not FEEDPARSER_OK:
            return []
        try:
            feed = feedparser.parse("https://www.bloomberght.com/rss")
            return [{"title": e.title, "link": e.link} for e in feed.entries[:12]]
        except Exception:
            return []

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Günün Öne Çıkanları (BIST 30)" if ui_lang == "TR" else "### Daily Movers (BIST 30)")
        with st.spinner("Piyasa verileri çekiliyor..." if ui_lang == "TR" else "Fetching market data..."):
            movers_df, oversold_df, macd_df = fetch_market_overview()
            
        if not movers_df.empty:
            m1, m2 = st.columns(2)
            with m1:
                st.success("En Çok Kazandıranlar" if ui_lang == "TR" else "Top Gainers")
                st.dataframe(movers_df.head(5).style.format({"Fiyat": "{:.2f} TL", "Değişim (%)": "{:+.2f}%"}), use_container_width=True, hide_index=True)
            with m2:
                st.error("En Çok Kaybettirenler" if ui_lang == "TR" else "Top Losers")
                st.dataframe(movers_df.tail(5).sort_values("Değişim (%)").style.format({"Fiyat": "{:.2f} TL", "Değişim (%)": "{:+.2f}%"}), use_container_width=True, hide_index=True)
        else:
            st.warning("Piyasa verisi alınamadı." if ui_lang == "TR" else "Could not fetch market data.")
            
        st.markdown("---")
        st.markdown("### Teknik Fırsat Radarı (Screener)" if ui_lang == "TR" else "### Technical Radar")
        
        s1, s2 = st.columns(2)
        with s1:
            st.info("Aşırı Satım (RSI < 35)" if ui_lang == "TR" else "Oversold (RSI < 35)")
            if not oversold_df.empty:
                st.dataframe(oversold_df.style.format({"RSI": "{:.1f}", "Fiyat": "{:.2f} TL"}), use_container_width=True, hide_index=True)
            else:
                st.caption("Aşırı satım bölgesinde BIST30 hissesi bulunamadı." if ui_lang == "TR" else "No oversold stocks found.")
        with s2:
            st.info("Trend Dönüşü (MACD Al Sinyali)" if ui_lang == "TR" else "Trend Reversal (MACD)")
            if not macd_df.empty:
                st.dataframe(macd_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Yeni MACD AL veren hisse bulunamadı." if ui_lang == "TR" else "No new MACD buy signals.")


    with c2:
        st.markdown("### Ekonomi Haberleri" if ui_lang == "TR" else "### Economy News")
        news_items = fetch_economy_news()
        if news_items:
            from html import escape as _esc
            for item in news_items:
                # Dış RSS içeriği HTML'e gömülüyor → XSS'e karşı escape;
                # javascript: vb. şemalar çizilmez
                _lnk = str(item.get("link", ""))
                if not _lnk.startswith(("http://", "https://")):
                    continue
                st.markdown(
                    f"<div style='background:#efe9db; padding:10px; border-radius:8px; margin-bottom:8px; border-left:4px solid #27509e;'>"
                    f"<a href='{_esc(_lnk, quote=True)}' target='_blank' style='text-decoration:none; color:#1a1712; font-size:14px;'>{_esc(str(item.get('title', '')))}</a>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Haber bulunamadı." if ui_lang == "TR" else "No news found.")

def render_smart_portfolio_page(ui_lang: str):
    """Sistem tarafından seçilen akıllı portföyler ve backtest sonuçları."""
    st.markdown("# 🤖 Sistem Portföyleri")
    st.caption(
        "Sistem BIST'teki ~70 hisseyi otomatik tarar, teknik kriterlere göre iki portföy oluşturur: "
        "**Agresif** (yüksek potansiyel, yüksek risk) ve **Defansif** (stabil, düşük oynaklık). "
        "Portföy seçimi tamamen sisteme aittir."
    )
    st.markdown("---")

    # Sidebar Kontroller
    with st.sidebar:
        st.markdown("## Portföy Tarama")
        # Kaynak istismarı koruması: ~500 hisselik tam tarama pahalıdır
        # (yfinance rate-limit / kaynak tüketimi) — anonim ziyaretçi tetikleyemesin
        force_scan = st.button("🔄 Yeniden Tara", use_container_width=True,
                               help="Tüm hisseleri sıfırdan tara (2-3 dk)") and _guard_write()
        st.markdown("---")
        st.markdown("### Backtest Ayarları")
        bt_period = st.selectbox(
            "Periyot", ["1y","2y"], index=1,
            format_func=lambda x: {"1y":"1 Yıl","2y":"2 Yıl"}[x],
            key="smart_bt_period_sel"
        )
        bt_mode_sp = st.radio(
            "Strateji Modu",
            options=["swing", "trend", "universal", "investor", "buyhold"],
            index=2,
            format_func=lambda x: BacktestEngine.MODES[x]["label"],
        )
        use_news_sp = st.checkbox("📰 Haber Filtresi", value=True,
                                  key="sp_news_filter")
        st.markdown("**Hangi portföyü test et?**")
        bt_run_agg = st.checkbox("🚀 Agresif", value=True, key="bt_run_agg")
        bt_run_def = st.checkbox("🛡️ Defansif", value=True, key="bt_run_def")
        run_backtest_btn = st.button("▶ Burada Backtest",
                                     type="primary", use_container_width=True,
                                     key="smart_bt_run",
                                     help="Portföyü bu sayfada test et")
        send_to_bt_btn = st.button("↗ Backtest Sayfasına Gönder",
                                    use_container_width=True,
                                    key="send_to_bt_btn",
                                    help="Seçili portföyü Backtest sekmesine aktar")
        st.markdown("---")
        st.caption(
            "**🚀 Agresif Kriterleri**\n"
            "- Skor ≥ 43\n- ADX > 20\n"
            "- Fiyat SMA200 üstü\n- RSI 35-72\n\n"
            "**🛡️ Defansif Kriterleri**\n"
            "- Skor ≥ 38\n- Fiyat SMA200 üstü\n"
            "- ATR% ≤ 4.5\n- RSI 30-62\n\n"
            "_Portföyler birbirinden farklıdır._"
        )

    # Tarama
    scan_results = None
    if force_scan or "smart_scan_results" not in st.session_state:
        progress_bar = st.progress(0, text="Tarama başlatılıyor...")
        status_txt   = st.empty()
        total        = len(BIST_SCAN_UNIVERSE)

        def _progress(ticker, idx, total):
            pct = (idx + 1) / total
            progress_bar.progress(pct, text=f"Tarıyor: {ticker} ({idx+1}/{total})")
            status_txt.caption(f"Son: {ticker}")

        with st.spinner("BIST hisseleri taranıyor..."):
            scan_results = PortfolioScanner.scan_all(force=force_scan, progress_cb=_progress)

        st.session_state["smart_scan_results"] = scan_results
        progress_bar.progress(1.0, text=f"Tarama tamamlandı — {len(scan_results)} hisse analiz edildi")
        status_txt.empty()
    else:
        scan_results = st.session_state["smart_scan_results"]
        cached = PortfolioScanner._load_cache()
        if cached:
            scan_date = "önbellekten"
            st.info(f"Son tarama önbellekten yüklendi ({len(scan_results)} hisse). Yeniden taramak için sol menüden '🔄 Yeniden Tara' butonuna bas.")

    if not scan_results:
        st.warning("Tarama sonucu bulunamadı.")
        return

    # Portföy Oluştur
    portfolios = SmartPortfolioBuilder.build(scan_results)
    aggressive = portfolios["aggressive"]
    defensive  = portfolios["defensive"]

    # "Backtest Sayfasına Gönder" butonu işlemi
    if send_to_bt_btn:
        combined_tickers = []
        if bt_run_agg:
            combined_tickers += [s["ticker"] for s in aggressive]
        if bt_run_def:
            combined_tickers += [s["ticker"] for s in defensive]
        if combined_tickers:
            st.session_state["preload_bt_tickers"] = list(dict.fromkeys(combined_tickers))
            st.session_state["nav_radio"] = "Backtest"
            st.rerun()
        else:
            st.warning("Portföy boş — önce tarama yapın.")

    # Tarama Özeti
    valid_count  = len([r for r in scan_results if not r.error])
    error_count  = len([r for r in scan_results if r.error])
    high_score   = len([r for r in scan_results if r.score >= 46])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taranan Hisse",    len(scan_results))
    c2.metric("Başarılı Analiz",  valid_count)
    c3.metric("Yüksek Skorlu",    high_score)
    c4.metric("Hata",             error_count)

    st.markdown("---")

    # İki Portföy Yan Yana
    col_agg, col_def = st.columns(2)

    def _render_portfolio_table(stocks: list, title: str, color: str):
        st.markdown(
            f"<div style='background:#efe9db;border:2px solid {color};"
            f"border-radius:10px;padding:12px 16px;margin-bottom:12px'>"
            f"<h4 style='color:{color};margin:0'>{title}</h4>"
            f"<span style='color:#6b6357;font-size:12px'>{len(stocks)} hisse seçildi</span>"
            f"</div>", unsafe_allow_html=True
        )
        if not stocks:
            st.warning("Bu kriterleri karşılayan hisse bulunamadı.")
            return
        for s in stocks:
            gc_badge = "🟡 GC" if s["golden_cross"] else ""
            obv_badge = "📈 OBV↑" if s["obv_trend"] == "yukari" else ""
            st.markdown(
                f"<div style='background:#f7f3ea;border:1px solid #d8d0c0;"
                f"border-radius:8px;padding:10px 14px;margin-bottom:6px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-size:16px;font-weight:700;color:{color}'>{s['ticker']}</span>"
                f"<span style='font-size:13px;color:#1a1712'>{s['price']:.2f} ₺</span>"
                f"</div>"
                f"<div style='font-size:12px;color:#6b6357;margin-top:4px'>"
                f"Skor: <b style='color:#1a1712'>{s['score']}</b> · "
                f"ADX: <b>{s.get('adx', '-')}</b> · "
                f"RSI: <b>{s['rsi']}</b> · "
                f"ATR%: <b>{s['atr_pct']}</b> · "
                f"52H: <b>%{s['week52_pos']}</b> · "
                f"1A: <b style='color:{'#1d6f4e' if s.get('momentum_1m',0)>=0 else '#9e2b25'}'>{s.get('momentum_1m',0):+.1f}%</b>"
                f"{gc_badge} {obv_badge}"
                f"</div>"
                f"<div style='font-size:11px;color:#8a8172;margin-top:3px'>{s['reason']}</div>"
                f"</div>", unsafe_allow_html=True
            )

    with col_agg:
        _render_portfolio_table(aggressive, "🚀 Agresif Portföy", "#b45309")

    with col_def:
        _render_portfolio_table(defensive, "🛡️ Defansif Portföy", "#27509e")

    # Tüm Tarama Sonuçları Tablosu
    st.markdown("---")
    with st.expander("📊 Tüm Tarama Sonuçları (Skor Sırası)", expanded=False):
        scan_df = pd.DataFrame([{
            "Hisse": r.ticker,
            "Skor":  round(r.score, 1),
            "RSI":   round(r.rsi, 1),
            "ADX":   round(r.adx, 1),
            "ATR%":  round(r.atr_pct, 2),
            "52H%":  round(r.week52_pos * 100, 1),
            "OBV":   r.obv_trend,
            "GC":    "✅" if r.golden_cross else "❌",
            "SMA200":"✅" if r.price_above_sma200 else "❌",
            "Hacim": "✅" if r.volume_ok else "❌",
            "Fiyat": round(r.current_price, 2),
        } for r in scan_results if not r.error])

        if not scan_df.empty:
            scan_df = scan_df.sort_values("Skor", ascending=False)

            def _score_color(v):
                if not isinstance(v, float): return ""
                if v >= 50: return "background-color: rgba(34,197,94,0.2)"
                if v >= 42: return "background-color: rgba(251,146,60,0.15)"
                return "background-color: rgba(239,68,68,0.1)"

            st.dataframe(
                scan_df.style.map(_score_color, subset=["Skor"]),
                use_container_width=True, hide_index=True, height=400,
            )

    # Backtest
    if run_backtest_btn and (aggressive or defensive):
        st.markdown("---")
        st.markdown("## 🧪 Sistem Portföyleri Backtest")

        run_id   = datetime.now().strftime("%Y%m%d_%H%M%S") + "_smart"
        run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        bt_results_agg = {}   # port_name → {ticker: {trades, summary}}

        for port_name, port_stocks, port_color in [
            ("Agresif",  aggressive, "#b45309"),
            ("Defansif", defensive,  "#27509e"),
        ]:
            if not port_stocks:
                continue

            tickers = [s["ticker"] for s in port_stocks]
            prog = st.progress(0, text=f"{port_name} başlatılıyor...")
            bt_results = {}

            for i4, ticker in enumerate(tickers):
                prog.progress((i4 + 1) / len(tickers), text=f"{port_name}: {ticker}")
                try:
                    trades, summary, daily = BacktestEngine._run_single(
                        ticker, bt_period, run_id + f"_{port_name}",
                        mode=bt_mode_sp, use_news=use_news_sp,
                        enable_short=False, enable_scaling=True,
                    )
                    BacktestEngine._save(
                        DB_PATH, run_id + f"_{port_name}",
                        run_date, ticker, bt_period, trades, summary, daily
                    )
                    bt_results[ticker] = {"trades": trades, "summary": summary}
                except Exception as exc4:
                    bt_results[ticker] = {"trades": [], "summary": {}, "error": str(exc4)}

            prog.progress(1.0, text=f"{port_name} tamamlandı!")
            bt_results_agg[port_name] = bt_results

        st.session_state["smart_bt_results"] = bt_results_agg
        st.session_state["smart_bt_period"]  = bt_period

    # Backtest sonuçları göster
    bt_results_agg = st.session_state.get("smart_bt_results", {})
    if bt_results_agg:
        st.markdown("---")
        st.markdown("## 🧪 Backtest Sonuçları")
        bt_period_used = st.session_state.get("smart_bt_period", bt_period)

        port_tabs_names = [n for n in ["Agresif", "Defansif"] if n in bt_results_agg and bt_results_agg[n]]
        if port_tabs_names:
            port_tabs = st.tabs([f"{'🚀' if n=='Agresif' else '🛡️'} {n}" for n in port_tabs_names])

            for tab_idx, port_name in enumerate(port_tabs_names):
                with port_tabs[tab_idx]:
                    bt_res = bt_results_agg[port_name]
                    valid_bt = {k: v for k, v in bt_res.items()
                                if "error" not in v and v.get("summary", {}).get("total_trades", 0) > 0}

                    if not valid_bt:
                        st.info(f"{port_name} portföyündeki hisselerde işlem üretilemedi.")
                        continue

                    # Özet metrikler
                    all_rets = [v["summary"]["total_return_pct"] for v in valid_bt.values()]
                    all_wrs  = [v["summary"]["win_rate"] for v in valid_bt.values()]
                    port_ret = sum(all_rets) / len(all_rets)
                    avg_wr   = sum(all_wrs) / len(all_wrs)
                    best_t  = max(valid_bt.items(), key=lambda x: x[1]["summary"]["total_return_pct"])
                    worst_t = min(valid_bt.items(), key=lambda x: x[1]["summary"]["total_return_pct"])

                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Portföy Getirisi",   f"%{port_ret:+.1f}")
                    pm2.metric("Ort. Kazanma",       f"%{avg_wr:.0f}")
                    pm3.metric("En İyi",  best_t[0],  f"%{best_t[1]['summary']['total_return_pct']:+.1f}")
                    pm4.metric("En Kötü", worst_t[0], f"%{worst_t[1]['summary']['total_return_pct']:+.1f}")

                    # Bar chart
                    bt_labels  = list(valid_bt.keys())
                    bt_returns = [valid_bt[t]["summary"]["total_return_pct"] for t in bt_labels]
                    port_color = "#b45309" if port_name == "Agresif" else "#27509e"
                    fig_bt = go.Figure(go.Bar(
                        x=bt_labels, y=bt_returns,
                        marker_color=[port_color if v >= 0 else "#9e2b25" for v in bt_returns],
                        text=[f"{v:+.1f}%" for v in bt_returns],
                        textposition="outside",
                    ))
                    fig_bt.update_layout(
                        title=f"{port_name} Portföy — Hisse Bazında Getiri",
                        paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                        font=dict(color=_theme()["ink"]),
                        yaxis=dict(gridcolor="#d8d0c0", ticksuffix="%"),
                        xaxis=dict(gridcolor="#d8d0c0"),
                        showlegend=False,
                        height=300, margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)

                    # Her hisse için fiyat + al/sat grafiği
                    st.markdown("**📈 Hisse Fiyat & AL/SAT Grafikleri**")
                    for tk, tk_res in valid_bt.items():
                        with st.expander(
                            f"{tk}  —  Getiri: %{tk_res['summary']['total_return_pct']:+.1f}  |  "
                            f"Kazanma: %{tk_res['summary']['win_rate']:.0f}  |  "
                            f"{tk_res['summary']['total_trades']} işlem",
                            expanded=False
                        ):
                            _render_price_chart_with_trades(
                                tk, bt_period_used, tk_res["trades"], height=380
                            )

    elif not run_backtest_btn:
        prev_runs = BacktestEngine.load_runs()
        smart_runs = [r for r in prev_runs if "_smart" in r.get("run_id","")]
        if smart_runs:
            st.info(f"Son sistem backtest: {smart_runs[0]['run_date']} — Sol menüden '▶ Backtest Çalıştır' ile yenile.")

    st.markdown("---")
    st.caption("⚠️ Sistem portföy seçimi teknik analize dayanır. Yatırım tavsiyesi değildir.")


def _render_price_chart_with_trades(ticker: str, period: str, trades: list, height: int = 420, market: str = "BIST"):
    """
    """
    try:
        yt = _yf_symbol(ticker, market)
        raw = yf.Ticker(yt).history(period=period, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty or "Close" not in raw.columns:
            st.warning(f"{ticker} için fiyat verisi alınamadı.")
            return
        raw.index = pd.to_datetime(raw.index).tz_localize(None)

        fig = go.Figure()

        # Fiyat çizgisi
        fig.add_trace(go.Scatter(
            x=raw.index, y=raw["Close"],
            mode="lines", name="Kapanış",
            line=dict(color="#8a8172", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.2f} ₺</b><extra></extra>",
        ))

        if not trades:
            fig.update_layout(
                title=f"{ticker} — Fiyat Grafiği (İşlem Yok)",
                paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                font=dict(color=_theme()["ink"]),
                yaxis=dict(gridcolor="#d8d0c0", title="Fiyat (₺)"),
                xaxis=dict(gridcolor="#d8d0c0"),
                height=height, margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            return

        EXIT_STYLE = {
            "TAKE_PROFIT":    ("#1d6f4e", "🟢 Hedef Fiyat"),
            "STOP_LOSS":      ("#9e2b25", "🔴 Stop-Loss"),
            "SAT_SINYAL":     ("#b45309", "🟠 SAT Sinyali"),
            "MAX_SURE":       ("#6d3f8e", "⏰ Max Süre"),
            "HALA_ACIK":      ("#27509e", "🔵 Açık Pozisyon"),
            "SMA200_KIRILIM": ("#9e2b25", "📉 SMA200 Kırılım"),
        }

        # Pozisyon şeritleri
        for t in trades:
            ret_val = t.get("return_pct", 0)
            color   = "#1d6f4e" if ret_val >= 0 else "#9e2b25"
            try:
                fig.add_vrect(
                    x0=t["entry_date"], x1=t["exit_date"],
                    fillcolor=color, opacity=0.07, line_width=0,
                    annotation_text=f"{ret_val:+.1f}%",
                    annotation_position="top left",
                    annotation=dict(font=dict(size=9, color=color)),
                )
            except Exception:
                pass

        # Giriş noktaları (yeşil ▲)
        entry_x   = [t["entry_date"]  for t in trades]
        entry_y   = [t["entry_price"] for t in trades]
        entry_txt = [
            f"<b>AL</b><br>{t['entry_date']}<br>Fiyat: {t['entry_price']:.2f} ₺<br>Skor: {t.get('entry_score', '-'):.0f}"
            for t in trades
        ]
        fig.add_trace(go.Scatter(
            x=entry_x, y=entry_y,
            mode="markers", name="AL ▲",
            marker=dict(symbol="triangle-up", size=14,
                        color="#1d6f4e", line=dict(color="#ffffff", width=1)),
            text=entry_txt, hoverinfo="text",
        ))

        # Çıkış noktaları (sebebe göre renkli ▼)
        seen_reasons = set()
        for t in trades:
            reason = t.get("exit_reason", "MAX_SURE")
            ecolor, ename = EXIT_STYLE.get(reason, ("#6b6357", reason))
            ret_val = t.get("return_pct", 0)
            show_legend = reason not in seen_reasons
            seen_reasons.add(reason)
            fig.add_trace(go.Scatter(
                x=[t["exit_date"]], y=[t["exit_price"]],
                mode="markers",
                name=ename,
                showlegend=show_legend,
                legendgroup=reason,
                marker=dict(symbol="triangle-down", size=12,
                            color=ecolor, line=dict(color="#ffffff", width=1)),
                text=[f"<b>{ename}</b><br>{t['exit_date']}<br>"
                      f"Fiyat: {t['exit_price']:.2f} ₺<br>"
                      f"Getiri: {ret_val:+.2f}%<br>Süre: {t.get('hold_days',0)} gün"],
                hoverinfo="text",
            ))

        fig.update_layout(
            title=f"{ticker} — Fiyat & AL/SAT Noktaları",
            paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
            font=dict(color=_theme()["ink"]),
            yaxis=dict(gridcolor="#d8d0c0", title="Fiyat (₺)"),
            xaxis=dict(gridcolor="#d8d0c0", rangeslider=dict(visible=False)),
            height=height, margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.warning(f"{ticker} fiyat grafiği oluşturulamadı: {exc}")


# SKOR DOĞRULAMA RAPORU SAYFASI

def render_validation_page(ui_lang: str):
    """Sinyal takip paneli — AL/SAT sinyallerini günlük, haftalık, aylık gösterir."""
    st.markdown("# " + ("Sinyal Takip Paneli" if ui_lang == "TR" else "Signal Tracker"))
    st.caption(
        "Sistemin AL ve SAT dediği hisseleri günlük, haftalık ve aylık olarak takip edin. "
        "Her sinyalin ardından fiyat nasıl hareket etti?"
        if ui_lang == "TR" else
        "Track BUY and SELL signals daily, weekly and monthly. "
        "How did the price move after each signal?"
    )

    # Sidebar: sadece güncelle + backfill
    with st.sidebar:
        st.markdown("## " + ("Sinyal Takip" if ui_lang == "TR" else "Signal Tracker"))

        if st.button(
            "Fiyatlari Guncelle" if ui_lang == "TR" else "Update Prices",
            use_container_width=True, type="primary",
            help="Bekleyen sinyallerin güncel fiyatlarını kontrol eder"
                 if ui_lang == "TR" else "Checks current prices for pending signals"
        ):
            with st.spinner("Fiyatlar kontrol ediliyor..." if ui_lang == "TR" else "Checking prices..."):
                _history_db.check_pending_signals()
            st.success("Guncellendi!" if ui_lang == "TR" else "Updated!")
            st.rerun()

        st.markdown("---")

        with st.expander(
            "Gecmis Veri Yukle" if ui_lang == "TR" else "Load Historical Data",
            expanded=False
        ):
            st.caption(
                "BIST30 hisseleri icin gecmise donuk sinyal testi yapar."
                if ui_lang == "TR" else "Runs historical signal test for BIST30 stocks."
            )
            backfill_btn = st.button(
                "Baslat" if ui_lang == "TR" else "Start",
                use_container_width=True,
                help="6 ay geriye gidip haftalık test noktaları oluşturur"
                     if ui_lang == "TR" else "Creates weekly test points going back 6 months"
            )

    # Backfill çalıştır (sabit 6 ay, haftalık)
    if backfill_btn:
        bist30 = BIST_SCAN_UNIVERSE[:30]
        progress_bar = st.progress(0, text="Baslatiliyor..." if ui_lang == "TR" else "Starting...")
        status_area = st.empty()

        def _bf_progress(ticker, step, total):
            pct = min(step / max(total, 1), 1.0)
            progress_bar.progress(pct, text=f"{ticker} ({step}/{total})")
            status_area.info(f"{ticker} analiz ediliyor..." if ui_lang == "TR" else f"Analyzing {ticker}...")

        added = _history_db.run_backfill(bist30, months_back=6,
                                         interval_days=7, progress_cb=_bf_progress)
        progress_bar.progress(1.0, text="Tamamlandi!" if ui_lang == "TR" else "Complete!")
        status_area.success(
            f"**{added}** sinyal eklendi."
            if ui_lang == "TR" else f"**{added}** signals added."
        )

    # Rapor verisini yükle
    all_data = _history_db.get_validation_report()
    if not all_data:
        st.info(
            "Henuz sinyal verisi yok. Hisse analiz ettiginizde sinyaller otomatik kaydedilir. "
            "Ayrica kenar cubugundaki **Gecmis Veri Yukle** ile toplu test yapabilirsiniz."
            if ui_lang == "TR" else
            "No signal data yet. Signals are auto-recorded when you analyze stocks. "
            "You can also use **Load Historical Data** in the sidebar for bulk testing."
        )
        return

    df_all = pd.DataFrame(all_data)
    df_all["signal_dt"] = pd.to_datetime(df_all["signal_date"].str[:10], errors="coerce")
    now = pd.Timestamp.now().normalize()

    # Zaman Dilimi Seçici (Günlük / Haftalık / Aylık)
    view_mode = st.radio(
        "Zaman Dilimi" if ui_lang == "TR" else "Time Period",
        ["Gunluk", "Haftalik", "Aylik"] if ui_lang == "TR" else ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="val_view_mode"
    )

    # Filtreleme aralığını belirle
    if view_mode in ("Gunluk", "Daily"):
        date_start = now - pd.Timedelta(days=1)
        period_label = "Bugun" if ui_lang == "TR" else "Today"
        ret_col, res_col = "return_1d_pct", "result_1d"
        period_desc = "1 Gun" if ui_lang == "TR" else "1 Day"
    elif view_mode in ("Haftalik", "Weekly"):
        date_start = now - pd.Timedelta(days=7)
        period_label = "Son 7 Gun" if ui_lang == "TR" else "Last 7 Days"
        ret_col, res_col = "return_7d_pct", "result_7d"
        period_desc = "7 Gun" if ui_lang == "TR" else "7 Days"
    else:
        date_start = now - pd.Timedelta(days=30)
        period_label = "Son 30 Gun" if ui_lang == "TR" else "Last 30 Days"
        ret_col, res_col = "return_30d_pct", "result_30d"
        period_desc = "30 Gun" if ui_lang == "TR" else "30 Days"

    df_period = df_all[df_all["signal_dt"] >= date_start].copy()

    # Özet Metrikler
    st.markdown("---")
    st.markdown(f"### {period_label}")

    al_sinyalleri = df_period[df_period["signal"].isin(["AL", "GUCLU AL"])]
    sat_sinyalleri = df_period[df_period["signal"].isin(["SAT", "GUCLU SAT"])]

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Toplam Sinyal" if ui_lang == "TR" else "Total Signals", len(df_period))
    with mc2:
        st.metric("AL Sinyali" if ui_lang == "TR" else "BUY Signals", len(al_sinyalleri))
    with mc3:
        st.metric("SAT Sinyali" if ui_lang == "TR" else "SELL Signals", len(sat_sinyalleri))
    with mc4:
        completed = df_period[df_period[res_col].isin(["Basarili", "Basarisiz"])]
        if not completed.empty:
            acc = round(len(completed[completed[res_col] == "Basarili"]) / len(completed) * 100, 1)
            st.metric(
                f"Dogruluk ({period_desc})" if ui_lang == "TR" else f"Accuracy ({period_desc})",
                f"{acc}%"
            )
        else:
            st.metric(
                f"Dogruluk ({period_desc})" if ui_lang == "TR" else f"Accuracy ({period_desc})",
                "Bekliyor" if ui_lang == "TR" else "Pending"
            )

    # Skor Sistemi Karşılaştırması — v1 (eski, kontrarian) vs v2 (momentum, 2026-07 kalibrasyonu)
    # Tüm zamanlardaki kayıtlar üzerinden hesaplanır (dönem filtresinden bağımsız),
    # çünkü v1 kayıtları geçmişte kaldı — adil karşılaştırma tüm veriyle olur.
    if "score_version" in df_all.columns and df_all["score_version"].nunique() > 1:
        st.markdown("---")
        st.markdown("### " + (
            "Skor Sistemi Karsilastirmasi (v1 eski / v2 momentum)"
            if ui_lang == "TR" else "Scoring System Comparison (v1 old / v2 momentum)"
        ))
        st.caption(
            "v2, 2026-07 kalibrasyonuyla gelen momentum-hizali skorlama. "
            "Iki sistemin gercek sinyal isabetleri asagida — veri biriktikce netlesir."
            if ui_lang == "TR" else
            "v2 is the momentum-aligned scoring from the 2026-07 calibration. "
            "Real-world hit rates of both systems below — clarity improves as data accumulates."
        )
        ver_rows = []
        for ver in sorted(df_all["score_version"].dropna().unique()):
            dv = df_all[df_all["score_version"] == ver]
            row = {("Sistem" if ui_lang == "TR" else "System"): ver,
                   ("Sinyal" if ui_lang == "TR" else "Signals"): len(dv)}
            for p_key, p_lbl in [("7d", "7g"), ("14d", "14g"), ("30d", "30g")]:
                rc = f"result_{p_key}"
                if rc in dv.columns:
                    done = dv[dv[rc].isin(["Basarili", "Basarisiz"])]
                    row[p_lbl if ui_lang == "TR" else p_key] = (
                        f"%{len(done[done[rc] == 'Basarili']) / len(done) * 100:.0f} ({len(done)})"
                        if len(done) > 0 else "—"
                    )
            ver_rows.append(row)
        st.dataframe(pd.DataFrame(ver_rows), use_container_width=True, hide_index=True)

    # AL Sinyalleri Tablosu
    if not al_sinyalleri.empty:
        st.markdown("---")
        st.markdown("### " + ("AL Sinyalleri" if ui_lang == "TR" else "BUY Signals"))

        al_display = al_sinyalleri[["ticker", "signal", "score", "signal_date", "signal_price"]].copy()
        al_display["signal_date"] = al_display["signal_date"].str[:10]
        # Tüm periyot getirilerini ekle
        for p_key, p_label in [("1d","1g"), ("3d","3g"), ("7d","7g"), ("14d","14g"), ("30d","30g")]:
            r_col = f"return_{p_key}_pct"
            res_c = f"result_{p_key}"
            if r_col in al_sinyalleri.columns:
                al_display[p_label] = al_sinyalleri.apply(
                    lambda row: f"{row[r_col]:+.2f}%" if pd.notna(row[r_col]) else
                                ("Bekliyor" if ui_lang == "TR" else "Pending"), axis=1
                )
            else:
                al_display[p_label] = "—"

        col_rename = {
            "ticker": "Hisse" if ui_lang == "TR" else "Ticker",
            "signal": "Sinyal" if ui_lang == "TR" else "Signal",
            "score": "Skor" if ui_lang == "TR" else "Score",
            "signal_date": "Tarih" if ui_lang == "TR" else "Date",
            "signal_price": "Fiyat" if ui_lang == "TR" else "Price",
        }
        al_display.rename(columns=col_rename, inplace=True)
        st.dataframe(al_display, use_container_width=True, hide_index=True)
    else:
        st.info(
            f"{period_label} icinde AL sinyali yok."
            if ui_lang == "TR" else f"No BUY signals in {period_label}."
        )

    # SAT Sinyalleri Tablosu
    if not sat_sinyalleri.empty:
        st.markdown("---")
        st.markdown("### " + ("SAT Sinyalleri" if ui_lang == "TR" else "SELL Signals"))

        sat_display = sat_sinyalleri[["ticker", "signal", "score", "signal_date", "signal_price"]].copy()
        sat_display["signal_date"] = sat_display["signal_date"].str[:10]
        for p_key, p_label in [("1d","1g"), ("3d","3g"), ("7d","7g"), ("14d","14g"), ("30d","30g")]:
            r_col = f"return_{p_key}_pct"
            if r_col in sat_sinyalleri.columns:
                sat_display[p_label] = sat_sinyalleri.apply(
                    lambda row: f"{row[r_col]:+.2f}%" if pd.notna(row[r_col]) else
                                ("Bekliyor" if ui_lang == "TR" else "Pending"), axis=1
                )
            else:
                sat_display[p_label] = "—"

        sat_display.rename(columns=col_rename, inplace=True)
        st.dataframe(sat_display, use_container_width=True, hide_index=True)
    else:
        st.info(
            f"{period_label} icinde SAT sinyali yok."
            if ui_lang == "TR" else f"No SELL signals in {period_label}."
        )

    # Genel Performans Özeti
    st.markdown("---")
    st.markdown("### " + ("Genel Performans" if ui_lang == "TR" else "Overall Performance"))

    _periods = [("1d", "1g"), ("3d", "3g"), ("7d", "7g"), ("14d", "14g"), ("30d", "30g")]
    _period_dfs = {}
    for p_key, p_label in _periods:
        result_col = f"result_{p_key}"
        if result_col in df_all.columns:
            _period_dfs[p_key] = df_all[df_all[result_col].isin(["Basarili", "Basarisiz"])].copy()
        else:
            _period_dfs[p_key] = pd.DataFrame()

    has_data = any(not v.empty for v in _period_dfs.values())
    if has_data:
        signal_order = ["GUCLU AL", "AL", "SAT", "GUCLU SAT"]
        rows_table = []
        for sig in signal_order + ["GENEL"]:
            row = {"Sinyal": sig}
            for p_key, p_label in _periods:
                pdf = _period_dfs[p_key]
                if pdf.empty:
                    row[f"{p_label} %"] = "—"
                    row[f"{p_label} Getiri"] = "—"
                    continue
                subset = pdf if sig == "GENEL" else pdf[pdf["signal"] == sig]
                n = len(subset)
                if n == 0:
                    row[f"{p_label} %"] = "—"
                    row[f"{p_label} Getiri"] = "—"
                    continue
                result_col = f"result_{p_key}"
                return_col = f"return_{p_key}_pct"
                acc = round(len(subset[subset[result_col] == "Basarili"]) / n * 100, 1)
                ret = round(subset[return_col].mean(), 2) if return_col in subset.columns else 0
                row[f"{p_label} %"] = f"{acc}% ({n})"
                row[f"{p_label} Getiri"] = f"{ret:+.2f}%"
            rows_table.append(row)

        df_table = pd.DataFrame(rows_table)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    # Sinyal Dağılımı Grafiği
    if not df_all.empty:
        st.markdown("---")
        st.markdown("### " + ("Sinyal Dagilimi" if ui_lang == "TR" else "Signal Distribution"))

        sig_counts = df_all["signal"].value_counts()
        colors_map = {
            "GUCLU AL": "#1d6f4e", "AL": "#3e8e6c",
            "SAT": "#b45309", "GUCLU SAT": "#9e2b25",
        }
        fig_pie = go.Figure(data=[go.Pie(
            labels=sig_counts.index.tolist(),
            values=sig_counts.values.tolist(),
            marker=dict(colors=[colors_map.get(s, "#6b6357") for s in sig_counts.index]),
            textinfo="label+percent+value",
            hole=0.4,
        )])
        fig_pie.update_layout(
            paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
            font=dict(color=_theme()["ink"]), height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(bgcolor="#efe9db"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Ham Veri (Expandable)
    if not df_all.empty:
        st.markdown("---")
        with st.expander(
            "Tum Sinyal Verileri (Ham)" if ui_lang == "TR" else "All Signal Data (Raw)",
            expanded=False
        ):
            display_cols = ["ticker", "signal", "score", "signal_date", "signal_price",
                           "return_1d_pct", "result_1d", "return_3d_pct", "result_3d",
                           "return_7d_pct", "result_7d", "return_14d_pct", "result_14d",
                           "return_30d_pct", "result_30d", "source"]
            available_cols = [c for c in display_cols if c in df_all.columns]
            st.dataframe(df_all[available_cols], use_container_width=True, hide_index=True)


# US SİNYAL TAKİP SAYFASI

def render_us_validation_page(ui_lang: str):
    """US piyasası sinyal takip paneli — AL/SAT sinyallerini günlük, haftalık, aylık gösterir."""
    st.markdown("# " + ("US Sinyal Takip" if ui_lang == "TR" else "US Signal Tracker"))
    st.caption(
        "US piyasasinda verilen AL ve SAT sinyallerini gunluk, haftalik ve aylik takip edin."
        if ui_lang == "TR" else
        "Track BUY and SELL signals for US stocks on a daily, weekly and monthly basis."
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## " + ("US Sinyal Takip" if ui_lang == "TR" else "US Signal Tracker"))
        if st.button(
            "Fiyatlari Guncelle" if ui_lang == "TR" else "Update Prices",
            use_container_width=True, type="primary",
            key="us_val_update"
        ):
            with st.spinner("Fiyatlar kontrol ediliyor..." if ui_lang == "TR" else "Checking prices..."):
                _history_db.check_pending_signals()
            st.success("Guncellendi!" if ui_lang == "TR" else "Updated!")
            st.rerun()

    # Veri yükle (sadece US kayıtları)
    all_data = _history_db.get_validation_report()
    us_data = [r for r in all_data if r.get("source", "").endswith("_us") or r.get("source") == "live_us"]
    if not us_data:
        st.info(
            "Henuz US sinyal verisi yok. **US Analiz** sayfasindan hisse analiz ettiginizde "
            "sinyaller otomatik kaydedilir."
            if ui_lang == "TR" else
            "No US signal data yet. Signals are auto-recorded when you analyze stocks on the **US Analiz** page."
        )
        return

    df_all = pd.DataFrame(us_data)
    df_all["signal_dt"] = pd.to_datetime(df_all["signal_date"].str[:10], errors="coerce")
    now = pd.Timestamp.now().normalize()

    # Zaman Dilimi Seçici
    view_mode = st.radio(
        "Time Period",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="us_val_view_mode"
    )

    if view_mode == "Daily":
        date_start = now - pd.Timedelta(days=1)
        period_label, period_desc = "Today", "1 Day"
        ret_col, res_col = "return_1d_pct", "result_1d"
    elif view_mode == "Weekly":
        date_start = now - pd.Timedelta(days=7)
        period_label, period_desc = "Last 7 Days", "7 Days"
        ret_col, res_col = "return_7d_pct", "result_7d"
    else:
        date_start = now - pd.Timedelta(days=30)
        period_label, period_desc = "Last 30 Days", "30 Days"
        ret_col, res_col = "return_30d_pct", "result_30d"

    df_period = df_all[df_all["signal_dt"] >= date_start].copy()

    # Özet Metrikler
    st.markdown("---")
    st.markdown(f"### {period_label}")

    buy_sigs  = df_period[df_period["signal"].isin(["AL", "GUCLU AL"])]
    sell_sigs = df_period[df_period["signal"].isin(["SAT", "GUCLU SAT"])]

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Signals", len(df_period))
    mc2.metric("BUY Signals", len(buy_sigs))
    mc3.metric("SELL Signals", len(sell_sigs))
    completed = df_period[df_period[res_col].isin(["Basarili", "Basarisiz"])]
    if not completed.empty:
        acc = round(len(completed[completed[res_col] == "Basarili"]) / len(completed) * 100, 1)
        mc4.metric(f"Accuracy ({period_desc})", f"{acc}%")
    else:
        mc4.metric(f"Accuracy ({period_desc})", "Pending")

    # BUY Signals Table
    _p_cols = [("1d","1D"), ("3d","3D"), ("7d","7D"), ("14d","14D"), ("30d","30D")]
    if not buy_sigs.empty:
        st.markdown("---")
        st.markdown("### BUY Signals")
        disp = buy_sigs[["ticker", "signal", "score", "signal_date", "signal_price"]].copy()
        disp["signal_date"] = disp["signal_date"].str[:10]
        for p_key, p_label in _p_cols:
            r_col = f"return_{p_key}_pct"
            if r_col in buy_sigs.columns:
                disp[p_label] = buy_sigs.apply(
                    lambda row: f"{row[r_col]:+.2f}%" if pd.notna(row[r_col]) else "Pending", axis=1
                )
            else:
                disp[p_label] = "—"
        disp.rename(columns={"ticker": "Ticker", "signal": "Signal", "score": "Score",
                             "signal_date": "Date", "signal_price": "Price"}, inplace=True)
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info(f"No BUY signals in {period_label}.")

    # SELL Signals Table
    if not sell_sigs.empty:
        st.markdown("---")
        st.markdown("### SELL Signals")
        disp = sell_sigs[["ticker", "signal", "score", "signal_date", "signal_price"]].copy()
        disp["signal_date"] = disp["signal_date"].str[:10]
        for p_key, p_label in _p_cols:
            r_col = f"return_{p_key}_pct"
            if r_col in sell_sigs.columns:
                disp[p_label] = sell_sigs.apply(
                    lambda row: f"{row[r_col]:+.2f}%" if pd.notna(row[r_col]) else "Pending", axis=1
                )
            else:
                disp[p_label] = "—"
        disp.rename(columns={"ticker": "Ticker", "signal": "Signal", "score": "Score",
                             "signal_date": "Date", "signal_price": "Price"}, inplace=True)
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info(f"No SELL signals in {period_label}.")

    # Overall Performance
    st.markdown("---")
    st.markdown("### Overall Performance")
    _periods = [("1d", "1D"), ("3d", "3D"), ("7d", "7D"), ("14d", "14D"), ("30d", "30D")]
    _period_dfs = {}
    for p_key, p_label in _periods:
        result_col = f"result_{p_key}"
        if result_col in df_all.columns:
            _period_dfs[p_key] = df_all[df_all[result_col].isin(["Basarili", "Basarisiz"])].copy()
        else:
            _period_dfs[p_key] = pd.DataFrame()

    if any(not v.empty for v in _period_dfs.values()):
        signal_order = ["GUCLU AL", "AL", "SAT", "GUCLU SAT"]
        rows_table = []
        for sig in signal_order + ["TOTAL"]:
            row = {"Signal": sig}
            for p_key, p_label in _periods:
                pdf = _period_dfs[p_key]
                if pdf.empty:
                    row[f"{p_label} %"] = "—"
                    row[f"{p_label} Ret"] = "—"
                    continue
                subset = pdf if sig == "TOTAL" else pdf[pdf["signal"] == sig]
                n = len(subset)
                if n == 0:
                    row[f"{p_label} %"] = "—"
                    row[f"{p_label} Ret"] = "—"
                    continue
                result_col = f"result_{p_key}"
                return_col = f"return_{p_key}_pct"
                acc = round(len(subset[subset[result_col] == "Basarili"]) / n * 100, 1)
                ret = round(subset[return_col].mean(), 2) if return_col in subset.columns else 0
                row[f"{p_label} %"] = f"{acc}% ({n})"
                row[f"{p_label} Ret"] = f"{ret:+.2f}%"
            rows_table.append(row)
        st.dataframe(pd.DataFrame(rows_table), use_container_width=True, hide_index=True)

    # Raw Data
    if not df_all.empty:
        st.markdown("---")
        with st.expander("All Signal Data (Raw)", expanded=False):
            display_cols = ["ticker", "signal", "score", "signal_date", "signal_price",
                           "return_1d_pct", "result_1d", "return_3d_pct", "result_3d",
                           "return_7d_pct", "result_7d", "return_14d_pct", "result_14d",
                           "return_30d_pct", "result_30d", "source"]
            available_cols = [c for c in display_cols if c in df_all.columns]
            st.dataframe(df_all[available_cols], use_container_width=True, hide_index=True)


# STRATEJİ ZAMAN MAKİNESİ SAYFASI

def render_time_machine_page(ui_lang: str):
    """
    Strateji Zaman Makinesi sayfası — tam özellikli.
    PIT analizi, AL/SAT grafikleri, equity curve, işlem geçmişi,
    5 farklı portföy stili + kullanıcı özel portföyü.
    """
    st.markdown("# ⏳ Strateji Zaman Makinesi")
    st.caption(
        "Bugünkü stratejimiz geçmişte uygulansaydı ne olurdu? "
        "PIT (Point-in-Time) analizi + BacktestEngine ile gerçek AL/SAT simülasyonu. "
        "**Anlık değil, sürekli başarı önemlidir.**"
    )
    st.markdown("---")

    _styles = TimeMachineEngine.PORTFOLIO_STYLES
    _style_keys = [k for k in _styles if k != "custom"]

    # Sidebar Kontroller
    with st.sidebar:
        st.markdown("## Zaman Makinesi Ayarları")
        tm_years = st.selectbox(
            "Kaç Yıl Geriye?", [1, 2, 3, 5], index=2,
            format_func=lambda x: f"{x} Yıl",
            key="tm_years_sel"
        )
        tm_market = st.radio(
            "Piyasa", ["BIST", "US"], index=0,
            key="tm_market_sel"
        )
        tm_style = st.radio(
            "Portföy Stili",
            _style_keys,
            index=0,
            format_func=lambda x: _styles[x]["label"],
            key="tm_style_sel"
        )
        st.caption(f"_{_styles[tm_style]['desc']}_")

        st.markdown("---")
        st.markdown("### Backtest Ayarları")
        tm_bt_mode = st.radio(
            "Strateji Modu",
            ["swing", "trend", "universal", "investor", "buyhold"],
            index=2,
            format_func=lambda x: BacktestEngine.MODES[x]["label"],
            key="tm_bt_mode_sel"
        )
        tm_bt_news = st.checkbox("📰 Haber Filtresi", value=True, key="tm_bt_news")

        st.markdown("---")
        run_pit_btn = st.button(
            "🕰️ Zaman Makinesi Çalıştır",
            type="primary", use_container_width=True,
            key="tm_run_btn",
            help=f"{tm_years} yıl önceki verilerle strateji testi + backtest"
        )
        run_live_btn = st.button(
            "📊 Bugünkü Canlı Portföy",
            use_container_width=True,
            key="tm_live_btn",
        )

        st.markdown("---")
        st.markdown("### 🎯 Özel Portföy")
        _all_tickers = BIST_SCAN_UNIVERSE if tm_market == "BIST" else US_POPULAR_TICKERS
        custom_tickers = st.multiselect(
            "Kendi hisselerini seç",
            options=sorted(_all_tickers),
            default=[],
            key="tm_custom_tickers",
            placeholder="Hisse seç...",
        )
        run_custom_btn = st.button(
            "🎯 Özel Portföy Test Et",
            use_container_width=True,
            key="tm_custom_btn",
            disabled=len(custom_tickers) == 0,
        )

        st.markdown("---")
        st.caption(
            "**Nasıl Çalışır?**\n\n"
            "1️⃣ Seçilen yıl kadar geriye gider\n"
            "2️⃣ O günkü teknik verilerle portföy seçer\n"
            "3️⃣ BacktestEngine ile AL/SAT simülasyonu çalıştırır\n"
            "4️⃣ Fiyat + AL/SAT grafikleri, equity curve gösterir\n"
            "5️⃣ Stratejiye 0-100 arası not verir"
        )

    # Önceki Çalışmaları Yükle
    prev_runs = TimeMachineEngine.load_previous_runs(tm_market)

    # ÖZEL PORTFÖY TESTİ
    if run_custom_btn and custom_tickers:
        st.markdown("## 🎯 Özel Portföy — Zaman Makinesi Testi")
        _tm_run_analysis(
            custom_tickers, tm_years, tm_market, "custom",
            tm_bt_mode, tm_bt_news, is_custom=True
        )

    # PIT ANALİZİ ÇALIŞTIR
    if run_pit_btn:
        style_label = _styles[tm_style]["label"]
        st.markdown(f"## 🕰️ {tm_years} Yıl Öncesi — {style_label} Portföy Testi")

        progress = st.progress(0, text="PIT skorları hesaplanıyor...")
        status_txt = st.empty()

        def _progress(msg, pct):
            progress.progress(pct, text=msg)
            status_txt.caption(msg)

        with st.spinner(f"{tm_years} yıl önceki veriler analiz ediliyor..."):
            result = TimeMachineEngine.run_full_pit(
                years_back=tm_years,
                market=tm_market,
                style=tm_style,
                progress_cb=_progress,
            )

        progress.progress(1.0, text="PIT analizi tamamlandı!")
        status_txt.empty()

        st.session_state["tm_last_result"] = result

        # Günlük snapshot kaydet
        try:
            TimeMachineEngine.save_daily_snapshot(result["run_id"], tm_market)
        except Exception:
            pass

        # BacktestEngine ile AL/SAT simülasyonu
        picks = result.get("picks", [])
        if picks:
            _tm_run_backtest_for_picks(picks, tm_years, tm_market, tm_bt_mode, tm_bt_news)

    # BUGÜNKÜ CANLI PORTFÖY
    if run_live_btn:
        st.markdown("## 📊 Bugünkü Canlı Portföy")
        with st.spinner("Güncel veriler taranıyor..."):
            scan_results = PortfolioScanner.scan_all(force=False)
            portfolios = SmartPortfolioBuilder.build(scan_results)
            live_stocks = portfolios["aggressive"] if tm_style == "aggressive" else portfolios["defensive"]

        if live_stocks:
            st.success(f"{len(live_stocks)} hisse bugünkü kriterlerle seçildi ({_styles[tm_style]['label']})")

            live_df = pd.DataFrame(live_stocks)
            display_cols = ["ticker", "score", "rsi", "adx", "atr_pct", "momentum_1m", "price", "reason"]
            available_cols = [c for c in display_cols if c in live_df.columns]
            st.dataframe(
                live_df[available_cols].rename(columns={
                    "ticker": "Hisse", "score": "Skor", "rsi": "RSI",
                    "adx": "ADX", "atr_pct": "ATR%", "momentum_1m": "1Ay%",
                    "price": "Fiyat", "reason": "Seçim Nedeni",
                }),
                use_container_width=True, hide_index=True,
            )

            st.info(
                "⏳ **Kritik İlke:** Bugün oluşturulan portföyden hemen çıkarım yapmak mantıksızdır. "
                "Strateji ancak zaman içinde kanıtlanır."
            )
        else:
            st.warning("Kriterlere uyan hisse bulunamadı.")
        st.markdown("---")

    # PIT SONUÇLARINI GÖSTER (session_state'den)
    result = st.session_state.get("tm_last_result")
    if result:
        _tm_render_report(result)

    # ÖNCEKİ ÇALIŞMALAR
    if prev_runs:
        st.markdown("---")
        with st.expander(f"📜 Önceki Çalışmalar ({len(prev_runs)} kayıt)", expanded=False):
            for run in prev_runs[:10]:
                _g = run.get("back_test_grade", 0)
                gc = "#1d6f4e" if _g >= 65 else "#b45309" if _g >= 40 else "#9e2b25"
                st.markdown(
                    f"<div style='background:#efe9db;border:1px solid #d8d0c0;"
                    f"border-radius:8px;padding:10px 14px;margin-bottom:6px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span style='font-weight:700;color:#1a1712'>"
                    f"{run.get('pit_date','')} → Bugün "
                    f"({run.get('portfolio','').title()})</span>"
                    f"<span style='color:{gc};font-weight:700;font-size:18px'>"
                    f"{_g:.0f}/100</span>"
                    f"</div>"
                    f"<div style='font-size:12px;color:#6b6357;margin-top:4px'>"
                    f"Portföy: %{run.get('portfolio_return_pct',0):+.1f} · "
                    f"Endeks: %{run.get('benchmark_return_pct',0):+.1f} · "
                    f"Alfa: %{run.get('alpha_pct',0):+.1f} · "
                    f"{run.get('stock_count',0)} hisse · "
                    f"{run.get('run_date','')}"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.caption(
        "⚠️ Zaman Makinesi geçmiş performansa dayalı strateji değerlendirmesidir. "
        "Geçmiş başarı gelecek başarıyı garanti etmez. Yatırım tavsiyesi değildir."
    )


# ZAMAN MAKİNESİ YARDIMCI FONKSİYONLARI

def _tm_run_analysis(tickers: list, years_back: int, market: str,
                     style: str, bt_mode: str, bt_news: bool,
                     is_custom: bool = False):
    """Özel portföy veya herhangi bir ticker listesi için tam analiz."""
    pit_date = datetime.now() - timedelta(days=years_back * 365)

    progress = st.progress(0, text="Hisseler analiz ediliyor...")

    # PIT skorları
    progress.progress(0.2, text="PIT teknik skorlar hesaplanıyor...")
    pit_results = TimeMachineEngine._compute_pit_scores(tickers, pit_date, market)

    if is_custom:
        # Özel portföy: filtre yok, tüm hisseleri al
        picks = [r for r in pit_results if r["ticker"] in [t.upper() for t in tickers]]
    else:
        picks = TimeMachineEngine._filter_portfolio(pit_results, style)

    if not picks:
        progress.progress(1.0, text="Tamamlandı")
        st.warning("Seçilen hisseler için PIT verisi bulunamadı.")
        return

    # Performans ölçümü
    progress.progress(0.5, text="Gerçek performans ölçülüyor...")
    picks = TimeMachineEngine._measure_performance(picks, pit_date, market)

    # Benchmark
    progress.progress(0.7, text="Benchmark karşılaştırılıyor...")
    bench_ret = TimeMachineEngine._benchmark_return(pit_date, market)
    grade = TimeMachineEngine._compute_grade(picks, bench_ret)

    returns = [p.get("return_pct", 0) for p in picks]
    avg_ret = sum(returns) / len(returns) if returns else 0
    alpha = round(avg_ret - bench_ret, 1)

    result = {
        "run_id": f"tm_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "pit_date": pit_date.strftime("%Y-%m-%d"),
        "picks": picks,
        "avg_return": round(avg_ret, 1),
        "bench_return": bench_ret,
        "alpha": alpha,
        "grade": grade,
        "style": style,
        "market": market,
    }
    st.session_state["tm_last_result"] = result

    progress.progress(0.85, text="Backtest çalıştırılıyor...")

    # Backtest
    _tm_run_backtest_for_picks(picks, years_back, market, bt_mode, bt_news)

    progress.progress(1.0, text="Tamamlandı!")


def _tm_run_backtest_for_picks(picks: list, years_back: int, market: str,
                               bt_mode: str, bt_news: bool):
    """PIT portföyündeki hisseler için BacktestEngine ile AL/SAT simülasyonu çalıştırır."""
    period = f"{years_back}y"
    run_id = f"tm_bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    tickers = [p["ticker"] for p in picks]
    bt_results = {}

    prog = st.progress(0, text="Backtest başlatılıyor...")
    for i, ticker in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"Backtest: {ticker} ({i+1}/{len(tickers)})")
        try:
            trades, summary, daily = BacktestEngine._run_single(
                ticker, period, run_id,
                mode=bt_mode, use_news=bt_news,
                enable_short=False, enable_scaling=True,
                market=market,
            )
            BacktestEngine._save(
                DB_PATH, run_id, run_date, ticker, period,
                trades, summary, daily
            )
            bt_results[ticker] = {"trades": trades, "summary": summary, "daily": daily}
        except Exception as exc:
            bt_results[ticker] = {"trades": [], "summary": {}, "daily": [], "error": str(exc)}

    prog.progress(1.0, text="Backtest tamamlandı!")
    st.session_state["tm_bt_results"] = bt_results
    st.session_state["tm_bt_period"]  = period
    st.session_state["tm_bt_run_id"]  = run_id


def _tm_render_report(result: dict):
    """PIT sonuçlarını + backtest sonuçlarını birlikte gösterir."""
    picks = result.get("picks", [])
    report = TimeMachineEngine.generate_report(result)
    style_info = TimeMachineEngine.PORTFOLIO_STYLES.get(result.get("style", "aggressive"), {})
    style_color = style_info.get("color", "#27509e")

    st.markdown("---")
    st.markdown(f"## 📋 Strateji Raporu — {result['pit_date']}'den Bugüne")

    # Metrik Kartları
    m1, m2, m3, m4, m5 = st.columns(5)
    grade = result["grade"]
    grade_color = "#1d6f4e" if grade >= 65 else "#b45309" if grade >= 40 else "#9e2b25"
    m1.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value' style='color:{grade_color}'>{grade:.0f}</div>"
        f"<div class='metric-label'>Strateji Notu</div>"
        f"</div>", unsafe_allow_html=True
    )
    m2.metric("Portföy Getirisi", f"%{result['avg_return']:+.1f}")
    m3.metric("Endeks Getirisi", f"%{result['bench_return']:+.1f}")
    alpha_color = "normal" if result["alpha"] >= 0 else "inverse"
    m4.metric("Alfa", f"%{result['alpha']:+.1f}",
              delta=f"%{result['alpha']:+.1f}", delta_color=alpha_color)
    m5.metric("Kazanma Oranı", f"%{report['win_rate']:.0f}")

    # Gauge Chart
    try:
        fig_gauge = create_gauge_chart(grade, title="Strateji Zaman Makinesi Notu")
        st.plotly_chart(fig_gauge, use_container_width=True)
    except Exception:
        pass

    # ANA SEKMELER
    tab_charts, tab_grade, tab_consist, tab_risk, tab_daily = st.tabs([
        "📈 AL/SAT Grafikleri", "📊 Geriye Dönük Skor",
        "🔍 Tutarlılık Analizi", "⚠️ Riskli Hisseler", "📅 Günlük Gözlem"
    ])

    # TAB 1: AL/SAT GRAFİKLERİ + EQUITY + İŞLEM GEÇMİŞİ
    with tab_charts:
        bt_results = st.session_state.get("tm_bt_results", {})
        bt_period = st.session_state.get("tm_bt_period", "3y")
        bt_run_id = st.session_state.get("tm_bt_run_id", "")

        if not bt_results:
            st.info("Backtest sonuçları henüz yok. 'Zaman Makinesi Çalıştır' butonuna basın.")
        else:
            valid_bt = {k: v for k, v in bt_results.items()
                        if "error" not in v and v.get("summary", {}).get("total_trades", 0) > 0}

            if valid_bt:
                # Portföy Özet Metrikleri
                all_rets = [v["summary"]["total_return_pct"] for v in valid_bt.values()]
                all_wrs  = [v["summary"]["win_rate"] for v in valid_bt.values()]
                port_ret = sum(all_rets) / len(all_rets)
                avg_wr   = sum(all_wrs) / len(all_wrs)
                total_trades = sum(v["summary"]["total_trades"] for v in valid_bt.values())
                best_t  = max(valid_bt.items(), key=lambda x: x[1]["summary"]["total_return_pct"])
                worst_t = min(valid_bt.items(), key=lambda x: x[1]["summary"]["total_return_pct"])

                bm1, bm2, bm3, bm4, bm5 = st.columns(5)
                bm1.metric("Backtest Portföy Getirisi", f"%{port_ret:+.1f}")
                bm2.metric("Ort. Kazanma Oranı", f"%{avg_wr:.0f}")
                bm3.metric("Toplam İşlem", f"{total_trades}")
                bm4.metric("En İyi", best_t[0], f"%{best_t[1]['summary']['total_return_pct']:+.1f}")
                bm5.metric("En Kötü", worst_t[0], f"%{worst_t[1]['summary']['total_return_pct']:+.1f}")

                # Hisse Bazlı Getiri Bar Chart
                bt_labels = list(valid_bt.keys())
                bt_returns = [valid_bt[t]["summary"]["total_return_pct"] for t in bt_labels]
                fig_bt_bar = go.Figure(go.Bar(
                    x=bt_labels, y=bt_returns,
                    marker_color=[style_color if v >= 0 else "#9e2b25" for v in bt_returns],
                    text=[f"{v:+.1f}%" for v in bt_returns],
                    textposition="outside",
                ))
                fig_bt_bar.update_layout(
                    title="Backtest — Hisse Bazlı Getiri",
                    paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                    font=dict(color=_theme()["ink"]),
                    yaxis=dict(gridcolor="#d8d0c0", ticksuffix="%"),
                    xaxis=dict(gridcolor="#d8d0c0"),
                    showlegend=False,
                    height=320, margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_bt_bar, use_container_width=True)

                # Equity Curve (Tüm Portföy)
                st.markdown("### 💰 Portföy Equity Curve (Sermaye Eğrisi)")
                _tm_render_equity_curve(valid_bt, bt_run_id, style_color)

                # Hisse Bazlı Detay (AL/SAT Grafikleri)
                st.markdown("### 📈 Hisse Bazlı AL/SAT Grafikleri & İşlem Geçmişi")
                for tk, tk_res in valid_bt.items():
                    summary = tk_res["summary"]
                    trades = tk_res["trades"]
                    with st.expander(
                        f"**{tk}**  —  Getiri: %{summary['total_return_pct']:+.1f}  |  "
                        f"Kazanma: %{summary['win_rate']:.0f}  |  "
                        f"{summary['total_trades']} işlem",
                        expanded=False
                    ):
                        # Alt sekmeler: Grafik / İşlem Tablosu / Teknik Göstergeler
                        stab1, stab2, stab3 = st.tabs([
                            "📈 Fiyat & AL/SAT", "📋 İşlem Geçmişi", "📊 Teknik Göstergeler"
                        ])

                        with stab1:
                            _render_price_chart_with_trades(
                                tk, bt_period, trades, height=400,
                                market=result.get("market", "BIST")
                            )

                        with stab2:
                            _tm_render_trade_table(trades, tk)

                        with stab3:
                            _tm_render_technical_chart(tk, bt_period, bt_run_id,
                                                      result.get("market", "BIST"))

            else:
                # Backtest var ama işlem üretilememiş
                no_trade_tickers = [k for k, v in bt_results.items()
                                    if v.get("summary", {}).get("total_trades", 0) == 0
                                    and "error" not in v]
                error_tickers = [k for k, v in bt_results.items() if "error" in v]
                if no_trade_tickers:
                    st.info(f"Şu hisselerde işlem üretilemedi: {', '.join(no_trade_tickers)}")
                if error_tickers:
                    st.warning(f"Şu hisselerde hata oluştu: {', '.join(error_tickers)}")

    # TAB 2: GERİYE DÖNÜK SKOR
    with tab_grade:
        st.markdown(report["grade_text"])
        st.markdown("---")

        if picks:
            pick_df = pd.DataFrame(picks)
            display_cols = ["ticker", "score", "price", "current_price", "return_pct", "rsi", "adx", "signal"]
            available_cols = [c for c in display_cols if c in pick_df.columns]
            pick_df_display = pick_df[available_cols].rename(columns={
                "ticker": "Hisse", "score": "PIT Skor", "price": "PIT Fiyat",
                "current_price": "Güncel Fiyat", "return_pct": "Getiri%",
                "rsi": "PIT RSI", "adx": "PIT ADX", "signal": "PIT Sinyal",
            })

            def _ret_color(val):
                if not isinstance(val, (int, float)): return ""
                if val > 10:   return "background-color: rgba(34,197,94,0.25)"
                elif val > 0:  return "background-color: rgba(34,197,94,0.1)"
                elif val > -10: return "background-color: rgba(239,68,68,0.1)"
                return "background-color: rgba(239,68,68,0.25)"

            st.dataframe(
                pick_df_display.style.map(
                    _ret_color, subset=["Getiri%"] if "Getiri%" in pick_df_display.columns else []
                ),
                use_container_width=True, hide_index=True,
            )

            # Bar chart — PIT getiri + benchmark
            labels = [p["ticker"] for p in picks]
            returns = [p.get("return_pct", 0) for p in picks]
            colors = ["#1d6f4e" if r >= 0 else "#9e2b25" for r in returns]

            fig_bar = go.Figure(go.Bar(
                x=labels, y=returns,
                marker_color=colors,
                text=[f"{v:+.1f}%" for v in returns],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title=f"PIT Portföy — Hisse Bazlı Getiri ({result['pit_date']} → Bugün)",
                paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                font=dict(color=_theme()["ink"]),
                yaxis=dict(gridcolor="#d8d0c0", ticksuffix="%"),
                xaxis=dict(gridcolor="#d8d0c0"),
                showlegend=False,
                height=350, margin=dict(l=10, r=10, t=50, b=10),
            )
            fig_bar.add_hline(
                y=result["bench_return"],
                line_dash="dash", line_color="#27509e",
                annotation_text=f"Endeks: %{result['bench_return']:+.1f}",
                annotation_position="top right",
                annotation=dict(font=dict(color="#27509e", size=12)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 3: TUTARLILIK ANALİZİ
    with tab_consist:
        st.markdown("### 🔍 Strateji Tutarlılık Analizi")
        st.markdown(report["consistency"])
        st.markdown("---")
        st.markdown("### 📌 Bugünkü Portföy için Ne Anlama Geliyor?")
        if grade >= 65:
            st.success(
                "Strateji geçmiş sınavını başarıyla geçti. "
                "Bugünkü seçimlere makul güven duyulabilir — "
                "ancak piyasa koşulları değişebilir."
            )
        elif grade >= 40:
            st.warning(
                "Strateji kısmen başarılı. Pozisyon boyutunu küçük tutun."
            )
        else:
            st.error(
                "Strateji geçmişte zorlanmış. Parametre optimizasyonu düşünün."
            )
        st.markdown("---")
        st.markdown(
            "**Overconfidence (Aşırı Güven) Kontrolü:**\n\n"
            "- Geçmiş başarı gelecek başarıyı garanti etmez\n"
            "- Piyasa rejimleri değişir (boğa → ayı, düşük volatilite → yüksek)\n"
            "- Alfa negatifse strateji **endeksin bile gerisinde** kalmış demek\n"
            "- Stratejinin spekülatif yükselişe mi denk geldiğini değerlendirin"
        )

    # TAB 4: RİSKLİ HİSSELER
    with tab_risk:
        st.markdown("### ⚠️ Riskli Hisseler")
        st.caption("Gelecek 3 ayda en riskli görünen hisseler")
        risky = report.get("risky_stocks", [])
        if risky:
            for r in risky:
                ret = r.get("return_pct", 0)
                color = "#9e2b25" if ret < -5 else "#b45309" if ret < 5 else "#6b6357"
                st.markdown(
                    f"<div style='background:#efe9db;border:1px solid {color};"
                    f"border-radius:8px;padding:10px 14px;margin-bottom:6px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span style='font-size:16px;font-weight:700;color:{color}'>{r['ticker']}</span>"
                    f"<span style='color:{color};font-size:14px'>%{ret:+.1f}</span>"
                    f"</div>"
                    f"<div style='font-size:12px;color:#6b6357;margin-top:4px'>"
                    f"Risk: {r['reason']}"
                    f"</div></div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Risk verileri hesaplanamadı.")

    # TAB 5: GÜNLÜK GÖZLEM
    with tab_daily:
        st.markdown("### 📅 Günlük Gözlem Notu")
        st.markdown(report["daily_note"])
        st.markdown("---")
        daily_df = TimeMachineEngine.load_daily_data(result.get("run_id", ""))
        if not daily_df.empty:
            st.markdown("**Son Kaydedilen Günlük Veriler:**")
            st.dataframe(
                daily_df[["snapshot_date", "ticker", "price", "daily_change_pct", "cumulative_return_pct"]].rename(
                    columns={
                        "snapshot_date": "Tarih", "ticker": "Hisse",
                        "price": "Fiyat", "daily_change_pct": "Günlük%",
                        "cumulative_return_pct": "Kümülatif%",
                    }
                ),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Henüz günlük veri yok. Her çalıştırmada otomatik kaydedilir.")


def _tm_render_equity_curve(bt_results: dict, run_id: str, color: str):
    """Tüm portföy için birleşik equity (sermaye) eğrisi çizer."""
    try:
        all_daily = []
        for ticker, res in bt_results.items():
            daily_rows = res.get("daily", [])
            if daily_rows:
                for d in daily_rows:
                    all_daily.append({
                        "date": d.get("date") or d[0] if isinstance(d, (list, tuple)) else "",
                        "ticker": ticker,
                        "price": d.get("price") or (d[2] if isinstance(d, (list, tuple)) and len(d) > 2 else 0),
                    })

        if not all_daily:
            # Fallback: DB'den yükle
            try:
                daily_df = BacktestEngine.load_daily_scores(run_id, None)
                if not daily_df.empty and "price" in daily_df.columns:
                    daily_df["date"] = daily_df["date"].astype(str)
                    all_daily = daily_df[["date","ticker","price"]].to_dict("records")
            except Exception:
                pass

        if not all_daily:
            st.caption("Equity curve verisi bulunamadı.")
            return

        df = pd.DataFrame(all_daily)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)

        # Her hisse için normalize (100 bazlı)
        fig = go.Figure()
        for ticker in df["ticker"].unique():
            tdf = df[df["ticker"] == ticker].sort_values("date")
            if tdf.empty or tdf["price"].iloc[0] == 0:
                continue
            base = tdf["price"].iloc[0]
            tdf = tdf.copy()
            tdf["norm"] = (tdf["price"] / base) * 100

            fig.add_trace(go.Scatter(
                x=tdf["date"], y=tdf["norm"],
                mode="lines", name=ticker,
                line=dict(width=1.5),
                hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>Endeksli: %{{y:.1f}}<extra></extra>",
            ))

        # Portföy ortalaması
        if len(df["ticker"].unique()) > 1:
            pivot = df.pivot_table(index="date", columns="ticker", values="price")
            for col in pivot.columns:
                base = pivot[col].dropna().iloc[0] if not pivot[col].dropna().empty else 1
                if base > 0:
                    pivot[col] = pivot[col] / base * 100
            avg = pivot.mean(axis=1).dropna()
            fig.add_trace(go.Scatter(
                x=avg.index, y=avg.values,
                mode="lines", name="PORTFÖY ORT.",
                line=dict(color=color, width=3, dash="solid"),
                hovertemplate="<b>Portföy Ort.</b><br>%{x|%Y-%m-%d}<br>Endeksli: %{y:.1f}<extra></extra>",
            ))

        # 100 referans çizgisi
        fig.add_hline(y=100, line_dash="dot", line_color="#8a8172", opacity=0.5)

        fig.update_layout(
            title="Portföy Equity Curve (100 Bazlı)",
            paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
            font=dict(color=_theme()["ink"]),
            yaxis=dict(gridcolor="#d8d0c0", title="Endeksli Değer"),
            xaxis=dict(gridcolor="#d8d0c0"),
            height=400, margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.caption(f"Equity curve oluşturulamadı: {exc}")


def _tm_render_trade_table(trades: list, ticker: str):
    """İşlem geçmişi tablosu gösterir."""
    if not trades:
        st.info(f"{ticker} için işlem bulunamadı.")
        return

    EXIT_LABELS = {
        "TAKE_PROFIT":    "🟢 Hedef",
        "STOP_LOSS":      "🔴 Stop",
        "SAT_SINYAL":     "🟠 SAT Sinyali",
        "MAX_SURE":       "⏰ Max Süre",
        "HALA_ACIK":      "🔵 Açık",
        "SMA200_KIRILIM": "📉 SMA200 Kırılım",
    }

    trade_data = []
    for t in trades:
        trade_data.append({
            "Giriş": t.get("entry_date", ""),
            "Giriş Fiyat": f"{t.get('entry_price', 0):.2f}",
            "Çıkış": t.get("exit_date", "—"),
            "Çıkış Fiyat": f"{t.get('exit_price', 0):.2f}" if t.get("exit_price") else "—",
            "Sebep": EXIT_LABELS.get(t.get("exit_reason", ""), t.get("exit_reason", "")),
            "Getiri%": f"{t.get('return_pct', 0):+.2f}%",
            "Süre": f"{t.get('hold_days', 0)} gün",
            "Giriş Skor": f"{t.get('entry_score', 0):.0f}",
        })

    trade_df = pd.DataFrame(trade_data)
    st.dataframe(trade_df, use_container_width=True, hide_index=True)

    # İşlem özet metrikleri
    rets = [t.get("return_pct", 0) for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam İşlem", len(trades))
    c2.metric("Kazanma", f"%{len(wins)/max(1,len(rets))*100:.0f}")
    c3.metric("Ort. Kazanç", f"%{sum(wins)/max(1,len(wins)):+.1f}" if wins else "—")
    c4.metric("Ort. Kayıp", f"%{sum(losses)/max(1,len(losses)):+.1f}" if losses else "—")


def _tm_render_technical_chart(ticker: str, period: str, run_id: str,
                               market: str = "BIST"):
    """Teknik göstergeler — RSI + MACD + Skor overlay grafiği."""
    try:
        yt = _yf_symbol(ticker, market)
        raw = yf.Ticker(yt).history(period=period, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty or "Close" not in raw.columns:
            st.warning(f"{ticker} teknik veri alınamadı.")
            return
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        col_rn = {c: c.strip().title() for c in raw.columns
                  if c.strip().title() in ("Open", "High", "Low", "Close", "Volume")}
        if col_rn:
            raw = raw.rename(columns=col_rn)

        close = raw["Close"]

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # SMA
        sma50  = close.rolling(50, min_periods=20).mean()
        sma200 = close.rolling(200, min_periods=50).mean()

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f"{ticker} Fiyat + SMA", "RSI (14)", "MACD"),
        )

        # Fiyat + SMA
        fig.add_trace(go.Scatter(x=raw.index, y=close, name="Fiyat",
                                 line=dict(color="#1a1712", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=raw.index, y=sma50, name="SMA50",
                                 line=dict(color="#a2701d", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=raw.index, y=sma200, name="SMA200",
                                 line=dict(color="#27509e", width=1, dash="dot")), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=raw.index, y=rsi, name="RSI",
                                 line=dict(color="#6d3f8e", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#9e2b25", opacity=0.4, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#1d6f4e", opacity=0.4, row=2, col=1)

        # MACD
        colors = ["#1d6f4e" if v >= 0 else "#9e2b25" for v in macd_hist.fillna(0)]
        fig.add_trace(go.Bar(x=raw.index, y=macd_hist, name="MACD Hist",
                             marker_color=colors, opacity=0.7), row=3, col=1)
        fig.add_trace(go.Scatter(x=raw.index, y=macd_line, name="MACD",
                                 line=dict(color="#27509e", width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=raw.index, y=signal_line, name="Signal",
                                 line=dict(color="#b45309", width=1, dash="dot")), row=3, col=1)

        fig.update_layout(
            paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
            font=dict(color="#1a1712", size=11),
            height=600, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified",
        )
        for i in range(1, 4):
            fig.update_yaxes(gridcolor="#d8d0c0", row=i, col=1)
            fig.update_xaxes(gridcolor="#d8d0c0", row=i, col=1)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.warning(f"{ticker} teknik grafik oluşturulamadı: {exc}")


# US MARKETS PAGE

def render_us_markets_page(ui_lang: str, mode_override: str = None):
    """ABD Piyasaları — Analiz & Backtest sayfasi."""
    st.markdown("# US Markets — NASDAQ / NYSE")
    st.caption(
        "Analyze US stocks with technical scoring, news sentiment, and backtesting. "
        "All indicators and thresholds are the same as BIST, adapted for US market indices."
    )

    us_mode = mode_override or "analysis"

    # Sidebar
    with st.sidebar:
        st.markdown("## US Market Settings")

        if us_mode == "analysis":
            us_ticker = st.selectbox(
                "Select Stock",
                options=[""] + sorted(US_POPULAR_TICKERS),
                format_func=lambda x: x if x else "Type or select...",
                key="us_ticker_select",
            )
            us_custom = st.text_input(
                "Or enter ticker manually",
                placeholder="e.g. AAPL, NVDA, TSLA",
                key="us_custom_ticker",
            ).strip().upper()
            us_symbol = us_custom if us_custom else us_ticker

            us_period = st.selectbox(
                "Period", options=["6mo", "1y", "2y"], index=1,
                format_func=lambda x: {"6mo": "6 Months", "1y": "1 Year", "2y": "2 Years"}[x],
                key="us_analysis_period",
            )
            us_analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

        else:  # backtest
            us_bt_tickers = st.multiselect(
                "Stocks to Backtest",
                options=sorted(US_POPULAR_TICKERS),
                default=["AAPL", "NVDA", "MSFT", "TSLA", "AMZN"],
                placeholder="Search US stocks...",
                key="us_bt_tickers_select",
            )
            us_bt_period = st.selectbox(
                "Period", options=["1y", "2y", "3y"], index=1,
                format_func=lambda x: {"1y": "1 Year", "2y": "2 Years", "3y": "3 Years"}[x],
                key="us_bt_period",
            )
            us_bt_mode = st.radio(
                "Strategy",
                options=["swing", "trend", "universal", "investor", "buyhold"],
                index=2,
                format_func=lambda x: BacktestEngine.MODES[x]["label"],
                key="us_bt_mode",
            )
            us_bt_short = st.checkbox("Short Selling", value=False, key="us_short")
            us_bt_scaling = st.checkbox("Scale Out", value=True, key="us_scaling")
            us_bt_btn = st.button("Run Backtest", type="primary",
                                  use_container_width=True, key="us_bt_run")

    # ANALYSIS MODE
    if us_mode == "analysis":
        if not (us_analyze_btn and us_symbol):
            st.info("Select a US stock from the sidebar and click **Analyze**.")
            return

        with st.spinner(f"Analyzing {us_symbol}..."):
            try:
                raw = yf.Ticker(us_symbol).history(period=us_period, auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                if raw.empty or "Close" not in raw.columns:
                    st.error(
                        f"**{us_symbol}** için veri bulunamadı. Hisse kodunu kontrol edin "
                        f"(örn: AAPL, MSFT, GOOGL)."
                        if ui_lang == "TR" else
                        f"No data found for **{us_symbol}**. Please check the ticker symbol "
                        f"(e.g., AAPL, MSFT, GOOGL)."
                    )
                    return
                if "Open"   not in raw.columns: raw["Open"]   = raw["Close"]
                if "High"   not in raw.columns: raw["High"]   = raw["Close"]
                if "Low"    not in raw.columns: raw["Low"]    = raw["Close"]
                if "Volume" not in raw.columns: raw["Volume"] = 0.0
            except Exception as exc:
                st.error(
                    f"**{us_symbol}** verisi alınırken hata oluştu. "
                    f"İnternet bağlantınızı kontrol edin veya farklı bir hisse deneyin."
                    if ui_lang == "TR" else
                    f"Error fetching **{us_symbol}**. Check your internet connection or try a different ticker."
                )
                log.warning("US veri çekme hatası (%s): %s", us_symbol, exc)
                return

        # Score
        scores, atr_s, rsi_s = BacktestEngine._vectorized_scores(raw)
        last_score = float(scores.iloc[-1])
        last_rsi   = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50.0
        last_price = float(raw["Close"].iloc[-1])
        last_atr   = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0.0

        # Info from yfinance
        try:
            info = yf.Ticker(us_symbol).info or {}
        except Exception:
            info = {}
        company_name = info.get("longName", us_symbol)
        sector       = info.get("sector", US_SECTOR_MAP.get(us_symbol, "N/A"))
        market_cap   = info.get("marketCap", 0)
        pe_ratio     = info.get("trailingPE", 0)

        # Header metrics
        st.markdown(f"### {company_name} ({us_symbol})")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Price", f"${last_price:.2f}")
        m2.metric("Score", f"{last_score:.0f}/100")
        m3.metric("RSI", f"{last_rsi:.1f}")
        m4.metric("ATR", f"${last_atr:.2f}")
        m5.metric("P/E", f"{pe_ratio:.1f}x" if pe_ratio else "N/A")
        m6.metric("Sector", sector)

        # Signal
        if last_score >= 60:
            signal, sig_color = "STRONG BUY", "#1d6f4e"
        elif last_score >= 48:
            signal, sig_color = "BUY", "#3e8e6c"
        elif last_score >= 35:
            signal, sig_color = "NEUTRAL", "#6b6357"
        elif last_score >= 22:
            signal, sig_color = "SELL", "#b45309"
        else:
            signal, sig_color = "STRONG SELL", "#9e2b25"

        # Sinyali doğrulama sistemine kaydet
        _signal_map = {"STRONG BUY": "GUCLU AL", "BUY": "AL",
                       "SELL": "SAT", "STRONG SELL": "GUCLU SAT"}
        _mapped = _signal_map.get(signal)
        if _mapped:
            try:
                _history_db.record_signal(us_symbol, _mapped, last_score,
                                          last_price, source="live_us")
            except Exception:
                pass

        # Gauge Chart + Signal (BIST ile ayni)
        col_gauge, col_signal = st.columns([1, 1.5])
        with col_gauge:
            st.markdown(
                f"<div class='signal-box' style='color:{sig_color}'>"
                f"{signal}</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(create_gauge_chart(last_score, title="US Buy Score"), use_container_width=True)

        with col_signal:
            st.markdown("#### Technical Breakdown")
            # SMA status
            sma50_val  = raw["Close"].rolling(50, min_periods=20).mean().iloc[-1]
            sma200_val = raw["Close"].rolling(200, min_periods=50).mean().iloc[-1]
            above_sma50  = last_price > sma50_val if not np.isnan(sma50_val) else False
            above_sma200 = last_price > sma200_val if not np.isnan(sma200_val) else False
            st.markdown(f"**SMA50**: {'Above' if above_sma50 else 'Below'} ({'${:.2f}'.format(sma50_val) if not np.isnan(sma50_val) else 'N/A'})")
            st.markdown(f"**SMA200**: {'Above' if above_sma200 else 'Below'} ({'${:.2f}'.format(sma200_val) if not np.isnan(sma200_val) else 'N/A'})")
            st.markdown(f"**RSI**: {last_rsi:.1f} {'(Oversold)' if last_rsi < 30 else '(Overbought)' if last_rsi > 70 else ''}")
            st.markdown(f"**ATR**: ${last_atr:.2f} ({last_atr/last_price*100:.1f}%)")
            if market_cap:
                cap_str = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.0f}M"
                st.markdown(f"**Market Cap**: {cap_str}")

        # Add to Portfolio button
        st.markdown("---")
        pf_col1, pf_col2, pf_col3 = st.columns([2, 1.5, 1])
        with pf_col1:
            pf_qty = st.number_input("Quantity", min_value=0, step=1, value=0, key="us_pf_qty")
        with pf_col2:
            pf_cost = st.number_input("Cost ($)", min_value=0.0, value=round(last_price, 2), format="%.2f", key="us_pf_cost")
        with pf_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add to Portfolio", key="us_add_pf",
                         use_container_width=True) and _guard_write():
                _history_db.add_portfolio(f"{us_symbol}:US", pf_cost, pf_qty)
                st.success(f"{us_symbol} added to portfolio!")

        # Combined chart: Candlestick + Volume + RSI
        st.markdown("---")
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{us_symbol} Price", "Volume", "RSI"),
        )

        fig.add_trace(go.Candlestick(
            x=raw.index, open=raw["Open"], high=raw["High"],
            low=raw["Low"], close=raw["Close"],
            increasing_line_color="#1d6f4e", decreasing_line_color="#9e2b25",
            name="Price",
        ), row=1, col=1)

        sma50  = raw["Close"].rolling(50, min_periods=20).mean()
        sma200 = raw["Close"].rolling(200, min_periods=50).mean()
        fig.add_trace(go.Scatter(x=raw.index, y=sma50, name="SMA50",
                                 line=dict(color="#a2701d", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=raw.index, y=sma200, name="SMA200",
                                 line=dict(color="#6d3f8e", width=1)), row=1, col=1)

        vol_colors = ["#1d6f4e" if raw["Close"].iloc[j] >= raw["Open"].iloc[j]
                       else "#9e2b25" for j in range(len(raw))]
        fig.add_trace(go.Bar(x=raw.index, y=raw["Volume"], name="Volume",
                             marker_color=vol_colors, opacity=0.7), row=2, col=1)

        fig.add_trace(go.Scatter(x=raw.index, y=rsi_s, name="RSI",
                                 line=dict(color="#27509e", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#9e2b25", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#1d6f4e", row=3, col=1)

        fig.update_layout(
            paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
            font=dict(color=_theme()["ink"]),
            height=650, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        for ax in ["yaxis", "yaxis2", "yaxis3"]:
            fig.update_layout(**{ax: dict(gridcolor="#d8d0c0")})
        for ax in ["xaxis", "xaxis2", "xaxis3"]:
            fig.update_layout(**{ax: dict(gridcolor="#d8d0c0")})

        st.plotly_chart(fig, use_container_width=True)

        # News (English)
        if NEWS_ENGINE_AVAILABLE:
            with st.expander("News Sentiment (EN)", expanded=False):
                try:
                    from news_engine import analyze_news, render_news_panel
                    news_result = analyze_news(us_symbol, days=14, language="EN")
                    render_news_panel(news_result)
                except Exception as exc:
                    st.warning(f"News error: {exc}")

        # Score history chart
        with st.expander("Score History", expanded=False):
            fig_score = go.Figure()
            fig_score.add_trace(go.Scatter(
                x=raw.index, y=scores,
                mode="lines", name="Technical Score",
                line=dict(color="#27509e", width=1.5),
            ))
            fig_score.add_hline(y=48, line_dash="dash", line_color="#1d6f4e",
                                annotation_text="BUY")
            fig_score.add_hline(y=22, line_dash="dash", line_color="#9e2b25",
                                annotation_text="SELL")
            fig_score.update_layout(
                paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                font=dict(color=_theme()["ink"]),
                yaxis=dict(gridcolor="#d8d0c0", title="Score", range=[-5, 105]),
                xaxis=dict(gridcolor="#d8d0c0"),
                height=300, margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_score, use_container_width=True)

    # BACKTEST MODE
    elif us_mode == "backtest":
        # Backtest calistir (buton basilmissa)
        if "us_bt_run_id" not in st.session_state:
            st.session_state["us_bt_run_id"] = None

        if us_bt_btn and us_bt_tickers:
            run_id   = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_us_{us_bt_mode}"
            run_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            progress = st.progress(0, text="Starting US backtest...")
            status   = st.empty()
            total    = len(us_bt_tickers)

            for idx, ticker in enumerate(us_bt_tickers):
                status.info(f"Processing **{ticker}** ({idx+1}/{total})...")
                try:
                    trades, summary, daily = BacktestEngine._run_single(
                        ticker, us_bt_period, run_id,
                        mode=us_bt_mode, use_news=False,
                        enable_short=us_bt_short, enable_scaling=us_bt_scaling,
                        market="US",
                    )
                    BacktestEngine._save(DB_PATH, run_id, run_date,
                                         ticker, us_bt_period, trades, summary, daily)
                    _dir = ""
                    if summary.get("short_trades", 0) > 0:
                        _dir = f" (L:{summary['long_trades']} S:{summary['short_trades']})"
                    status.success(
                        f"**{ticker}** — {summary['total_trades']} trades{_dir} | "
                        f"Win: {summary['win_rate']:.0f}% | "
                        f"Return: {summary['total_return_pct']:+.1f}% | "
                        f"Capital: ${summary['final_capital']:,.0f}"
                    )
                except Exception as exc:
                    status.warning(f"**{ticker}** — Error: {exc}")
                progress.progress((idx + 1) / total)

            progress.progress(1.0, text="Backtest complete!")
            st.session_state["us_bt_run_id"] = run_id
            st.rerun()

        # Backtest sonuclari goster
        all_runs = BacktestEngine.load_runs()
        us_runs  = [r for r in all_runs if "_us_" in r.get("run_id", "")]
        if not us_runs:
            st.info("Select US stocks from the sidebar and click **Run Backtest**.")
            return

        run_ids = list(dict.fromkeys(r["run_id"] for r in us_runs))
        default_run = st.session_state.get("us_bt_run_id", run_ids[0])
        if default_run not in run_ids:
            default_run = run_ids[0]

        col_sel, _ = st.columns([2, 3])
        with col_sel:
            selected_run = st.selectbox(
                "Select Backtest Run",
                options=run_ids,
                index=run_ids.index(default_run),
                format_func=lambda r: next(
                    (
                        x["run_date"] + f"  ({x['period']})"
                        + ("  [" + r.rsplit("_", 1)[-1].upper() + "]" if "_" in r else "")
                        for x in us_runs if x["run_id"] == r
                    ), r
                ),
                label_visibility="collapsed",
                key="us_bt_run_select",
            )

        run_rows = [r for r in us_runs if r["run_id"] == selected_run]
        if not run_rows:
            return

        # TAB: Summary + Detail
        tab_summary, tab_detail = st.tabs(["Summary", "Stock Detail"])

        with tab_summary:
            valid = [r for r in run_rows if r["total_trades"] > 0]
            if valid:
                total_trades = sum(r["total_trades"] for r in valid)
                total_wins   = sum(r["winning_trades"] for r in valid)
                avg_win_rate = total_wins / total_trades * 100
                avg_return   = sum(r["avg_return_pct"] * r["total_trades"] for r in valid) / total_trades
                best_h  = max(valid, key=lambda r: r["total_return_pct"])
                worst_h = min(valid, key=lambda r: r["total_return_pct"])

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Trades",    total_trades)
                m2.metric("Win Rate",        f"{avg_win_rate:.1f}%")
                m3.metric("Avg Return/Trade", f"{avg_return:+.2f}%")
                m4.metric("Best",  best_h["ticker"],  f"{best_h['total_return_pct']:+.1f}%")
                m5.metric("Worst", worst_h["ticker"], f"{worst_h['total_return_pct']:+.1f}%")

            st.markdown("---")
            summary_rows = []
            for row in run_rows:
                emoji = ("+" if row["win_rate"] >= 55 and row["total_return_pct"] > 0
                         else "-" if row["win_rate"] < 45 or row["total_return_pct"] < -5
                         else "~")
                summary_rows.append({
                    "":           emoji,
                    "Stock":      row["ticker"],
                    "Trades":     row["total_trades"],
                    "Win %":      row["win_rate"],
                    "Avg %":      row["avg_return_pct"],
                    "Total %":    row["total_return_pct"],
                    "Max DD %":   -row["max_drawdown_pct"],
                    "Best %":     row["best_trade_pct"],
                    "Worst %":    row["worst_trade_pct"],
                    "Avg Days":   row["avg_hold_days"],
                })

            if summary_rows:
                df_sum = pd.DataFrame(summary_rows)
                st.dataframe(df_sum, use_container_width=True, hide_index=True)

            # Bar chart
            if valid:
                st.markdown("---")
                sorted_valid = sorted(valid, key=lambda r: r["total_return_pct"], reverse=True)
                fig_bar = go.Figure(go.Bar(
                    x=[r["ticker"] for r in sorted_valid],
                    y=[r["total_return_pct"] for r in sorted_valid],
                    marker_color=["#1d6f4e" if r["total_return_pct"] >= 0 else "#9e2b25"
                                  for r in sorted_valid],
                    text=[f"{r['total_return_pct']:+.1f}%" for r in sorted_valid],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    title="Cumulative Return by Stock",
                    paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                    font=dict(color=_theme()["ink"]),
                    yaxis=dict(gridcolor="#d8d0c0", ticksuffix="%"),
                    xaxis=dict(gridcolor="#d8d0c0"),
                    showlegend=False,
                    height=360, margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab_detail:
            ticker_opts = [r["ticker"] for r in run_rows]
            if not ticker_opts:
                st.info("No stocks in this backtest run.")
            else:
                sel_ticker = st.selectbox("Select Stock", ticker_opts, key="us_bt_detail_ticker")
                trades_raw = BacktestEngine.load_trades(selected_run, sel_ticker)
                run_period = next((r["period"] for r in run_rows if r["ticker"] == sel_ticker), "2y")

                if not trades_raw:
                    st.info(f"No trades for {sel_ticker} in this backtest.")
                else:
                    sel_row = next((r for r in run_rows if r["ticker"] == sel_ticker), {})
                    if sel_row:
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Trades", sel_row.get("total_trades", 0))
                        s2.metric("Win Rate", f"{sel_row.get('win_rate', 0):.0f}%")
                        s3.metric("Total Return", f"{sel_row.get('total_return_pct', 0):+.2f}%")
                        s4.metric("Max Drawdown", f"{sel_row.get('max_drawdown_pct', 0):.1f}%")

                    st.markdown("---")

                    # Price chart with BUY/SELL signals
                    _render_price_chart_with_trades(sel_ticker, run_period, trades_raw, height=420, market="US")

                    # Equity curve
                    returns_list = [t["return_pct"] for t in trades_raw]
                    equity = [100.0]
                    labels = ["Start"]
                    for r2, t2 in zip(returns_list, trades_raw):
                        equity.append(equity[-1] * (1 + r2 / 100))
                        labels.append(t2.get("exit_date", ""))

                    fig_eq = go.Figure()
                    eq_color = "#1d6f4e" if equity[-1] >= 100 else "#9e2b25"
                    eq_fill  = "rgba(34,197,94,0.08)" if equity[-1] >= 100 else "rgba(239,68,68,0.08)"
                    fig_eq.add_trace(go.Scatter(
                        x=labels, y=equity,
                        mode="lines+markers",
                        line=dict(color=eq_color, width=2),
                        fill="tozeroy", fillcolor=eq_fill,
                        name="Capital",
                        hovertemplate="%{x}<br>Capital: %{y:.1f}<extra></extra>",
                    ))
                    fig_eq.add_hline(y=100, line_dash="dash", line_color="#8a8172",
                                     annotation_text="Start (100)")
                    fig_eq.update_layout(
                        title=f"{sel_ticker} — Cumulative Equity Curve",
                        paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                        font=dict(color=_theme()["ink"]),
                        yaxis=dict(gridcolor="#d8d0c0", title="Capital (100=start)"),
                        xaxis=dict(gridcolor="#d8d0c0"),
                        height=340, margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

                    # Trade list table
                    with st.expander("Trade History", expanded=False):
                        trade_rows = []
                        for t in trades_raw:
                            trade_rows.append({
                                "Direction":  t.get("direction", "LONG"),
                                "Entry Date": t.get("entry_date", ""),
                                "Entry $":    t.get("entry_price", 0),
                                "Exit Date":  t.get("exit_date", ""),
                                "Exit $":     t.get("exit_price", 0),
                                "Return %":   t.get("return_pct", 0),
                                "Days":       t.get("hold_days", 0),
                                "Reason":     t.get("exit_reason", ""),
                            })
                        if trade_rows:
                            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)


# US STOCK LIST PAGE

def render_us_stock_list_page(ui_lang: str):
    """US hisse listesi — kategorilere ayrilmis, fiyat degisimli."""
    st.markdown("# US Stock List")
    st.caption("Click a stock to navigate to US Analysis page.")
    st.markdown("---")

    search = st.text_input(
        "Search Stock", placeholder="e.g. AAPL, Tesla...",
        key="us_list_search"
    ).strip().upper()

    # Sector categories
    categories = {}
    for ticker in US_POPULAR_TICKERS:
        sector = US_SECTOR_MAP.get(ticker, "Other")
        categories.setdefault(sector, []).append(ticker)

    @st.cache_data(ttl=900, show_spinner=False)
    def _fetch_us_daily(tickers_tuple):
        result = {}
        if not tickers_tuple:
            return result
        try:
            symbols = list(tickers_tuple)
            raw = yf.download(symbols, period="5d", auto_adjust=True, progress=False)
            if raw.empty:
                return result
            is_multi = len(symbols) > 1
            for t in tickers_tuple:
                try:
                    if is_multi:
                        if isinstance(raw.columns, pd.MultiIndex):
                            lvl0 = raw.columns.get_level_values(0)
                            if "Close" in lvl0:
                                col = raw["Close"][t].dropna()
                            elif t in lvl0:
                                col = raw[t]["Close"].dropna()
                            else:
                                continue
                        else:
                            col = raw["Close"].dropna() if "Close" in raw.columns else pd.Series(dtype=float)
                    else:
                        col = raw["Close"].dropna() if "Close" in raw.columns else pd.Series(dtype=float)
                    if len(col) >= 2:
                        cur  = float(col.iloc[-1])
                        prev = float(col.iloc[-2])
                        pct  = (cur - prev) / prev * 100
                        result[t] = {"price": cur, "pct": round(pct, 2), "up": pct >= 0}
                except Exception:
                    pass
        except Exception:
            pass
        return result

    with st.spinner("Fetching US market data..."):
        daily_data = _fetch_us_daily(tuple(US_POPULAR_TICKERS))

    for sector, tickers in sorted(categories.items()):
        filtered = [t for t in tickers if not search or search in t]
        if not filtered:
            continue

        with st.expander(f"{sector} ({len(filtered)})", expanded=len(categories) <= 6):
            cols = st.columns(4)
            for i, ticker in enumerate(filtered):
                with cols[i % 4]:
                    data = daily_data.get(ticker, {})
                    price = data.get("price", 0)
                    pct   = data.get("pct", 0)
                    color = "#1d6f4e" if pct >= 0 else "#9e2b25"
                    arrow = "+" if pct >= 0 else ""

                    if st.button(
                        f"{ticker}",
                        key=f"us_list_{ticker}",
                        use_container_width=True,
                    ):
                        st.session_state["us_ticker_select"] = ticker
                        st.session_state["nav_radio"] = "US Analiz"
                        st.session_state["nav_page"]  = "US Analiz"
                        st.rerun()

                    st.markdown(
                        f"<div style='font-size:11px;color:{color};margin-top:-8px;text-align:center'>"
                        f"${price:.2f} ({arrow}{pct:.2f}%)</div>",
                        unsafe_allow_html=True,
                    )


# US SYSTEM PORTFOLIOS (Scanner)

def render_us_system_portfolios_page(ui_lang: str):
    """US hisselerini tarar, Agresif/Defansif portfoyler olusturur + inline backtest."""
    st.markdown("# US System Portfolios")
    st.caption(
        "Automatically scans US stocks using technical indicators to build "
        "Aggressive, Defensive, Momentum, Value and Stable portfolios."
    )

    # Sidebar Kontroller (BIST ile simetrik)
    with st.sidebar:
        st.markdown("## US Portfolio Settings")
        scan_btn = st.button("🔄 Scan US Stocks", type="primary",
                             use_container_width=True, key="us_scan_btn")
        st.markdown("---")
        st.markdown("### Backtest Settings")
        us_sp_period = st.selectbox(
            "Backtest Period",
            options=["1y", "2y", "3y"],
            index=0,
            format_func=lambda x: {"1y": "1 Year", "2y": "2 Years", "3y": "3 Years"}[x],
            key="us_sp_bt_period"
        )
        us_sp_mode = st.radio(
            "Strategy Mode",
            options=["swing", "trend", "universal", "investor", "buyhold"],
            index=2,
            format_func=lambda x: BacktestEngine.MODES[x]["label"],
            key="us_sp_bt_mode"
        )
        us_sp_news = st.checkbox("📰 News Filter", value=True, key="us_sp_news")
        st.markdown("---")
        us_sp_styles = st.multiselect(
            "Portfolio Styles",
            options=["aggressive", "defensive", "momentum", "value", "stable"],
            default=["aggressive", "defensive"],
            format_func=lambda x: {
                "aggressive": "🚀 Aggressive",
                "defensive": "🛡️ Defensive",
                "momentum": "⚡ Momentum",
                "value": "💎 Value",
                "stable": "🏦 Stable",
            }.get(x, x),
            key="us_sp_styles"
        )
        run_bt_btn = st.button("▶ Run Backtest on Portfolios",
                               use_container_width=True, key="us_sp_bt_btn")
        st.markdown("---")
        st.caption(
            "**Portfolio Criteria:**\n\n"
            "**Aggressive:** Score≥43, ADX>20, SMA200 above\n"
            "**Defensive:** Score≥38, ATR≤4.5%, SMA200 above\n"
            "**Momentum:** 1M%>3, 3M%>5, RSI 45-75\n"
            "**Value:** RSI<45, SMA200 above, dip recovery\n"
            "**Stable:** ATR<3.5%, SMA200 above, low volatility"
        )

    if not scan_btn and "us_scan_results" not in st.session_state:
        st.info("Click **Scan US Stocks** in sidebar to run the scanner.")
        return

    if scan_btn:
        progress = st.progress(0, text="Scanning US stocks...")
        results = []
        total = len(US_POPULAR_TICKERS)
        symbols = list(US_POPULAR_TICKERS)

        try:
            bulk = yf.download(symbols, period="1y", auto_adjust=True, progress=False, group_by="ticker")
        except Exception:
            bulk = None

        for idx, ticker in enumerate(US_POPULAR_TICKERS):
            progress.progress((idx + 1) / total, text=f"Scanning {ticker}...")
            result = StockScanResult(ticker=ticker)
            try:
                # Extract from bulk download
                df = None
                if bulk is not None:
                    try:
                        if isinstance(bulk.columns, pd.MultiIndex):
                            lvl0 = bulk.columns.get_level_values(0)
                            if ticker in lvl0:
                                df = bulk[ticker].copy()
                            elif "Close" in lvl0 and ticker in bulk["Close"].columns:
                                price_cols = [c for c in ("Open","High","Low","Close","Volume") if c in lvl0]
                                df = pd.DataFrame({pc: bulk[pc][ticker] for pc in price_cols if ticker in bulk[pc].columns})
                        elif len(symbols) == 1:
                            df = bulk.copy()
                    except Exception:
                        df = None

                if df is None or df.empty or len(df) < 60:
                    df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                if df.empty or len(df) < 60:
                    result.error = "Insufficient data"
                    results.append(result)
                    continue

                df.index = pd.to_datetime(df.index).tz_localize(None)
                col_rn = {c: c.strip().title() for c in df.columns
                          if c.strip().title() in ("Open","High","Low","Close","Volume")}
                if col_rn:
                    df = df.rename(columns=col_rn)
                if "High" not in df.columns: df["High"] = df["Close"]
                if "Low" not in df.columns: df["Low"] = df["Close"]
                if "Volume" not in df.columns: df["Volume"] = 0.0

                tech = TechnicalEngine.compute(df)
                result.score = tech.score
                result.rsi = tech.rsi
                result.adx = getattr(tech, "adx", 0)
                result.atr_pct = (tech.atr / float(df["Close"].iloc[-1]) * 100) if tech.atr > 0 else 0
                result.current_price = float(df["Close"].iloc[-1])
                result.data_rows = len(df)
                result.price_above_sma200 = tech.price_vs_sma200 == "above" if hasattr(tech, "price_vs_sma200") else (float(df["Close"].iloc[-1]) > df["Close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else False)
                result.price_above_sma50 = float(df["Close"].iloc[-1]) > df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else False

                # Momentum
                if len(df) >= 21:
                    result.momentum_1m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-21]) - 1) * 100
                if len(df) >= 63:
                    result.momentum_3m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-63]) - 1) * 100

                result.adx_strong = result.adx > 22
                results.append(result)
            except Exception as exc:
                result.error = str(exc)
                results.append(result)

        progress.progress(1.0, text="Scan complete!")
        st.session_state["us_scan_results"] = results

    # Display results
    scan_results = st.session_state.get("us_scan_results", [])
    if not scan_results:
        return

    valid = [r for r in scan_results if not r.error and r.data_rows >= 60 and r.current_price > 0]

    # Portföy stilleri (TimeMachineEngine filtreleri ile uyumlu)
    STYLE_INFO = {
        "aggressive": ("🚀 Aggressive", "#1d6f4e"),
        "defensive":  ("🛡️ Defensive", "#27509e"),
        "momentum":   ("⚡ Momentum", "#6d3f8e"),
        "value":      ("💎 Value", "#0e7a6e"),
        "stable":     ("🏦 Stable", "#a2701d"),
    }

    def _filter_us_portfolio(stocks, style):
        """Scan sonuçlarını portföy stiline göre filtreler."""
        selected = []
        for r in sorted(stocks, key=lambda x: x.score, reverse=True):
            if style == "aggressive":
                if r.score >= 43 and r.adx > 20 and r.price_above_sma200:
                    selected.append(r)
            elif style == "defensive":
                if r.score >= 38 and r.atr_pct <= 4.5 and r.price_above_sma200:
                    selected.append(r)
            elif style == "momentum":
                if r.momentum_1m > 3 and r.momentum_3m > 5 and 45 <= r.rsi <= 75 and r.score >= 35:
                    selected.append(r)
            elif style == "value":
                if r.rsi < 45 and r.price_above_sma200 and r.score >= 30:
                    selected.append(r)
            elif style == "stable":
                if r.atr_pct < 3.5 and r.price_above_sma200 and r.rsi < 70 and r.score >= 35:
                    selected.append(r)
            if len(selected) >= 7:
                break
        return selected

    portfolios = {}
    used_tickers = set()
    for style in us_sp_styles:
        remaining = [r for r in valid if r.ticker not in used_tickers]
        picks = _filter_us_portfolio(remaining, style)
        portfolios[style] = picks
        used_tickers.update(r.ticker for r in picks)

    # Display
    def _render_portfolio_card(title, stocks, color):
        st.markdown(f"### {title}")
        if not stocks:
            st.info("No stocks met the criteria.")
            return
        for r in stocks:
            sector = US_SECTOR_MAP.get(r.ticker, "Other")
            mom_color = "#1d6f4e" if r.momentum_1m > 0 else "#9e2b25"
            st.markdown(
                f"<div style='background:#efe9db;border:1px solid {color};border-radius:8px;"
                f"padding:10px;margin:4px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-weight:700;color:{color};font-size:16px'>{r.ticker}</span>"
                f"<span style='color:#1a1712;font-weight:600'>${r.current_price:.2f}</span>"
                f"</div>"
                f"<div style='font-size:12px;color:#6b6357;margin-top:4px'>"
                f"{sector} | Score: {r.score:.0f} | RSI: {r.rsi:.0f} | ADX: {r.adx:.0f} | "
                f"ATR: {r.atr_pct:.1f}% | "
                f"<span style='color:{mom_color}'>1M: {r.momentum_1m:+.1f}%</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # Portföy kartları (2'li sütunlar halinde)
    style_list = list(portfolios.keys())
    for i in range(0, len(style_list), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(style_list):
                style = style_list[idx]
                label, color = STYLE_INFO.get(style, (style.title(), "#6b6357"))
                with col:
                    _render_portfolio_card(label, portfolios[style], color)

    # Inline Backtest
    if run_bt_btn:
        all_bt_tickers = []
        for style, picks in portfolios.items():
            all_bt_tickers.extend([r.ticker for r in picks])
        all_bt_tickers = list(dict.fromkeys(all_bt_tickers))  # deduplicate

        if all_bt_tickers:
            st.markdown("---")
            st.markdown("## 🧪 Portfolio Backtest Results")
            run_id = f"us_sp_bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            bt_progress = st.progress(0, text="Running backtest...")
            bt_summaries = {}

            for k, ticker in enumerate(all_bt_tickers):
                bt_progress.progress((k + 1) / len(all_bt_tickers),
                                     text=f"Backtest: {ticker} ({k+1}/{len(all_bt_tickers)})")
                try:
                    trades, summary, daily = BacktestEngine._run_single(
                        ticker, us_sp_period, run_id,
                        mode=us_sp_mode, use_news=us_sp_news,
                        enable_short=False, enable_scaling=True,
                        market="US",
                    )
                    BacktestEngine._save(DB_PATH, run_id, run_date, ticker,
                                         us_sp_period, trades, summary, daily)
                    bt_summaries[ticker] = {"trades": trades, "summary": summary}
                except Exception as exc:
                    bt_summaries[ticker] = {"trades": [], "summary": {}, "error": str(exc)}

            bt_progress.progress(1.0, text="Backtest complete!")

            # Sonuçlar
            valid_bt = {k: v for k, v in bt_summaries.items()
                        if "error" not in v and v.get("summary", {}).get("total_trades", 0) > 0}

            if valid_bt:
                rets = [v["summary"]["total_return_pct"] for v in valid_bt.values()]
                wrs  = [v["summary"]["win_rate"] for v in valid_bt.values()]

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Avg Return", f"%{sum(rets)/len(rets):+.1f}")
                mc2.metric("Avg Win Rate", f"%{sum(wrs)/len(wrs):.0f}")
                mc3.metric("Total Trades", sum(v["summary"]["total_trades"] for v in valid_bt.values()))
                mc4.metric("Stocks Tested", len(valid_bt))

                # Hisse bazlı detay
                for ticker, res in valid_bt.items():
                    summary = res["summary"]
                    trades = res["trades"]
                    with st.expander(
                        f"**{ticker}** — Return: %{summary['total_return_pct']:+.1f} | "
                        f"Win: %{summary['win_rate']:.0f} | {summary['total_trades']} trades",
                        expanded=False
                    ):
                        _render_price_chart_with_trades(
                            ticker, us_sp_period, trades, height=380, market="US"
                        )
                        _tm_render_trade_table(trades, ticker)

    # Summary table
    st.markdown("---")
    st.markdown("### Full Scan Results")
    scan_rows = []
    for r in sorted(valid, key=lambda x: x.score, reverse=True):
        scan_rows.append({
            "Stock": r.ticker,
            "Sector": US_SECTOR_MAP.get(r.ticker, "Other"),
            "Score": round(r.score, 1),
            "RSI": round(r.rsi, 1),
            "ADX": round(r.adx, 1),
            "ATR %": round(r.atr_pct, 1),
            "Price": round(r.current_price, 2),
            "1M %": round(r.momentum_1m, 1),
            "3M %": round(r.momentum_3m, 1),
            "SMA200": "Above" if r.price_above_sma200 else "Below",
        })
    if scan_rows:
        st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)


def render_backtest_page(ui_lang: str):
    """Birleşik Backtest sayfası — Manuel Hisse Seçimi + Zaman Makinesi tek çatı altında."""
    st.markdown("# 🧪 Backtest")
    st.caption(
        "Geçmiş verileri kullanarak AL/SAT simülasyonu yapar. "
        "Manuel hisse seçimi veya Zaman Makinesi ile otomatik portföy testi. "
        "**Komisyon: round-trip %0.4 dahil.**"
    )

    # Mod seçici
    bt_mode_tab = st.radio(
        "Backtest Modu",
        ["Manuel Hisse Seçimi", "Zaman Makinesi"],
        horizontal=True,
        key="bt_mode_tab",
        help="Manuel: Kendi seçtiğin hisselerle test. Zaman Makinesi: Sistem otomatik portföy seçer ve test eder."
    )
    st.markdown("---")

    # Ayarlar — ANA ALANDA (eskiden kenar çubuğundaydı: mobilde panel gizli
    # kaldığı için "buton yok" görünüyordu; kritik kontroller artık hep görünür)
    c_sol, c_sag = st.columns([1, 2])
    with c_sol:
        bt_mode = st.radio(
            "Strateji Modu",
            options=["swing", "trend", "universal", "investor", "buyhold"],
            index=2,
            format_func=lambda x: BacktestEngine.MODES[x]["label"],
            help="\n".join(
                f"**{BacktestEngine.MODES[k]['label']}**: {BacktestEngine.MODES[k]['desc']}"
                for k in ["swing", "trend", "universal", "investor", "buyhold"]
            ),
            key="bt_strategy_mode",
        )
        use_news_filter = st.checkbox(
            "Haber Filtresi",
            value=True,
            help="AL sinyalinde negatif haber varsa atlar.",
            key="bt_news_filter",
        )
        cfg_disp = BacktestEngine.MODES.get(bt_mode, BacktestEngine.MODES["universal"])
        st.caption(
            f"**{cfg_disp['label']}** — {cfg_disp['desc']}  \n"
            f"AL ≥{cfg_disp['BUY_THRESHOLD']} · SAT ≤{cfg_disp['SELL_THRESHOLD']} · "
            f"Stop %{cfg_disp['STOP_PCT']*100:.0f} · TP %{cfg_disp['TP_PCT']*100:.0f} · "
            f"Max {cfg_disp['MAX_HOLD_DAYS']} gün"
        )

    with c_sag:
        if bt_mode_tab == "Manuel Hisse Seçimi":
            _preload = st.session_state.pop("preload_bt_tickers", None)
            _default_tickers = _preload if _preload else [t for t in BACKTEST_TICKERS if t in BIST_SCAN_UNIVERSE]
            selected_tickers = st.multiselect(
                "Test Edilecek Hisseler",
                options=sorted(get_scan_universe()),
                default=[t for t in _default_tickers if t in BIST_SCAN_UNIVERSE],
                placeholder="Hisse kodu ara...",
                help="5-10 hisse önerilir.",
                key="bt_manual_tickers",
            )
            pc1, pc2 = st.columns(2)
            with pc1:
                bt_period = st.selectbox(
                    "Veri Periyodu",
                    options=["1y", "2y", "3y"],
                    index=1,
                    format_func=lambda x: {"1y": "1 Yıl", "2y": "2 Yıl", "3y": "3 Yıl"}[x],
                    key="bt_period_sel",
                )
            with pc2:
                risk_slider = st.slider(
                    "Islem Basina Risk %",
                    min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="bt_risk",
                ) / 100.0
            with st.expander("Gelişmiş Ayarlar"):
                enable_short = st.checkbox("Aciga Satis (Short)", value=False, key="bt_short")
                enable_scaling = st.checkbox("Kademeli Kar Al", value=True, key="bt_scaling")
                enable_optimizer = st.checkbox("Walk-Forward Optimize", value=False, key="bt_optimizer")
            run_btn = st.button(
                "▶ Backtest Çalıştır",
                type="primary", use_container_width=True,
                disabled=len(selected_tickers) < 1,
                key="bt_run_manual",
            )
        else:
            # Zaman Makinesi ayarları
            _styles = TimeMachineEngine.PORTFOLIO_STYLES
            _style_keys = [k for k in _styles if k != "custom"]
            tc1, tc2 = st.columns(2)
            with tc1:
                tm_years = st.selectbox(
                    "Kac Yil Geriye?", [1, 2, 3, 5], index=2,
                    format_func=lambda x: f"{x} Yil",
                    key="bt_tm_years",
                )
                tm_market = st.radio("Piyasa", ["BIST", "US"], index=0,
                                     horizontal=True, key="bt_tm_market")
                st.caption("⚠️ Zaman Makinesi ESKİ portföy mantığını simüle eder — "
                           "güncel sistemin tarihsel kanıtı pipeline_backtest çıktısıdır (CLAUDE.md Roadmap-A).")
            with tc2:
                tm_style = st.selectbox(
                    "Portfoy Stili",
                    _style_keys,
                    index=0,
                    format_func=lambda x: _styles[x]["label"],
                    key="bt_tm_style",
                )
                st.caption(f"_{_styles[tm_style]['desc']}_")
            _all_tickers = get_scan_universe() if tm_market == "BIST" else US_POPULAR_TICKERS
            custom_tickers = st.multiselect(
                "Ozel Portfoy (istege bagli)",
                options=sorted(_all_tickers),
                default=[],
                key="bt_tm_custom",
                placeholder="Kendi hisselerini sec...",
            )
            run_btn = st.button(
                "▶ Zaman Makinesi Çalıştır",
                type="primary", use_container_width=True,
                key="bt_run_tm",
            )
            # Dummy vars for manual mode
            selected_tickers = []
            bt_period = "2y"
            enable_short = False
            enable_scaling = True
            enable_optimizer = False
            risk_slider = 0.02

    # ==================== MANUEL BACKTEST ====================
    if bt_mode_tab == "Manuel Hisse Seçimi":
        if run_btn and selected_tickers:
            progress_bar = st.progress(0, text="Baslatiliyor...")
            status_area  = st.empty()
            total = len(selected_tickers)
            run_id   = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{bt_mode}"
            run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

            for idx, ticker in enumerate(selected_tickers):
                status_area.info(f"**{ticker}** isleniyor... ({idx+1}/{total})")
                try:
                    opt_params = None
                    if enable_optimizer:
                        try:
                            status_area.info(f"{ticker} optimize ediliyor...")
                            opt_params = BacktestEngine.optimize(ticker, "6mo")
                        except Exception:
                            opt_params = None
                    trades, summary, daily_scores = BacktestEngine._run_single(
                        ticker, bt_period, run_id,
                        mode=bt_mode, use_news=use_news_filter,
                        enable_short=enable_short, enable_scaling=enable_scaling,
                        risk_per_trade=risk_slider, optimized_params=opt_params,
                    )
                    BacktestEngine._save(DB_PATH, run_id, run_date, ticker, bt_period, trades, summary, daily_scores)
                    status_area.success(
                        f"**{ticker}** — {summary.get('total_trades',0)} islem | "
                        f"Kazanma: %{summary.get('win_rate',0):.0f} | "
                        f"Getiri: %{summary.get('total_return_pct',0):+.1f}"
                    )
                except Exception as exc:
                    status_area.warning(f"**{ticker}** — {exc}")
                progress_bar.progress((idx + 1) / total)

            progress_bar.progress(1.0, text="Tamamlandi!")
            st.session_state["bt_run_id"] = run_id
            st.rerun()

        # Kayıtlı sonuçları göster
        _render_backtest_results(ui_lang)

    # ==================== ZAMAN MAKİNESİ ====================
    else:
        if run_btn:
            _styles = TimeMachineEngine.PORTFOLIO_STYLES
            is_custom = len(custom_tickers) > 0

            if is_custom:
                st.markdown(f"## Ozel Portfoy — {tm_years} Yil Testi")
                _tm_run_analysis(
                    custom_tickers, tm_years, tm_market, "custom",
                    bt_mode, use_news_filter, is_custom=True
                )
            else:
                style_label = _styles[tm_style]["label"]
                st.markdown(f"## {tm_years} Yil Oncesi — {style_label} Portfoy Testi")

                progress = st.progress(0, text="PIT skorlari hesaplaniyor...")
                status_txt = st.empty()

                def _progress(msg, pct):
                    progress.progress(pct, text=msg)
                    status_txt.caption(msg)

                with st.spinner(f"{tm_years} yil onceki veriler analiz ediliyor..."):
                    result = TimeMachineEngine.run_full_pit(
                        years_back=tm_years,
                        market=tm_market,
                        style=tm_style,
                        progress_cb=_progress,
                    )

                progress.progress(1.0, text="PIT analizi tamamlandi!")
                status_txt.empty()

                st.session_state["tm_last_result"] = result

                try:
                    TimeMachineEngine.save_daily_snapshot(result["run_id"], tm_market)
                except Exception:
                    pass

                picks = result.get("picks", [])
                if picks:
                    _tm_run_backtest_for_picks(picks, tm_years, tm_market, bt_mode, use_news_filter)

        # Zaman Makinesi sonuçlarını göster
        result = st.session_state.get("tm_last_result")
        if result:
            _tm_render_report(result)

        # Onceki calismalar
        try:
            tm_market_val = st.session_state.get("bt_tm_market", "BIST")
            prev_runs = TimeMachineEngine.load_previous_runs(tm_market_val)
            if prev_runs:
                st.markdown("---")
                with st.expander(f"Onceki Calismalar ({len(prev_runs)} kayit)", expanded=False):
                    for run in prev_runs[:10]:
                        _g = run.get("back_test_grade", 0)
                        gc = "#1d6f4e" if _g >= 65 else "#b45309" if _g >= 40 else "#9e2b25"
                        st.markdown(
                            f"<div style='background:#efe9db;border:1px solid #d8d0c0;"
                            f"border-radius:8px;padding:10px 14px;margin-bottom:6px'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                            f"<span style='font-weight:700;color:#1a1712'>"
                            f"{run.get('pit_date','')} -> Bugun "
                            f"({run.get('portfolio','').title()})</span>"
                            f"<span style='color:{gc};font-weight:700;font-size:18px'>"
                            f"{_g:.0f}/100</span>"
                            f"</div>"
                            f"<div style='font-size:12px;color:#6b6357;margin-top:4px'>"
                            f"Portfoy: %{run.get('portfolio_return_pct',0):+.1f} | "
                            f"Endeks: %{run.get('benchmark_return_pct',0):+.1f} | "
                            f"Alfa: %{run.get('alpha_pct',0):+.1f} | "
                            f"{run.get('stock_count',0)} hisse | "
                            f"{run.get('run_date','')}"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )
        except Exception:
            pass

    st.markdown("---")
    st.caption("Backtest gecmis performansi simule eder. Yatirim tavsiyesi degildir.")


def _render_backtest_results(ui_lang: str):
    """Manuel backtest sonuclarini gosterir."""
    all_runs = BacktestEngine.load_runs()
    if not all_runs:
        st.info("Yukarıdan hisseleri seçip **Backtest Çalıştır** butonuna bas.")
        return

    run_ids = list(dict.fromkeys(r["run_id"] for r in all_runs))
    default_run = st.session_state.get("bt_run_id", run_ids[0])
    if default_run not in run_ids:
        default_run = run_ids[0]

    col_sel, _ = st.columns([2, 3])
    with col_sel:
        selected_run = st.selectbox(
            "Backtest Sec",
            options=run_ids,
            index=run_ids.index(default_run),
            format_func=lambda r: next(
                (
                    x["run_date"] + f"  ({x['period']})"
                    + ("  [" + r.rsplit("_", 1)[-1].upper() + "]" if "_" in r else "")
                    for x in all_runs if x["run_id"] == r
                ), r
            ),
            label_visibility="collapsed",
        )

    run_rows = [r for r in all_runs if r["run_id"] == selected_run]
    if not run_rows:
        return

    tab_ozet, tab_detay = st.tabs(["Ozet", "Hisse Detayi"])

    with tab_ozet:
        valid = [r for r in run_rows if r["total_trades"] > 0]

        if valid:
            total_trades = sum(r["total_trades"] for r in valid)
            total_wins   = sum(r["winning_trades"] for r in valid)
            avg_win_rate = total_wins / total_trades * 100
            avg_return   = sum(r["avg_return_pct"] * r["total_trades"] for r in valid) / total_trades
            best_h  = max(valid, key=lambda r: r["total_return_pct"])
            worst_h = min(valid, key=lambda r: r["total_return_pct"])

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Toplam Islem",     total_trades)
            m2.metric("Kazanma Orani",    f"%{avg_win_rate:.1f}")
            m3.metric("Ort. Getiri/Islem", f"%{avg_return:+.2f}")
            m4.metric("En Iyi",  best_h["ticker"],  f"%{best_h['total_return_pct']:+.1f}")
            m5.metric("En Kotu", worst_h["ticker"], f"%{worst_h['total_return_pct']:+.1f}")

        st.markdown("---")

        def _color_cell(val, good_high=True):
            if isinstance(val, (int, float)):
                return "color: #1d6f4e" if (val > 0) == good_high else "color: #9e2b25"
            return ""

        summary_rows = []
        for row in run_rows:
            emoji = ("+" if row["win_rate"] >= 55 and row["total_return_pct"] > 0
                     else "-" if row["win_rate"] < 45 or row["total_return_pct"] < -5
                     else "~")
            summary_rows.append({
                "":            emoji,
                "Hisse":       row["ticker"],
                "Islem":       row["total_trades"],
                "Kazanma %":   row["win_rate"],
                "Ort. %":      row["avg_return_pct"],
                "Toplam %":    row["total_return_pct"],
                "Max DD %":    -row["max_drawdown_pct"],
                "En Iyi %":    row["best_trade_pct"],
                "En Kotu %":   row["worst_trade_pct"],
                "Ort. Gun":    row["avg_hold_days"],
            })

        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            st.dataframe(
                df_sum.style
                .map(_color_cell, subset=["Ort. %", "Toplam %", "En Iyi %"])
                .map(lambda v: _color_cell(v, False), subset=["Max DD %", "En Kotu %"])
                .format({
                    "Kazanma %": "{:.1f}%", "Ort. %": "{:+.2f}%",
                    "Toplam %": "{:+.2f}%", "Max DD %": "{:.1f}%",
                    "En Iyi %": "{:+.2f}%", "En Kotu %": "{:+.2f}%",
                    "Ort. Gun": "{:.1f}",
                }),
                use_container_width=True, hide_index=True,
            )

        if valid:
            st.markdown("---")
            sorted_valid = sorted(valid, key=lambda r: r["total_return_pct"], reverse=True)
            fig_bar = go.Figure(go.Bar(
                x=[r["ticker"] for r in sorted_valid],
                y=[r["total_return_pct"] for r in sorted_valid],
                marker_color=["#1d6f4e" if r["total_return_pct"] >= 0 else "#9e2b25"
                              for r in sorted_valid],
                text=[f"{r['total_return_pct']:+.1f}%" for r in sorted_valid],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Hisse Bazinda Kumulatif Backtest Getirisi",
                paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                font=dict(color=_theme()["ink"]),
                yaxis=dict(gridcolor="#d8d0c0", ticksuffix="%"),
                xaxis=dict(gridcolor="#d8d0c0"),
                showlegend=False,
                height=360, margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_detay:
        ticker_opts = [r["ticker"] for r in run_rows]
        if not ticker_opts:
            st.info("Bu backtest'te islenmis hisse yok.")
        else:
            sel_ticker = st.selectbox("Hisse Sec", ticker_opts, key="bt_detail_ticker2")
            trades_raw = BacktestEngine.load_trades(selected_run, sel_ticker)
            run_period = next((r["period"] for r in run_rows if r["ticker"] == sel_ticker), "2y")

            if not trades_raw:
                st.info(f"{sel_ticker} icin bu backtest'te islem kaydi yok.")
            else:
                sel_row = next((r for r in run_rows if r["ticker"] == sel_ticker), {})
                if sel_row:
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Islem", sel_row.get("total_trades", 0))
                    s2.metric("Kazanma", f"%{sel_row.get('win_rate', 0):.0f}")
                    s3.metric("Toplam Getiri", f"%{sel_row.get('total_return_pct', 0):+.2f}")
                    s4.metric("Max Drawdown", f"%{sel_row.get('max_drawdown_pct', 0):.1f}")

                st.markdown("---")
                _render_price_chart_with_trades(sel_ticker, run_period, trades_raw, height=420)

                returns_list = [t["return_pct"] for t in trades_raw]
                equity = [100.0]
                labels = ["Baslangic"]
                for r2, t2 in zip(returns_list, trades_raw):
                    equity.append(equity[-1] * (1 + r2 / 100))
                    labels.append(t2.get("exit_date", ""))

                fig_eq = go.Figure()
                eq_color = "#1d6f4e" if equity[-1] >= 100 else "#9e2b25"
                eq_fill  = "rgba(34,197,94,0.08)" if equity[-1] >= 100 else "rgba(239,68,68,0.08)"
                fig_eq.add_trace(go.Scatter(
                    x=labels, y=equity,
                    mode="lines+markers",
                    line=dict(color=eq_color, width=2),
                    fill="tozeroy", fillcolor=eq_fill,
                    name="Sermaye",
                    hovertemplate="%{x}<br>Sermaye: %{y:.1f}<extra></extra>",
                ))
                fig_eq.add_hline(y=100, line_dash="dash", line_color="#8a8172",
                                 annotation_text="Baslangic (100)")
                fig_eq.update_layout(
                    title=f"{sel_ticker} — Kumulatif Sermaye Egrisi",
                    paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                    font=dict(color=_theme()["ink"]),
                    yaxis=dict(gridcolor="#d8d0c0", title="Sermaye"),
                    xaxis=dict(gridcolor="#d8d0c0", tickangle=-30),
                    height=280, margin=dict(l=10, r=10, t=45, b=50),
                    showlegend=False,
                )
                st.plotly_chart(fig_eq, use_container_width=True)

                with st.expander("Islem Listesi", expanded=False):
                    reason_map = {
                        "STOP_LOSS": "Stop-Loss", "TAKE_PROFIT": "Hedef",
                        "SAT_SINYAL": "SAT Sinyali", "MAX_SURE": "Max Sure",
                        "HALA_ACIK": "Acik Pozisyon",
                    }
                    df_t = pd.DataFrame(trades_raw)
                    cols_show = [c for c in
                        ["entry_date","entry_price","exit_date","exit_price",
                         "exit_reason","return_pct","hold_days","entry_score",
                         "stop_loss","take_profit"]
                        if c in df_t.columns]
                    df_t = df_t[cols_show].rename(columns={
                        "entry_date": "Giris", "entry_price": "Giris TL",
                        "exit_date": "Cikis", "exit_price": "Cikis TL",
                        "exit_reason": "Neden", "return_pct": "Getiri%",
                        "hold_days": "Gun", "entry_score": "Skor",
                        "stop_loss": "Stop", "take_profit": "Hedef",
                    })
                    if "Neden" in df_t.columns:
                        df_t["Neden"] = df_t["Neden"].map(lambda x: reason_map.get(x, x))

                    def _rc(v):
                        if isinstance(v, (int, float)):
                            return "color:#1d6f4e;font-weight:600" if v > 0 else "color:#9e2b25;font-weight:600"
                        return ""

                    st.dataframe(
                        df_t.style.map(_rc, subset=["Getiri%"])
                        .format({"Getiri%": "{:+.2f}%", "Giris TL": "{:.2f}",
                                 "Cikis TL": "{:.2f}", "Stop": "{:.2f}",
                                 "Hedef": "{:.2f}", "Skor": "{:.1f}"}),
                        use_container_width=True, hide_index=True,
                    )

                col_pie, col_monthly = st.columns([1, 1.5])
                with col_pie:
                    reason_counts = {}
                    for t3 in trades_raw:
                        r3 = reason_map.get(t3.get("exit_reason",""), t3.get("exit_reason",""))
                        reason_counts[r3] = reason_counts.get(r3, 0) + 1
                    fig_pie = go.Figure(go.Pie(
                        labels=list(reason_counts.keys()),
                        values=list(reason_counts.values()),
                        hole=0.4, textinfo="label+percent", textfont=dict(size=11),
                    ))
                    fig_pie.update_layout(
                        title="Cikis Sebepleri",
                        paper_bgcolor=_theme()["bg"], font=dict(color=_theme()["ink"]),
                        showlegend=False, height=280,
                        margin=dict(l=10, r=10, t=45, b=10),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_monthly:
                    df_m = pd.DataFrame(trades_raw)
                    if "entry_date" in df_m.columns and "return_pct" in df_m.columns:
                        df_m["ay"] = pd.to_datetime(df_m["entry_date"]).dt.to_period("M").astype(str)
                        monthly = df_m.groupby("ay").agg(
                            Islem=("ay","count"),
                            Kazanan=("return_pct", lambda x: (x>0).sum()),
                            Ort=("return_pct","mean"),
                            Top=("return_pct","sum"),
                        ).reset_index().rename(columns={"ay":"Ay","Ort":"Ort%","Top":"Top%"})
                        monthly["Ort%"] = monthly["Ort%"].round(2)
                        monthly["Top%"] = monthly["Top%"].round(2)
                        st.markdown("**Aylik Ozet**")
                        def _rc2(v):
                            if isinstance(v, float):
                                return "color:#1d6f4e" if v > 0 else "color:#9e2b25"
                            return ""
                        st.dataframe(
                            monthly.style.map(_rc2, subset=["Ort%","Top%"])
                            .format({"Ort%":"{:+.2f}%","Top%":"{:+.2f}%"}),
                            use_container_width=True, hide_index=True, height=260,
                        )


def render_portfolio_page(ui_lang):
    title = "Portföyüm & Favoriler" if ui_lang == "TR" else "Portfolio & Favorites"
    st.markdown(f"# {title}")

    st.markdown("### Yeni Hisse Ekle" if ui_lang == "TR" else "### Add Stock")
    with st.form("portfolio_ekle_form", clear_on_submit=True):
        cp1, cp2, cp3, cp4 = st.columns([2, 1.5, 1.5, 1])
        with cp1:
            p_tick = st.text_input(
                "Hisse Kodu" if ui_lang == "TR" else "Ticker",
                placeholder="Örn: THYAO",
                help="BIST kodu girin (örn: THYAO, GARAN). US hisseleri için AAPL:US formatı kullanın."
                     if ui_lang == "TR" else "Enter BIST ticker (e.g. THYAO). For US stocks use AAPL:US format."
            ).strip().upper()
        with cp2:
            p_cost = st.number_input(
                "Maliyet (TL)" if ui_lang == "TR" else "Cost (TL)",
                min_value=0.0, format="%.2f",
                help="Hisse başına ortalama alış fiyatınız" if ui_lang == "TR" else "Average cost per share"
            )
        with cp3:
            p_qty = st.number_input(
                "Adet" if ui_lang == "TR" else "Quantity",
                min_value=0, step=1,
                help="Toplam adet (0 = sadece takip listesi)" if ui_lang == "TR" else "Total shares (0 = watchlist only)"
            )
        with cp4:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Ekle (Add)", use_container_width=True)

        if submitted and p_tick and _guard_write():
            _history_db.add_portfolio(p_tick, p_cost, p_qty)
            st.success(f"{p_tick} portföye eklendi." if ui_lang == "TR" else f"{p_tick} added to portfolio.")
            st.rerun()

    st.markdown("---")
    st.markdown("### Mevcut Portföy" if ui_lang == "TR" else "### Current Portfolio")

    portfolio_items = _history_db.get_portfolio()
    if not portfolio_items:
        st.info(
            "Portföyünüz henüz boş. Yukarıdaki formu kullanarak hisse ekleyebilir "
            "veya **Hisse Analizi** sayfasında bir hisseyi analiz edip **Favorilere Ekle** butonuna basabilirsiniz."
            if ui_lang == "TR" else
            "Your portfolio is empty. Use the form above to add stocks, "
            "or analyze a stock in the **Analysis** page and click **Add to Portfolio**."
        )
        return

    # Gerçek zamanlı fiyatlar (5 dak. cache, toplu çekim)
    @st.cache_data(ttl=600, show_spinner=False)
    def _fetch_realtime_prices(tickers: tuple) -> dict:
        """Portföydeki tüm hisseler için anlık fiyatları tek sorguda çeker (BIST + US)."""
        result = {}
        if not tickers:
            return result
        # US hisseleri :US suffix ile saklanir (e.g. AAPL:US)
        bist_ticks = [t for t in tickers if not t.endswith(":US")]
        us_ticks   = [t for t in tickers if t.endswith(":US")]

        # BIST fiyatları
        if bist_ticks:
            yt_bist = [_yf_symbol(t, "BIST") for t in bist_ticks]
            try:
                df = yf.download(yt_bist, period="2d", group_by="ticker", auto_adjust=True, progress=False)
                for tick in bist_ticks:
                    yt = _yf_symbol(tick, "BIST")
                    try:
                        col = df[yt]["Close"] if len(bist_ticks) > 1 else df["Close"]
                        col = col.dropna()
                        result[tick] = float(col.iloc[-1]) if not col.empty else 0.0
                    except Exception:
                        result[tick] = 0.0
            except Exception:
                pass

        # US fiyatları
        if us_ticks:
            us_symbols = [t.replace(":US", "") for t in us_ticks]
            try:
                df_us = yf.download(us_symbols, period="2d", group_by="ticker", auto_adjust=True, progress=False)
                for tick, sym in zip(us_ticks, us_symbols):
                    try:
                        col = df_us[sym]["Close"] if len(us_symbols) > 1 else df_us["Close"]
                        col = col.dropna()
                        result[tick] = float(col.iloc[-1]) if not col.empty else 0.0
                    except Exception:
                        result[tick] = 0.0
            except Exception:
                pass
        return result

    tickers_tuple = tuple(item["ticker"] for item in portfolio_items)
    with st.spinner("Fiyatlar güncelleniyor..." if ui_lang == "TR" else "Fetching prices..."):
        realtime_prices = _fetch_realtime_prices(tickers_tuple)

    history = _history_db.load_all()
    p_hist = {r["ticker"]: r for r in history}

    total_cost = 0.0
    total_val  = 0.0
    chart_labels, chart_kz_pct, chart_alloc = [], [], []

    for item in portfolio_items:
        tick    = item["ticker"]
        maliyet = item["buy_price"]
        adet    = item["quantity"]

        # US or BIST detection
        is_us = tick.endswith(":US")
        display_tick = tick.replace(":US", "") if is_us else tick
        currency = "$" if is_us else "TL"
        market_badge = " <span style='color:#27509e;font-size:10px'>US</span>" if is_us else ""

        # Gerçek zamanlı fiyat, yoksa cache'deki analiz fiyatı
        guncel_fiyat = realtime_prices.get(tick, 0.0)
        if guncel_fiyat == 0.0:
            hist_data = p_hist.get(tick)
            guncel_fiyat = hist_data["current_price"] if hist_data and hist_data.get("current_price") else 0.0

        hist_data = p_hist.get(tick)
        sinyal = hist_data["signal"] if hist_data else "N/A"
        skor   = hist_data["total_score"] if hist_data else 0

        item_cost = maliyet * adet
        item_val  = guncel_fiyat * adet
        kar_zarar = item_val - item_cost
        kar_pct   = (kar_zarar / item_cost * 100) if item_cost > 0 else 0

        total_cost += item_cost
        total_val  += item_val

        # Grafik verileri
        chart_labels.append(display_tick)
        chart_kz_pct.append(round(kar_pct, 2))
        chart_alloc.append(item_cost)

        c_color  = "#1d6f4e" if sinyal in ["AL", "GUCLU AL"] else "#9e2b25" if sinyal in ["SAT", "GUCLU SAT"] else "#6b6357"
        kz_color = "#1d6f4e" if kar_zarar >= 0 else "#9e2b25"

        with st.container():
            st.markdown(f"""
    """, unsafe_allow_html=True)
            if st.button("Sil (Remove)", key=f"p_del_page_{tick}") and _guard_write():
                _history_db.remove_portfolio(tick)
                st.rerun()

    # Genel Durum Özeti
    st.markdown("---")
    genel_kz  = total_val - total_cost
    genel_pct = (genel_kz / total_cost * 100) if total_cost > 0 else 0
    gk_color  = "#1d6f4e" if genel_kz >= 0 else "#9e2b25"
    st.markdown(
        f"<div style='background:#f7f3ea; padding:20px; border-radius:12px; border:2px solid {gk_color}; text-align:center;'>"
        f"<h3 style='margin:0;'>Genel Durum</h3>"
        f"<p style='font-size:18px; color:#4d463c; margin:10px 0;'>Toplam Maliyet: <b>{total_cost:,.2f} TL</b> &nbsp;|&nbsp; Güncel Değer: <b>{total_val:,.2f} TL</b></p>"
        f"<h2 style='color:{gk_color}; margin:0;'>Kâr/Zarar: {genel_kz:+,.2f} TL (%{genel_pct:+.2f})</h2>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # P&L Grafikleri
    if len(portfolio_items) > 1:
        st.markdown("---")
        st.markdown("### Portföy Grafikleri" if ui_lang == "TR" else "### Portfolio Charts")
        g1, g2 = st.columns(2)

        with g1:
            # K/Z bar chart
            bar_colors = ["#1d6f4e" if v >= 0 else "#9e2b25" for v in chart_kz_pct]
            fig_bar = go.Figure(go.Bar(
                x=chart_labels, y=chart_kz_pct,
                marker_color=bar_colors,
                text=[f"{v:+.1f}%" for v in chart_kz_pct],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Hisse Bazında K/Z (%)" if ui_lang == "TR" else "P&L per Stock (%)",
                paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["panel"],
                font=dict(color=_theme()["ink"]),
                yaxis=dict(gridcolor="#d8d0c0", ticksuffix="%"),
                xaxis=dict(gridcolor="#d8d0c0"),
                showlegend=False,
                height=360, margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with g2:
            # Dağılım pie chart
            fig_pie = go.Figure(go.Pie(
                labels=chart_labels,
                values=chart_alloc,
                hole=0.45,
                textinfo="label+percent",
                marker=dict(line=dict(color="#f7f3ea", width=2)),
            ))
            fig_pie.update_layout(
                title="Portföy Dağılımı (Maliyet)" if ui_lang == "TR" else "Portfolio Allocation (Cost)",
                paper_bgcolor=_theme()["bg"],
                font=dict(color=_theme()["ink"]),
                height=360, margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# BIST Hisse Listesi (BIST30 + seçili BIST50/100)
BIST_STOCKS = {
    "BIST 30": [
        "AKBNK","ARCLK","ASELS","BIMAS","DOHOL","EKGYO","ENKAI","EREGL",
        "FROTO","GARAN","GUBRF","HALKB","ISCTR","KCHOL","KOZAA","KOZAL",
        "KRDMD","LOGO","MGROS","ODAS","PETKM","PGSUS","SAHOL","SASA",
        "SISE","TAVHL","TCELL","THYAO","TKFEN","TOASO","TTKOM","TUPRS",
        "VAKBN","VESBE","YKBNK",
    ],
    "Bankacılık": ["AKBNK","GARAN","HALKB","ISCTR","VAKBN","YKBNK","QNBFB","ALBRK"],
    "Havacılık & Ulaşım": ["THYAO","PGSUS","TAVHL","CLEBI","UCAK"],
    "Teknoloji": ["ASELS","LOGO","NETAS","INDES","DGATE","KAREL"],
    "Enerji": ["ODAS","AKSEN","ZOREN","AYEN","ENJSA","EUPWR"],
    "Sanayi & Otomotiv": ["FROTO","TOASO","ARCLK","VESBE","TTRAK","OTKAR"],
    "Perakende": ["BIMAS","MGROS","SOKM","MAVI","LCWGR"],
    "Holding": ["KCHOL","SAHOL","DOHOL","ENKAI","TKFEN"],
    "Çelik & Metal": ["EREGL","KRDMD","CEMTS","IEYHO"],
    "Kimya & Petrokimya": ["TUPRS","PETKM","SASA","GUBRF"],
}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_bist_daily_changes(tickers: tuple) -> dict:
    """
    """
    result = {}
    if not tickers:
        return result
    try:
        symbols = [_yf_symbol(t, "BIST") for t in tickers]
        raw = yf.download(
            symbols, period="5d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            return result

        def _get_close(raw_df: pd.DataFrame, sym: str, is_multi: bool) -> pd.Series:
            """Versiyon bağımsız Close serisi döndürür."""
            if not is_multi:
                # Tek hisse — flat DataFrame
                return raw_df["Close"].dropna() if "Close" in raw_df.columns else pd.Series(dtype=float)

            cols = raw_df.columns
            if not isinstance(cols, pd.MultiIndex):
                return raw_df["Close"].dropna() if "Close" in raw_df.columns else pd.Series(dtype=float)

            lvl0 = cols.get_level_values(0)
            lvl1 = cols.get_level_values(1)

            # Yeni format: ilk seviye Price ("Close","Open",...), ikinci Ticker
            if "Close" in lvl0:
                try:
                    return raw_df["Close"][sym].dropna()
                except Exception:
                    pass

            # Eski format: ilk seviye Ticker, ikinci Price
            if sym in lvl0:
                try:
                    return raw_df[sym]["Close"].dropna()
                except Exception:
                    pass

            return pd.Series(dtype=float)

        is_multi = len(symbols) > 1
        for t, sym in zip(tickers, symbols):
            try:
                col = _get_close(raw, sym, is_multi)
                if len(col) >= 2:
                    cur  = float(col.iloc[-1])
                    prev = float(col.iloc[-2])
                    pct  = (cur - prev) / prev * 100
                    result[t] = {"price": cur, "pct": round(pct, 2), "up": pct >= 0}
                elif len(col) == 1:
                    result[t] = {"price": float(col.iloc[-1]), "pct": 0.0, "up": True}
            except Exception:
                pass
    except Exception:
        pass
    return result


def render_bist_list_page(ui_lang):
    """BIST hisse listesi — kategori bazlı, günlük değişimli, tıklanabilir."""
    st.markdown("# 📋 BIST Hisse Listesi" if ui_lang == "TR" else "# 📋 BIST Stock List")
    st.caption("Hisseye tıkla → otomatik olarak Hisse Analizi sekmesine geçer ve analizi başlatır.")
    st.markdown("---")

    # Arama kutusu
    search = st.text_input(
        "Hisse Ara", placeholder="örn: THYAO, Türk Hava...",
        key="bist_list_search"
    ).strip().upper()

    # Tüm tickerları düz listeye al (arama için)
    all_tickers_flat = list({t for cats in BIST_STOCKS.values() for t in cats})

    # Kategori sekmeleri
    cat_names = list(BIST_STOCKS.keys())
    tabs = st.tabs(cat_names)

    for tab, cat in zip(tabs, cat_names):
        with tab:
            tickers_in_cat = tuple(BIST_STOCKS[cat])

            # Arama filtresi
            if search:
                tickers_in_cat = tuple(t for t in tickers_in_cat if search in t)
                if not tickers_in_cat:
                    st.info("Arama sonucu bulunamadı.")
                    continue

            with st.spinner("Fiyatlar güncelleniyor..."):
                prices = fetch_bist_daily_changes(tickers_in_cat)

            # Başlık satırı
            header_cols = st.columns([2, 2, 2, 3])
            header_cols[0].markdown("**Hisse**")
            header_cols[1].markdown("**Fiyat (₺)**")
            header_cols[2].markdown("**Günlük Değ.**")
            header_cols[3].markdown("**İşlem**")
            st.markdown("<hr style='margin:4px 0 8px 0;border-color:#d8d0c0'>", unsafe_allow_html=True)

            for tk in tickers_in_cat:
                info = prices.get(tk)
                c1, c2, c3, c4 = st.columns([2, 2, 2, 3])

                with c1:
                    st.markdown(
                        f"<img src='{_logo_url(tk)}' width='20' "
                        f"style='vertical-align:middle;border-radius:5px;margin-right:6px'>"
                        f"<b>{tk}</b>",
                        unsafe_allow_html=True)

                with c2:
                    if info:
                        st.markdown(f"{info['price']:.2f} ₺")
                    else:
                        st.markdown("—")

                with c3:
                    if info:
                        p_color = "#1d6f4e" if info["up"] else "#9e2b25"
                        arrow   = "▲" if info["up"] else "▼"
                        st.markdown(
                            f"<span style='color:{p_color};font-weight:600'>"
                            f"{arrow} {info['pct']:+.2f}%</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("<span style='color:#6b6357'>—</span>", unsafe_allow_html=True)

                with c4:
                    if st.button(
                        f"📊 Analiz Et",
                        key=f"list_analyze_{cat}_{tk}",
                        help=f"{tk} analiz et",
                        use_container_width=True,
                    ):
                        st.session_state["selected_ticker"] = tk
                        st.session_state["auto_analyze"]    = True
                        st.session_state["nav_page"]        = "Hisse Analizi"
                        st.rerun()


def render_analysis_page(ui_lang):

    # Session state ile hisse seçimi
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = "THYAO"

    # Girişler ANA ALANDA — mobilde kenar çubuğu gizli kalıyordu (backtest dersi)
    gc1, gc2, gc3 = st.columns([2, 1, 1])
    with gc1:
        ticker_input = st.text_input(
            "Hisse Kodu / Stock Code",
            value=st.session_state["selected_ticker"],
            placeholder="örn: THYAO, GARAN, SISE",
            help="Borsa İstanbul kodu (.IS otomatik eklenir)",
            key="ticker_text_input",
        ).strip().upper()
        if ticker_input != st.session_state["selected_ticker"]:
            st.session_state["selected_ticker"] = ticker_input
    with gc2:
        period = st.selectbox(
            "Periyot",
            options=["6mo", "1y", "2y"], index=1,
            format_func=lambda x: {"6mo": "6 Ay", "1y": "1 Yıl", "2y": "2 Yıl"}[x],
        )
    with gc3:
        analyst_target = st.number_input(
            "Hedef Fiyat (ops.)",
            min_value=0.0, value=0.0, step=0.5,
            help="Elle analist hedefi — boşsa TradingView konsensüsü otomatik kullanılır."
                 if ui_lang == "TR" else "Manual target; TradingView consensus auto-used if 0.",
        )
        analyst_target = analyst_target if analyst_target > 0 else None

    bc1, bc2 = st.columns([3, 1])
    with bc1:
        analyze_btn = st.button(
            "🔍 Analiz Et" if ui_lang == "TR" else "🔍 Analyze",
            use_container_width=True, type="primary"
        )
    with bc2:
        force_update = st.checkbox(
            "Zorla Yenile" if ui_lang == "TR" else "Force Update",
            value=False,
            help="Önbelleği yoksayıp taze veri çeker." if ui_lang == "TR" else "Bypass cache."
        )
    # Liste veya geçmişten seçilince otomatik analiz flag'i
    if st.session_state.get("auto_analyze"):
        analyze_btn = True
        st.session_state["auto_analyze"] = False

    with st.sidebar:
        # Analiz Geçmişi (ikincil bilgi — kenar çubuğunda kalması doğal)
        history = _history_db.load_all()
        if history:
            st.markdown("### " + ("Analiz Geçmişi" if ui_lang == "TR" else "Analysis History"))
            for row in history:
                sig   = row["signal"]
                color = {
                    "GUCLU AL": "#1d6f4e", "AL": "#3e8e6c",
                    "NOTR": "#6b6357",
                    "SAT": "#b45309",   "GUCLU SAT": "#9e2b25",
                }.get(sig, "#6b6357")
                count = row.get("analysis_count", 1)
                price = row.get("current_price") or 0
                cnt_label = f" · ×{count}" if count > 1 else ""

                h_col1, h_col2 = st.columns([3, 1])
                with h_col1:
                    # Tıklanabilir hisse butonu → seç + analizi otomatik başlat
                    if st.button(
                        f"{row['ticker']}",
                        key=f"hist_pick_{row['ticker']}",
                        help=f"{row['ticker']} tekrar analiz et",
                        use_container_width=True,
                    ):
                        st.session_state["selected_ticker"] = row["ticker"]
                        st.session_state["auto_analyze"]    = True
                        st.rerun()
                    st.markdown(
                        f"<div style='font-size:10px;color:#8a8172;margin-top:-8px'>"
                        f"<span style='color:{color}'>{sig} {row['total_score']:.0f}{cnt_label}</span>"
                        f" · {price:.2f} ₺ · {row['last_analyzed'][:10]}</div>",
                        unsafe_allow_html=True,
                    )
                with h_col2:
                    if st.button("🗑", key=f"del_hist_{row['ticker']}",
                                 help=f"{row['ticker']} sil") and _guard_write():
                        _history_db.delete(row["ticker"])
                        st.rerun()

        st.markdown("---")

    title_text = "BIST Akilli Yatirim Asistani" if ui_lang == "TR" else "BIST Smart Investment Assistant"
    st.markdown(f"# {title_text}")

    if not analyze_btn:
        # İlk açılış / Onboarding
        st.info(
            "Sol menüden hisse kodu girin ve **Analiz Et** butonuna basın."
            if ui_lang == "TR" else
            "Enter a stock code in the sidebar and click **Analyze**."
        )
        with st.expander(
            "Nasıl Çalışır? — Skor Sistemi Rehberi" if ui_lang == "TR" else
            "How It Works — Scoring Guide", expanded=False
        ):
            st.markdown(
                """
    """ if ui_lang == "TR" else
                """
    """
            )
        return

    # Input validation
    if not ticker_input or len(ticker_input) < 2:
        st.warning(
            "Lütfen geçerli bir hisse kodu girin (örn: THYAO, GARAN)."
            if ui_lang == "TR" else
            "Please enter a valid stock ticker (e.g., THYAO, GARAN)."
        )
        return
    # Sadece harf ve rakam kabul et
    import re as _re
    if not _re.match(r'^[A-Z0-9]+$', ticker_input):
        st.warning(
            "Hisse kodu sadece harf ve rakam içermelidir."
            if ui_lang == "TR" else
            "Stock ticker must contain only letters and numbers."
        )
        return

    start_time = time.time()
    
    # Akıllı Cache (Önbellek) Kontrolü
    cached_data = _history_db.load_full(ticker_input) if not force_update else None
    is_cached = False
    
    if cached_data:
        last_time_str, cached_score = cached_data
        try:
            last_t = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M")
            hours_diff = (datetime.now() - last_t).total_seconds() / 3600
        except Exception:
            hours_diff = 999
            
        if hours_diff < 8.0:  # 8 saatten yeniyse cache'den kullan
            score = cached_score
            is_cached = True
            elapsed = 0.0
            st.success(
                f"**{ticker_input}** ("
                f"Son Analiz: {last_time_str}) — Veriler önbellekten 0.1 saniyede yüklendi. "
                f"Güncellemek için yandaki *Zorla Yenile* kutusunu işaretleyebilirsiniz."
            )
            
    # Eğer cache'den yüklenemediyse normal analiz yap
    if not is_cached:
        spinner_msg = f"Analiz ediliyor: {ticker_input}..." if ui_lang == "TR" else f"Analyzing {ticker_input}..."
        with st.spinner(spinner_msg):
            score = compute_bist_score(ticker_input, analyst_target, period, language="BOTH")

        if score.stock.error:
            st.error(score.stock.error)
            return

        # Analiz sonucunu database'e kaydet
        try:
            _history_db.save(score)
        except Exception as _db_exc:
            log.warning("Analiz kaydetme hatası: %s", _db_exc)
            
        elapsed = time.time() - start_time

    risk = RiskEngine.compute(score)

    # Sistem Diagnostik Paneli (gizli, geliştirici modu)
    with st.expander("🔧 Sistem Diagnostiği", expanded=False):
        t_diag = score.technical
        df_diag = score.stock.df
        ok   = "✅"
        warn = "⚠️"
        err  = "❌"

        def _chk(val, zero_bad=True):
            if val is None:       return err
            if zero_bad and val == 0.0: return warn
            return ok

        cols_present = list(df_diag.columns) if df_diag is not None and not df_diag.empty else []
        has_high  = "High"   in cols_present
        has_low   = "Low"    in cols_present
        has_vol   = "Volume" in cols_present
        data_rows = len(df_diag) if df_diag is not None else 0

        st.markdown(f"""
    """)
        if t_diag.adx == 0.0:
            st.warning("ADX sıfır — log dosyasına (bist_analyzer.log) bak. Büyük ihtimal High/Low veri sorunu.")
        if t_diag.atr == 0.0:
            st.warning("ATR sıfır — High/Low kolon kontrolü başarısız olmuş olabilir.")
        log_path = LOG_PATH
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as lf:
                lines = lf.readlines()
            recent = [l for l in lines[-200:] if "WARNING" in l or "ERROR" in l]
            if recent:
                st.markdown("**Son Uyarı/Hatalar (log):**")
                st.code("".join(recent[-20:]), language="text")
            else:
                st.success("Log temiz — son 200 satırda uyarı/hata yok.")

    # Şirket Bilgi Kartı
    info = score.stock.info
    if info:
        isim = info.get("longName", ticker_input)
        sektor = info.get("sector", "Bilinmiyor")
        endustri = info.get("industry", "Bilinmiyor")
        ozet = info.get("longBusinessSummary", "Şirket özeti bulunamadı.")
        web = info.get("website", "")
        
        st.markdown("---")
        with st.expander(f"{isim} — Şirket Profili & Bilgiler" if ui_lang == "TR" else f"{isim} — Company Profile", expanded=False):
            st.markdown(f"**Sektör:** {sektor} | **Endüstri:** {endustri}")
            st.markdown(ozet)
            if web:
                st.markdown(f"**Web:** [{web}]({web})")
                
            if st.button("Favorilere (Portföye) Ekle" if ui_lang == "TR" else "Add to Portfolio",
                         key=f"fav_{ticker_input}") and _guard_write():
                _history_db.add_portfolio(ticker_input, score.stock.current_price, 0)
                st.success(f"{ticker_input} portföyünüze/favorilerinize eklendi!" if ui_lang == "TR" else f"{ticker_input} added!")

    # Top Metrics (responsive: 3+3 layout)
    t = score.technical
    metrics = [
        (f"{score.stock.current_price:.2f}", "Fiyat (TL)" if ui_lang == "TR" else "Price"),
        (f"{t.rsi:.1f}", "RSI (14)"),
        (f"{t.adx:.1f}", "ADX" + (" (Trend Gücü)" if ui_lang == "TR" else " (Trend)")),
        (f"{t.stoch_k:.1f} / {t.stoch_d:.1f}", "Stochastic %K/%D"),
        (f"{score.stock.pe_ratio:.1f}x" if score.stock.pe_ratio else "N/A",
         "F/K Oranı" if ui_lang == "TR" else "P/E Ratio"),
        (f"%{score.valuation.prim_pct:.1f}" if score.valuation.prim_pct is not None else "—",
         "Prim Potansiyeli" if ui_lang == "TR" else "Upside"),
    ]
    # İlk satır 3 metrik, ikinci satır 3 metrik (mobilde daha okunur)
    row1 = st.columns(3)
    row2 = st.columns(3)
    all_cols = row1 + row2
    for col, (val, label) in zip(all_cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge + Sub-scores
    col_gauge, col_detail = st.columns([1, 1.5])
    with col_gauge:
        st.markdown(
            f"<div class='signal-box' style='color:{score.signal_color}'>"
            f"{score.signal}</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(create_gauge_chart(score.total_score), use_container_width=True)

    with col_detail:
        st.markdown(
            "#### " + ("Skor Bileşenleri" if ui_lang == "TR" else "Score Components")
        )
        _sub_labels = {
            "TR": {
                "Technical Analysis": ("Teknik Analiz", "RSI, MACD, SMA, hacim trendleri"),
                "News Sentiment":     ("Haber Sentiment", "20+ kaynaktan duygu analizi"),
                "Upside Potential":   ("Prim Potansiyeli", "Analist hedef fiyat karşılaştırması"),
                "Valuation":          ("Değerleme", "F/K, PD/DD, ROE oranları"),
            },
            "EN": {
                "Technical Analysis": ("Technical Analysis", "RSI, MACD, SMA, volume trends"),
                "News Sentiment":     ("News Sentiment", "Sentiment from 20+ sources"),
                "Upside Potential":   ("Upside Potential", "Analyst target price comparison"),
                "Valuation":          ("Valuation", "P/E, P/B, ROE ratios"),
            },
        }
        _labels = _sub_labels.get(ui_lang, _sub_labels["EN"])
        for key, (val, weight) in {
            "Technical Analysis": (score.teknik_score,    WEIGHTS["teknik"]),
            "News Sentiment":     (score.sentiment_score, WEIGHTS["sentiment"]),
            "Upside Potential":   (score.prim_score,      WEIGHTS["prim"]),
            "Valuation":          (score.deger_score,     WEIGHTS["deger"]),
        }.items():
            loc_label, loc_desc = _labels.get(key, (key, ""))
            _val_int = max(0, min(100, int(val)))
            _quality = ("Güçlü" if val >= 65 else "Zayıf" if val < 40 else "Orta") if ui_lang == "TR" \
                       else ("Strong" if val >= 65 else "Weak" if val < 40 else "Moderate")
            st.markdown(f"**{loc_label}** — {_quality} ({weight}%)", help=loc_desc)
            st.progress(_val_int, text=f"{val:.0f}/100")

    # Chart
    st.markdown("---")
    st.plotly_chart(
        create_candlestick_chart(score.stock.df, ticker_input),
        use_container_width=True,
    )

    # Risk & Exit Strategy
    st.markdown("---")
    st.markdown("### 🛡️ Risk & Exit Strategy")
    if risk.saturation_warning:
        st.warning(risk.saturation_message)

    t_risk = score.technical
    atr_used = t_risk.atr if t_risk.atr > 0 else score.stock.current_price * 0.02

    # R/R renk kodu
    rr = risk.risk_reward_ratio
    if rr >= 2.0:
        rr_color, rr_label = "#1d6f4e", "Mükemmel"
    elif rr >= 1.5:
        rr_color, rr_label = "#3e8e6c", "Kabul Edilebilir"
    elif rr >= 1.0:
        rr_color, rr_label = "#a2701d", "Zayıf"
    else:
        rr_color, rr_label = "#9e2b25", "Yetersiz"

    # Tight stop mesafesi %
    tight_pct  = (score.stock.current_price - risk.stop_loss_tight)  / score.stock.current_price * 100
    normal_pct = (score.stock.current_price - risk.stop_loss_normal) / score.stock.current_price * 100
    wide_pct   = (score.stock.current_price - risk.stop_loss_wide)   / score.stock.current_price * 100
    tp1_pct    = (risk.take_profit_1 - score.stock.current_price)    / score.stock.current_price * 100
    tp2_pct    = (risk.take_profit_2 - score.stock.current_price)    / score.stock.current_price * 100

    # TP2 kaynağı etiketi
    if score.valuation.target_price and score.valuation.target_price > score.stock.current_price * 1.03:
        tp2_source = "Analist Hedefi"
    elif t_risk.week52_high and t_risk.week52_high > score.stock.current_price * 1.05 and t_risk.week52_position < 0.85:
        tp2_source = "52H Yüksek"
    else:
        tp2_source = "ATR Bazlı"

    st.markdown(
        f"<div style='background:#efe9db;border:1px solid #d8d0c0;border-radius:8px;"
        f"padding:10px 14px;margin-bottom:12px;font-size:13px;color:#6b6357'>"
        f"📐 <b>ATR (14g):</b> <span style='color:#1a1712'>{atr_used:.2f} TL</span> &nbsp;|&nbsp; "
        f"<b>Oynaklık:</b> <span style='color:#1a1712'>%{t_risk.atr_pct:.2f}</span> &nbsp;|&nbsp; "
        f"<b>Stop-loss</b> ATR çarpanına göre hesaplandı (1.5×, 2.5×, SMA200)</div>",
        unsafe_allow_html=True,
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**🔴 Stop-Loss Seviyeleri**")
        st.markdown(f"""
    """, unsafe_allow_html=True)

    with col_r2:
        st.markdown("**🟢 Hedef Fiyatlar**")
        st.markdown(f"""
    """, unsafe_allow_html=True)

    # Teknik Gösterge Detayları
    st.markdown("---")
    st.markdown("### Teknik Gösterge Detayları" if ui_lang == "TR" else "### Technical Indicator Details")

    def _signal_badge(cond: bool, true_label: str, false_label: str,
                      true_color: str = "#1d6f4e", false_color: str = "#9e2b25") -> str:
        color, label = (true_color, true_label) if cond else (false_color, false_label)
        return (f"<span style='background:{color};color:#fff;padding:2px 8px;"
                f"border-radius:4px;font-size:12px;font-weight:600'>{label}</span>")

    tab_sma, tab_osc, tab_vol, tab_risk_detail = st.tabs([
        "Trend & SMA", "Osilatörler", "Hacim & OBV", "Volatilite & Pozisyon"
    ])

    with tab_sma:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**SMA Seviyeleri**")
            gc_color  = "#1d6f4e" if t.golden_cross else "#9e2b25"
            gc_label  = "Golden Cross" if t.golden_cross else "Death Cross"
            gap_label = f"Gap: {t.sma_gap_pct:+.1f}%"
            st.markdown(
                f"SMA 50: **{t.sma50:.2f} TL**  \n"
                f"SMA 200: **{t.sma200:.2f} TL**  \n"
                f"{_signal_badge(t.golden_cross, gc_label, gc_label, gc_color, gc_color)} "
                f"<span style='color:#6b6357;font-size:12px'>{gap_label}</span>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**Fiyat / SMA Konumu**")
            st.markdown(
                f"{_signal_badge(t.price_above_sma50,  'Fiyat > SMA50',  'Fiyat < SMA50')}  \n\n"
                f"{_signal_badge(t.price_above_sma200, 'Fiyat > SMA200', 'Fiyat < SMA200')}",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown("**ADX — Trend Gücü**")
            adx_color = "#1d6f4e" if t.adx > 25 else "#a2701d" if t.adx > 20 else "#9e2b25"
            adx_label = "Güçlü Trend" if t.adx > 25 else ("Zayıf Trend" if t.adx > 20 else "Yatay Piyasa")
            st.markdown(
                f"ADX: <b style='color:{adx_color}'>{t.adx:.1f}</b> — {adx_label}  \n"
                f"<span style='color:#6b6357;font-size:12px'>"
                f"ADX &gt; 25: trend sinyalleri güvenilir  \n"
                f"ADX &lt; 20: yatay piyasa, sinyaller filtreli</span>",
                unsafe_allow_html=True,
            )

    with tab_osc:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**RSI (14)**")
            rsi = t.rsi
            if rsi < 30:
                rsi_label, rsi_color = "Aşırı Satım — Alım Fırsatı", "#1d6f4e"
            elif rsi < 40:
                rsi_label, rsi_color = "Ucuz Bölge", "#3e8e6c"
            elif rsi < 60:
                rsi_label, rsi_color = "Normal Bölge", "#6b6357"
            elif rsi < 70:
                rsi_label, rsi_color = "Hafif Pahalı", "#a2701d"
            else:
                rsi_label, rsi_color = "Aşırı Alım — Dikkat!", "#9e2b25"
            st.markdown(
                f"RSI: <b style='color:{rsi_color}'>{rsi:.1f}</b> — {rsi_label}",
                unsafe_allow_html=True,
            )
            st.progress(int(rsi))

        with c2:
            st.markdown("**Stochastic (%K / %D)**")
            sk, sd = t.stoch_k, t.stoch_d
            if sk < 20:
                st_label, st_color = "Aşırı Satım", "#1d6f4e"
            elif sk > 80:
                st_label, st_color = "Aşırı Alım", "#9e2b25"
            elif sk > sd:
                st_label, st_color = "Yukarı Kesişim", "#3e8e6c"
            else:
                st_label, st_color = "Aşağı Kesişim", "#b45309"
            st.markdown(
                f"%K: <b style='color:{st_color}'>{sk:.1f}</b> &nbsp; %D: **{sd:.1f}** — {st_label}",
                unsafe_allow_html=True,
            )
            st.progress(int(max(0, min(100, sk))))

        st.markdown("**MACD**")
        hist = t.macd_histogram
        hist_color = "#1d6f4e" if hist > 0 else "#9e2b25"
        macd_strength = "Güçlü" if abs(hist) / max(t.current_price, 1) * 100 >= 0.5 else "Zayıf"
        st.markdown(
            f"MACD: **{t.macd:.4f}** &nbsp;|&nbsp; Sinyal: **{t.macd_signal:.4f}** &nbsp;|&nbsp; "
            f"Histogram: <b style='color:{hist_color}'>{hist:+.4f}</b> "
            f"({macd_strength} {'Boğa' if hist > 0 else 'Ayı'} Momentumu)",
            unsafe_allow_html=True,
        )

        st.markdown("**Bollinger Bantları (20, 2σ)**")
        bb_pos_pct = round(t.bb_position * 100)
        if t.bb_position <= 0.20:
            bb_label, bb_color = "Alt Banda Yakın — Alım Bölgesi", "#1d6f4e"
        elif t.bb_position <= 0.50:
            bb_label, bb_color = "Alt Yarı — Normal", "#3e8e6c"
        elif t.bb_position <= 0.80:
            bb_label, bb_color = "Üst Yarı — Dikkat", "#a2701d"
        else:
            bb_label, bb_color = "Üst Banda Yakın — Pahalı", "#9e2b25"
        squeeze_txt = " 🔥 <b>Bant Daralması!</b> (Volatilite patlama sinyali)" if t.bb_squeeze else ""
        st.markdown(
            f"Alt: **{t.bb_lower:.2f}** &nbsp;|&nbsp; Orta: **{t.bb_middle:.2f}** "
            f"&nbsp;|&nbsp; Üst: **{t.bb_upper:.2f}**  \n"
            f"Pozisyon: <b style='color:{bb_color}'>{bb_pos_pct}%</b> — {bb_label}{squeeze_txt}",
            unsafe_allow_html=True,
        )

    with tab_vol:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hacim Analizi**")
            st.markdown(
                _signal_badge(t.volume_breakout, "Hacimli Kırılım!", "Normal Hacim",
                              "#1d6f4e", "#6b6357"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**OBV (On-Balance Volume)**")
            obv_colors = {"yukari": "#1d6f4e", "asagi": "#9e2b25", "notr": "#6b6357"}
            obv_labels = {"yukari": "Yükselen — Birikim", "asagi": "Düşen — Dağıtım", "notr": "Yatay"}
            obv_c = obv_colors.get(t.obv_trend, "#6b6357")
            obv_l = obv_labels.get(t.obv_trend, "Yatay")
            st.markdown(
                f"OBV Trendi: <b style='color:{obv_c}'>{obv_l}</b>",
                unsafe_allow_html=True,
            )
            if t.obv_divergence:
                st.success("OBV Pozitif Iraksama: Fiyat düşerken OBV yükseliyor — Gizli Birikim Sinyali!")

    with tab_risk_detail:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ATR — Oynaklık (14 gün)**")
            atr_label = "Yüksek Volatilite" if t.atr_pct > 3 else ("Orta Volatilite" if t.atr_pct > 1.5 else "Düşük Volatilite")
            atr_color = "#9e2b25" if t.atr_pct > 3 else ("#a2701d" if t.atr_pct > 1.5 else "#1d6f4e")
            st.markdown(
                f"ATR: **{t.atr:.2f} TL** &nbsp;|&nbsp; "
                f"ATR%: <b style='color:{atr_color}'>{t.atr_pct:.2f}%</b> — {atr_label}  \n"
                f"<span style='color:#6b6357;font-size:12px'>Günlük beklenen hareket aralığı ±{t.atr:.2f} TL</span>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**52 Haftalık Pozisyon**")
            w52_pct = round(t.week52_position * 100)
            if t.week52_position <= 0.20:
                w52_label, w52_color = "Yıllık Dibe Yakın — İdeal Alım Bölgesi", "#1d6f4e"
            elif t.week52_position <= 0.50:
                w52_label, w52_color = "Alt Yarı — Makul", "#3e8e6c"
            elif t.week52_position <= 0.80:
                w52_label, w52_color = "Üst Yarı — Dikkatli Ol", "#a2701d"
            else:
                w52_label, w52_color = "Yıllık Zirveye Yakın — Yüksek Risk", "#9e2b25"
            st.markdown(
                f"52H Düşük: **{t.week52_low:.2f} TL** &nbsp;|&nbsp; 52H Yüksek: **{t.week52_high:.2f} TL**  \n"
                f"Pozisyon: <b style='color:{w52_color}'>{w52_pct}%</b> — {w52_label}",
                unsafe_allow_html=True,
            )

    # Phase 2 Tabs
    if FAZ2_AVAILABLE:
        st.markdown("---")
        tab_kap, tab_target, tab_wl = st.tabs([
            "KAP Disclosures",
            "Analyst Targets",
            "Watchlist",
        ])

        with tab_kap:
            st.markdown("#### Recent KAP Disclosures (14 days)")
            wl = WatchlistManager()

            col_wl1, col_wl2, col_wl3 = st.columns([2, 1, 1])
            with col_wl1:
                wl_ticker = st.text_input("Add Stock", value=ticker_input,
                                          key="wl_ticker", placeholder="THYAO")
            with col_wl2:
                wl_threshold = st.number_input("Buy Threshold", min_value=50,
                                               max_value=90, value=65, key="wl_thr")
            with col_wl3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add", key="wl_add"):
                    wl.add(wl_ticker.upper(), alert_threshold=float(wl_threshold))
                    st.success(f"{wl_ticker.upper()} added.")

            for item in wl.get_all():
                ca, cb, cc = st.columns([2, 2, 1])
                with ca:
                    c = "#1d6f4e" if item.last_score >= 55 else \
                        "#9e2b25" if item.last_score < 40 else "#6b6357"
                    st.markdown(
                        f"**{item.ticker}** — "
                        f"<span style='color:{c}'>{item.last_score:.0f}/100</span>",
                        unsafe_allow_html=True,
                    )
                with cb:
                    last = item.last_checked[:10] if item.last_checked else "never"
                    st.caption(f"Threshold: {item.alert_threshold:.0f} · Last: {last}")
                with cc:
                    if st.button("Remove", key=f"del_{item.ticker}"):
                        wl.remove(item.ticker)
                        st.rerun()

            st.markdown("---")
            st.markdown("**Auto-Refresh Scheduler**")
            if "scheduler" not in st.session_state:
                st.session_state.scheduler = None

            cs1, cs2 = st.columns(2)
            with cs1:
                if st.button("Start Scheduler",
                             disabled=st.session_state.scheduler is not None):
                    svc = SchedulerService(wl)
                    svc.start()
                    st.session_state.scheduler = svc
                    st.success("Scheduler started (every 30 min).")
            with cs2:
                if st.button("Stop Scheduler",
                             disabled=st.session_state.scheduler is None):
                    if st.session_state.scheduler:
                        st.session_state.scheduler.stop()
                        st.session_state.scheduler = None
                        st.info("Scheduler stopped.")

            if st.session_state.scheduler:
                for job in st.session_state.scheduler.get_jobs():
                    st.caption(f"{job['name']} -> {job['next_run']}")

    # News Panel
    st.markdown("---")
    st.markdown("### News & Sentiment Analysis")
    news_result = getattr(score, "_news_result", None)
    if NEWS_ENGINE_AVAILABLE and news_result:
        render_news_panel(news_result)
    elif score.sentiment.headlines:
        for i, h in enumerate(score.sentiment.headlines[:6], 1):
            st.markdown(f"**{i}.** {h}")
    else:
        st.info("No news data available for this ticker.")


_TR_AYLAR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
_TR_GUNLER = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


@st.cache_data(ttl=600, show_spinner=False)
def _kokpit_macro() -> list:
    """Kokpit makro şeridi: USD/TL, EUR/TL, Altın, BIST100 (+günlük %)."""
    semboller = [("USDTRY=X", "USD/TL", "₺{:.2f}"), ("EURTRY=X", "EUR/TL", "₺{:.2f}"),
                 ("GC=F", "Altın (ons)", "${:,.0f}"), ("XU100.IS", "BIST 100", "{:,.0f}")]
    out = []
    try:
        bulk = yf.download([s for s, _, _ in semboller], period="5d", interval="1d",
                           auto_adjust=True, progress=False, group_by="ticker")
        for sym, ad, fmt in semboller:
            try:
                cl = bulk[sym]["Close"].dropna() if isinstance(bulk.columns, pd.MultiIndex) else bulk["Close"].dropna()
                if len(cl) >= 2:
                    son, onceki = float(cl.iloc[-1]), float(cl.iloc[-2])
                    out.append((ad, fmt.format(son), f"{(son/onceki-1)*100:+.2f}%"))
            except Exception:
                continue
    except Exception as exc:
        log.warning("Kokpit makro hatası: %s", exc)
    return out


def render_kokpit_page(ui_lang):
    """Piyasa Defteri — gazete manşeti tarzı tek bakış kokpiti.

    Tasarım ilkesi: 'bugün ne yapmalıyım?' sorusuna 10 saniyede cevap.
    Ağır indirme yok — rejim cache'i, tarama cache'i ve DB'den okur.
    """
    t = _theme()
    now = datetime.now()
    tarih = f"{now.day} {_TR_AYLAR[now.month-1]} {now.year}, {_TR_GUNLER[now.weekday()]}"

    # ---- MANŞET ----
    regime = compute_market_regime()
    _rejim_renk = {"Boğa": t["up"], "Nötr": t["warn"], "Ayı": t["down"]}.get(regime["regime"], t["muted"])
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"border-bottom:{t['rule']};padding-bottom:10px;margin-bottom:4px'>"
        f"<span style=\"font-family:{t['font_head']};font-size:34px;"
        f"font-weight:800;color:{t['ink']}\">Piyasa Defteri</span>"
        f"<span style='font-family:\"Source Serif 4\",serif;font-style:italic;"
        f"color:{t['muted']};font-size:13px'>{tarih}</span>"
        f"<span style='font-family:Inter,sans-serif;font-size:12px;font-weight:600;"
        f"letter-spacing:1px;text-transform:uppercase;color:{_rejim_renk}'>"
        f"● {regime['regime']} Piyasası — {regime['score']}/7</span></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-family:Inter,sans-serif;font-size:11.5px;color:{t['muted']};"
        f"margin-bottom:16px'>{' · '.join(regime['detay'])}</div>",
        unsafe_allow_html=True)

    # Makro şerit (Piyasa Özeti'nden taşındı — o sayfa emekli edildi)
    _makro = _kokpit_macro()
    if _makro:
        mc = st.columns(len(_makro))
        for _col, (_ad, _val, _chg) in zip(mc, _makro):
            _col.metric(_ad, _val, _chg)

    # ---- ALARMLAR (varsa manşetin hemen altında) ----
    try:
        alerts = PortfolioManager.unread_alerts()
        if alerts:
            st.markdown("#### 🚨 Dikkat Gerektirenler")
            for a in alerts[:5]:
                st.warning(a["mesaj"])
            st.caption("Tümü ve 'okundu' işaretleme: Portföy Yöneticisi sayfasında.")
    except Exception:
        alerts = []

    # ---- KPI ŞERİDİ: kullanıcı portföylerinin toplamı ----
    try:
        ports = [p for p in PortfolioManager.active_portfolios() if p.get("kind") != "golge"]
        noms, reels, xrels = [], [], []
        _kperfs = PortfolioManager.performances(ports)
        for p in ports:
            perf = _kperfs[p["id"]]
            if perf["gun"] >= 0:
                noms.append(perf["nominal"]); reels.append(perf["reel"]); xrels.append(perf["xu100_rel"])
        c1, c2, c3, c4 = st.columns(4)
        if noms:
            c1.metric("Portföyler (ort. nominal)", f"%{np.mean(noms):+.1f}")
            c2.metric("ENAG-Reel", f"%{np.mean(reels):+.1f}",
                      help="Enflasyondan arındırılmış gerçek getiri")
            c3.metric("XU100'e Göre", f"%{np.mean(xrels):+.1f}")
        else:
            c1.metric("Portföyler", "—")
            c2.metric("ENAG-Reel", "—")
            c3.metric("XU100'e Göre", "—")
        c4.metric("İzlenen Portföy", f"{len(ports)} + gölge")
    except Exception as exc:
        log.warning("Kokpit KPI hatası: %s", exc)

    st.markdown("---")
    col_sol, col_sag = st.columns([3, 2])

    # ---- SOL: BUGÜNÜN SİNYALLERİ (tarama cache'inden) ----
    with col_sol:
        st.markdown("### Günün Sinyalleri")
        try:
            scan = PortfolioScanner._load_cache() or []
            iyi = sorted([r for r in scan if not r.error and r.data_rows >= 100],
                         key=lambda r: r.score, reverse=True)
            if iyi:
                for r in iyi[:4] + iyi[-2:]:
                    sig, renk = _score_to_signal(r.score)
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;align-items:center;"
                        f"padding:7px 2px;border-bottom:1px solid {t['line2']};font-size:14px'>"
                        f"<span><img src='{_logo_url(r.ticker)}' width='18' "
                        f"style='vertical-align:middle;border-radius:4px;margin-right:8px'>"
                        f"<b>{r.ticker}</b> <span style='color:{t['muted']};font-size:12px'>"
                        f"{_sector_of(r.ticker)}</span></span>"
                        f"<span style='font-family:Inter,sans-serif'>₺{r.current_price:,.2f}</span>"
                        f"<span style='font-family:Inter,sans-serif;font-size:11.5px;font-weight:600;"
                        f"color:{renk}'>{sig} · {r.score:.0f}</span></div>",
                        unsafe_allow_html=True)
                st.caption("En güçlü 4 + en zayıf 2. Tam liste: BIST Listesi · Derin analiz: Hisse Analizi.")
            else:
                st.info("Henüz tarama verisi yok — Portföy Yöneticisi'nde 'Tara ve Portföy Öner' çalıştırınca burası dolar.")
        except Exception as exc:
            log.warning("Kokpit sinyal hatası: %s", exc)

    # ---- SAĞ: SİSTEM DURUMU + ENAG ----
    with col_sag:
        st.markdown("### Sistem Durumu")
        try:
            rh = _PMDB.execute(
                "SELECT date, regime, score FROM pm_regime_history ORDER BY date DESC LIMIT 1")["rows"]
            robot_txt = (f"✓ son tur {rh[0]['date']}" if rh else "henüz koşmadı")
        except Exception:
            robot_txt = "—"
        try:
            golge_n = _PMDB.execute(
                "SELECT COUNT(*) AS c FROM pm_portfolios WHERE kind='golge' AND status='aktif'")["rows"][0]["c"]
        except Exception:
            golge_n = "—"
        rates = InflationEngine.rates()
        son_ay = sorted(rates.keys())[-1] if rates else "—"
        son_enag = rates.get(son_ay, ("—",))[0]
        satirlar = [
            ("Günlük robot", robot_txt),
            ("Gölge portföy", f"{golge_n} aktif izlemede"),
            ("Depolama", "Turso bulut ✓" if _PMDB.is_cloud() else "lokal"),
            ("Son ENAG", f"{son_ay}: aylık %{son_enag}"),
            ("Skor sistemi", f"{SCORING_VERSION} (momentum, kalibre)"),
        ]
        for k, v in satirlar:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 2px;"
                f"border-bottom:1px solid {t['line2']};font-size:13.5px'>"
                f"<span style='color:{t['muted']}'>{k}</span>"
                f"<span style='font-family:Inter,sans-serif;font-size:12.5px'>{v}</span></div>",
                unsafe_allow_html=True)

        st.markdown("### Kısayollar")
        k1, k2 = st.columns(2)
        if k1.button("Portföy Yöneticisi →", use_container_width=True):
            st.session_state["nav_page"] = "Portfoy Yoneticisi"; st.rerun()
        if k2.button("Hisse Analizi →", use_container_width=True):
            st.session_state["nav_page"] = "Hisse Analizi"; st.rerun()


def render_portfolio_manager_page(ui_lang):
    """Portföy Yöneticisi — rejim analizi, vade×profil portföy önerisi, ENAG-reel performans defteri."""
    st.markdown("# " + ("Portföy Yöneticisi" if ui_lang == "TR" else "Portfolio Manager"))
    st.caption(
        "Piyasa rejimini okur, vade ve yatırımcı profiline göre portföy önerir, "
        "kaydettiğiniz portföylerin performansını nominal + ENAG-reel + XU100-göreli ölçer."
        if ui_lang == "TR" else
        "Reads market regime, proposes portfolios by horizon and investor profile, "
        "tracks saved portfolios in nominal + ENAG-real + XU100-relative terms."
    )

    # 1) PİYASA REJİMİ BANDI
    regime = compute_market_regime()
    st.markdown(
        f"<div style='background:#efe9db;border-left:6px solid {regime['color']};"
        f"border-radius:8px;padding:12px 16px;margin:8px 0'>"
        f"<b style='color:{regime['color']};font-size:18px'>Piyasa Rejimi: {regime['regime']}"
        f" ({regime['score']}/7)</b><br>"
        f"<span style='color:#6b6357;font-size:13px'>{' • '.join(regime['detay'])}</span></div>",
        unsafe_allow_html=True,
    )

    # RİSK ALARMLARI BANDI — robot veya oturum kontrolünün bulduğu uyarılar
    try:
        _alerts = PortfolioManager.unread_alerts()
        if _alerts:
            with st.container():
                st.markdown("### 🚨 " + ("Risk Alarmları" if ui_lang == "TR" else "Risk Alerts"))
                for a in _alerts:
                    st.warning(f"{a['mesaj']}  \n*{a['created_at']}*")
                if st.button("Tümünü okundu işaretle" if ui_lang == "TR" else "Mark all read",
                             key="pm_alerts_read"):
                    PortfolioManager.mark_alerts_read()
                    st.rerun()
    except Exception as exc:
        log.warning("Alarm bandı hatası: %s", exc)

    # HAFTALIK GÖLGE PORTFÖY SETİ — sistem kendi önerilerini otomatik kaydeder
    _week = datetime.now().strftime("%G-W%V")
    if st.session_state.get("pm_shadow_week") != _week:
        try:
            PortfolioManager._init_tables()
            _cnt = _PMDB.execute(
                "SELECT COUNT(*) AS c FROM pm_portfolios WHERE kind='golge' AND horizon != 'kontrol' AND name LIKE ?",
                (f"%{_week}%",))["rows"]
            _cntk = _PMDB.execute(
                "SELECT COUNT(*) AS c FROM pm_portfolios WHERE kind='golge' AND horizon = 'kontrol' AND name LIKE ?",
                (f"%{_week}%",))["rows"]
            _eksik_g = not _cnt or _cnt[0].get("c", 0) == 0
            _eksik_k = not _cntk or _cntk[0].get("c", 0) == 0
            if _eksik_g or _eksik_k:
                with st.spinner("Haftalık gölge/kontrol portföyleri oluşturuluyor (~1-2 dk)..."
                                if ui_lang == "TR" else "Creating weekly shadow portfolios..."):
                    _scan = PortfolioScanner.scan_all()
                    _n = PortfolioManager.ensure_shadow_batch(regime, _scan) if _eksik_g else 0
                    _n += PortfolioManager.ensure_control_batch(_scan) if _eksik_k else 0
                if _n:
                    st.toast(f"{_n} gölge/kontrol portföyü kaydedildi ({_week})")
            PortfolioManager.auto_archive_shadows()
            PortfolioManager.snapshot_tv_targets()  # haftalık analist hedef arşivi
        except Exception as exc:
            log.warning("Gölge portföy haftalık kontrol hatası: %s", exc)
        st.session_state["pm_shadow_week"] = _week

    st.caption(
        ("🟢 Depolama: **Turso bulut** — portföyler kalıcı, tüm cihazlardan aynı veri."
         if _PMDB.is_cloud() else
         "🟡 Depolama: **lokal SQLite** — Streamlit Cloud'da veriler uygulama uykusunda silinebilir. "
         "Kalıcılık için Turso bağlayın (TURSO_DATABASE_URL + TURSO_AUTH_TOKEN secrets).")
        if ui_lang == "TR" else
        ("🟢 Storage: **Turso cloud** — portfolios persist across devices."
         if _PMDB.is_cloud() else
         "🟡 Storage: **local SQLite** — on Streamlit Cloud data may reset on sleep.")
    )

    tab_new, tab_perf, tab_sector, tab_report, tab_enag = st.tabs([
        "Yeni Portföy Öner" if ui_lang == "TR" else "Propose New",
        "Portföylerim & Performans" if ui_lang == "TR" else "My Portfolios & Performance",
        "Sektör Rotasyonu" if ui_lang == "TR" else "Sector Rotation",
        "Aylık Rapor" if ui_lang == "TR" else "Monthly Report",
        "ENAG Verisi" if ui_lang == "TR" else "ENAG Data",
    ])

    # 2) YENİ PORTFÖY
    with tab_new:
        ptype = st.radio("Portföy Tipi" if ui_lang == "TR" else "Portfolio Type",
                         ["Vade × Profil", "Sektörel 🎯", "Trend 🚀"],
                         horizontal=True, key="pm_ptype")

        _sector_opts = [c for c in BIST_STOCKS.keys() if c != "BIST 30"]
        sel_sector = None
        if ptype == "Vade × Profil":
            c1, c2 = st.columns(2)
            with c1:
                horizon_label = st.selectbox("Vade" if ui_lang == "TR" else "Horizon",
                                             list(PortfolioManager.HORIZONS.keys()))
            with c2:
                profile = st.selectbox("Yatırımcı Profili" if ui_lang == "TR" else "Investor Profile",
                                       list(PortfolioManager.PROFILES.keys()), index=1)
            horizon = PortfolioManager.HORIZONS[horizon_label]
            _meta = _pm_meta(horizon, profile)
        elif ptype.startswith("Sektörel"):
            c1, c2 = st.columns(2)
            with c1:
                sel_sector = st.selectbox("Sektör", _sector_opts)
            with c2:
                profile = st.selectbox("Yatırımcı Profili", list(PortfolioManager.PROFILES.keys()), index=1)
            horizon = f"sektor:{sel_sector}"
            _meta = _pm_meta("sektor", profile)
        else:  # Trend
            profile = st.selectbox("Yatırımcı Profili", list(PortfolioManager.PROFILES.keys()), index=1)
            horizon = "trend"
            _meta = _pm_meta("trend", profile)

        st.markdown(
            f"<div style='background:#efe9db;border-radius:8px;padding:10px 14px;margin:6px 0'>"
            f"<img src='{_strategy_badge_uri(horizon, profile)}' width='26' style='vertical-align:middle;border-radius:7px;margin-right:8px'><b>{_meta['ad']}</b> — "
            f"<span style='color:#6b6357;font-size:13px'>{_meta['desc']}</span><br>"
            f"<span style='color:#8a8172;font-size:12px'>Profil: {PortfolioManager.PROFILES[profile]['aciklama']}</span></div>",
            unsafe_allow_html=True)

        if st.button("Tara ve Portföy Öner" if ui_lang == "TR" else "Scan & Propose",
                     type="primary", use_container_width=True):
            prog = st.progress(0, text="Taranıyor...")
            def _cb(t, i, n): prog.progress(min(i / max(n, 1), 1.0), text=f"{t} ({i}/{n})")
            scan = PortfolioScanner.scan_all(progress_cb=_cb)
            prog.empty()
            if horizon == "trend":
                picks, warning = PortfolioManager.propose_trend(scan, profile)
            elif horizon.startswith("sektor:"):
                picks, warning = PortfolioManager.propose_sector(scan, sel_sector, profile)
            else:
                picks, warning = PortfolioManager.propose(scan, horizon, profile, regime)
            st.session_state["pm_last_proposal"] = (picks, warning, horizon, profile)

        if "pm_last_proposal" in st.session_state:
            picks, warning, h_saved, p_saved = st.session_state["pm_last_proposal"]
            if warning:
                st.warning(warning)
            if not picks:
                if not warning:
                    st.info("Kriterlere uyan hisse bulunamadı — filtreler bilinçli olarak seçici. "
                            "Rejim/piyasa düzelince tekrar deneyin.")
            else:
                dfp = pd.DataFrame(picks)[["ticker", "sektor", "fiyat", "agirlik",
                                           "skor", "stop", "hedef", "gerekce"]]
                dfp.columns = ["Hisse", "Sektör", "Fiyat", "Ağırlık %",
                               "Skor", "Stop", "Hedef", "Gerekçe"]
                dfp = _with_logo_col(dfp)
                st.dataframe(dfp, use_container_width=True, hide_index=True,
                             column_config=_LOGO_COL_CFG)
                st.caption("Stop = giriş − ATR×katsayı (vadeye göre 2.5-3.5×; 2026-07 kalibrasyonuyla "
                           "genişletildi — dar stoplar gürültüye takılıyordu). Stop kesilirse çıkmak esastır. "
                           "Hedef ise otomatik satış değil, 'kâr almayı düşün' seviyesidir — momentum sürüyorsa "
                           "kazananı erken kesme.")
                _m2 = _pm_meta(h_saved, p_saved)
                pname = st.text_input("Portföy adı" if ui_lang == "TR" else "Portfolio name",
                                      value=f"{_m2['ad']} {datetime.now().strftime('%d.%m')}")
                if st.button("Portföyü Kaydet ve İzlemeye Al" if ui_lang == "TR" else "Save & Track",
                             use_container_width=True) and _guard_write():
                    pid = PortfolioManager.save_portfolio(pname, h_saved, p_saved, picks, regime["regime"])
                    del st.session_state["pm_last_proposal"]
                    st.success(f"Portföy #{pid} kaydedildi — performansı 'Portföylerim' sekmesinde izlenecek.")
                    st.rerun()

    # 3) PERFORMANS DEFTERİ
    with tab_perf:
        _all_active = PortfolioManager.active_portfolios()
        ports    = [p for p in _all_active if p.get("kind") != "golge"]
        shadows  = [p for p in _all_active if p.get("kind") == "golge"]
        # Toplu okumalar: 13 portfoy icin ~28 sorgu yerine 3 sorgu (taze veri, cache yok)
        _pids      = [p["id"] for p in _all_active]
        _navs_all  = PortfolioManager.nav_histories(_pids)
        _perfs_all = {p["id"]: PortfolioManager._perf_from_nav(p, _navs_all.get(p["id"], pd.DataFrame()))
                      for p in _all_active}
        _poss_all  = PortfolioManager.positions_all(_pids)

        # KARŞILAŞTIRMA GRAFİĞİ — tüm portföyler tek grafikte
        st.markdown("### " + ("Portföy Karşılaştırma" if ui_lang == "TR" else "Portfolio Comparison"))
        gc1, gc2 = st.columns([3, 2])
        with gc1:
            range_label = st.radio("Aralık" if ui_lang == "TR" else "Range",
                                   ["1 Hafta", "1 Ay", "3 Ay", "1 Yıl", "Tümü"],
                                   index=4, horizontal=True, key="pm_cmp_range")
        with gc2:
            golge_sec = st.radio("Gölge portföyler" if ui_lang == "TR" else "Shadows",
                                 ["Son kohort", "Tümü", "Yok"], index=0, horizontal=True,
                                 key="pm_cmp_shadow_mode",
                                 help="Her hafta 9 gölge + 2 kontrol açılır; hepsi çizilirse "
                                      "grafik okunmaz olur. 'Son kohort' yalnız en yeni haftayı "
                                      "gösterir — eski kohortlar arkada izlenmeye devam eder "
                                      "(Kombinasyon Karnesi hepsini toplar).")
        _range_days = {"1 Hafta": 7, "1 Ay": 30, "3 Ay": 90, "1 Yıl": 365, "Tümü": 9999}[range_label]
        _cutoff = (datetime.now() - timedelta(days=_range_days)).strftime("%Y-%m-%d")

        if golge_sec == "Yok":
            _chart_ports = ports
        elif golge_sec == "Tümü":
            _chart_ports = _all_active
        else:
            # Son kohort: gölge adları "... 2026-W29" ile biter — en yeni hafta etiketi
            _hafta = [p["name"].split()[-1] for p in shadows if "-W" in p["name"].split()[-1]]
            _son_h = max(_hafta) if _hafta else ""
            _chart_ports = ports + [p for p in shadows
                                    if _son_h and p["name"].split()[-1] == _son_h]
        fig_cmp = go.Figure()
        xu_series = {}   # tarih -> xu100 (benchmark için tüm portföylerden topla)
        _total_pts = 0
        # 1 Hafta görünümü: pozisyonlardan SAATLİK yeniden inşa (canlı, sık aralık);
        # diğer aralıklar: kayıtlı günlük NAV
        _saatlik = {}
        _xu_saat = pd.Series(dtype=float)
        if range_label == "1 Hafta":
            _saatlik, _xu_saat = PortfolioManager.intraday_navs(_chart_ports, _poss_all, _navs_all)
        for p in _chart_ports:
            _m = _pm_meta(p["horizon"], p["profile"])
            is_shadow = p.get("kind") == "golge"
            label = f"{_m['ikon']} {p['name']}"
            _hs = _saatlik.get(p["id"])
            if _hs is not None and len(_hs) >= 2:
                base = float(_hs.iloc[0])
                fig_cmp.add_trace(go.Scatter(
                    x=_hs.index, y=(_hs / base * 100.0).round(2), mode="lines",
                    name=label, opacity=0.55 if is_shadow else 1.0,
                    line=dict(width=1.2 if is_shadow else 2.5,
                              dash="dot" if is_shadow else "solid"),
                ))
                _total_pts += len(_hs)
                continue
            nav = _navs_all.get(p["id"], pd.DataFrame())
            nav = nav[nav["date"] >= _cutoff] if len(nav) else nav
            if nav.empty:
                continue
            base = float(nav["nav"].iloc[0])
            if base <= 0:
                continue
            rebased = nav["nav"] / base * 100.0
            fig_cmp.add_trace(go.Scatter(
                x=nav["date"], y=rebased.round(2), mode="lines+markers",
                name=label, opacity=0.55 if is_shadow else 1.0,
                line=dict(width=1.5 if is_shadow else 3,
                          dash="dot" if is_shadow else "solid"),
            ))
            _total_pts += len(nav)
            for d, x in zip(nav["date"], nav["xu100_close"]):
                if x and x > 0:
                    xu_series[d] = x
        if range_label == "1 Hafta" and len(_xu_saat) >= 2:
            _xb = float(_xu_saat.iloc[0])
            fig_cmp.add_trace(go.Scatter(
                x=_xu_saat.index, y=(_xu_saat / _xb * 100.0).round(2), mode="lines",
                name="XU100 (endeks)", line=dict(width=2, color="#8a8172", dash="dash")))
            xu_series = {}   # günlük benchmark çizgisini tekrar ekleme
        if xu_series:
            xs = sorted(xu_series.items())
            xu_base = xs[0][1]
            fig_cmp.add_trace(go.Scatter(
                x=[d for d, _ in xs], y=[round(v / xu_base * 100, 2) for _, v in xs],
                mode="lines", name="XU100 (endeks)",
                line=dict(width=2, color="#6b6357", dash="dash")))
        fig_cmp.update_layout(
            template=_theme()["plotly"], height=430,
            paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["bg"],
            yaxis_title="Değer (başlangıç = 100)",
            margin=dict(l=40, r=20, t=20, b=30),
            legend=dict(font=dict(size=10), orientation="h", y=-0.15),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        if _total_pts <= len(_chart_ports):
            st.caption("ℹ️ Grafik her gün bir nokta biriktirir — birkaç gün içinde eğriler belirginleşecek. "
                       "Düz çizgiler: gölge portföyler, kesikli gri: XU100 endeksi.")

        st.markdown("---")
        if not ports:
            st.info("Henüz kayıtlı portföy yok. 'Yeni Portföy Öner' sekmesinden oluşturun. "
                    "(Sistemin otomatik kayıtları aşağıdaki Gölge Portföyler bölümünde.)")
        for p in ports:
            perf = _perfs_all[p["id"]]
            _m = _pm_meta(p["horizon"], p["profile"])
            with st.expander(f"{_m['ad']} — {p['name']}  ({p['horizon']} / {p['profile']} / "
                             f"{p['created_at'][:10]} / başlangıç rejimi: {p.get('regime_at_start','?')})",
                             expanded=True):
                st.caption(f"**{_m['ad']}** — {_m['desc']}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Nominal", f"%{perf['nominal']:+.1f}")
                m2.metric("ENAG-Reel", f"%{perf['reel']:+.1f}",
                          help="Enflasyondan arındırılmış gerçek getiri (ENAG E-TÜFE)")
                m3.metric("XU100'e Göre", f"%{perf['xu100_rel']:+.1f}",
                          help="Endeksi yendiyseniz pozitif")
                m4.metric("Gün", perf["gun"])

                nav_df = _navs_all.get(p["id"], pd.DataFrame())
                if len(nav_df) >= 2:
                    chart = nav_df.set_index("date")[["nav"]].rename(columns={"nav": "Portföy (100 baz)"})
                    st.line_chart(chart, height=200)

                _render_position_table(_poss_all.get(p["id"], []))

                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("🔄 Rebalans Önerisi", key=f"pm_reb_{p['id']}",
                                 use_container_width=True):
                        with st.spinner("Güncel tarama ile karşılaştırılıyor..."):
                            scan = PortfolioScanner.scan_all()
                            sugg = PortfolioManager.rebalance_suggestions(p, scan)
                        if not sugg:
                            st.success("Değişiklik önerisi yok — portföy sağlıklı görünüyor.")
                        for s in sugg:
                            icon = {"ÇIKAR": "🔴", "KÂR AL": "🟡"}.get(s["tip"], "🟢")
                            st.markdown(
                                f"{icon} <img src='{_logo_url(s['ticker'])}' width='18' "
                                f"style='vertical-align:middle;border-radius:4px'> "
                                f"**{s['tip']} — {s['ticker']}**: {s['neden']}",
                                unsafe_allow_html=True)
                with rc2:
                    if st.button("📦 Arşivle (izlemeyi durdur)", key=f"pm_arc_{p['id']}",
                                 use_container_width=True) and _guard_write():
                        PortfolioManager.archive(p["id"])
                        st.rerun()

        # GÖLGE PORTFÖYLER — sistemin otomatik "ne olurdu?" kayıtları
        st.markdown("---")
        st.markdown("### " + ("Gölge Portföyler — sistemin otomatik kayıtları"
                                 if ui_lang == "TR" else "Shadow Portfolios — auto-tracked"))
        st.caption(
            "Sistem her hafta 9 kombinasyonun (3 vade × 3 profil) önerisini kendiliğinden "
            "kaydeder ve izler — sen hiçbir şey yapmasan bile. Amaç: 'sistemin önerileri "
            "gerçekte ne kazandırdı?' sorusuna tarafsız, ölçülmüş cevap."
            if ui_lang == "TR" else
            "Every week the system auto-saves its own 9 proposals (3 horizons × 3 profiles) "
            "and tracks them — unbiased proof of what the recommendations actually earned."
        )
        if not shadows:
            st.info("İlk gölge seti bu haftanın taramasıyla oluşacak." if ui_lang == "TR"
                    else "First shadow batch will be created with this week's scan.")
        else:
            srows = []
            for p in shadows:
                perf = _perfs_all[p["id"]]
                _m = _pm_meta(p["horizon"], p["profile"])
                srows.append({
                    "Rozet": _strategy_badge_uri(p["horizon"], p["profile"]),
                    "Portföy": f"{_m['ad']} — {p['name'].split(' ')[-1]}",
                    "Strateji": f"{p['horizon']}/{p['profile']}",
                    "Gün": perf["gun"],
                    "Nominal %": perf["nominal"],
                    "ENAG-Reel %": perf["reel"],
                    "XU100'e Göre %": perf["xu100_rel"],
                })
            sdf_shadow = pd.DataFrame(srows).sort_values("ENAG-Reel %", ascending=False)
            st.dataframe(sdf_shadow, use_container_width=True, hide_index=True, column_config={"Rozet": st.column_config.ImageColumn("", width="small")})

        # PORTFÖY İÇERİK GÖRÜNTÜLEYİCİ — gölge dahil her portföyün içine bak
        st.markdown("---")
        st.markdown("### " + ("Portföy İçeriği Görüntüle" if ui_lang == "TR" else "Inspect Portfolio Contents"))
        _tum = PortfolioManager.all_portfolios()   # aktif + arşiv, kullanıcı + gölge
        if _tum:
            def _plabel(p):
                _mi = _pm_meta(p["horizon"], p["profile"])
                durum = "" if p.get("status") == "aktif" else " · arşiv"
                tip = " (gölge)" if p.get("kind") == "golge" else ""
                return f"{_mi['ad']} — {p['name']}{tip}{durum}"
            _sec = st.selectbox(
                "Portföy seç" if ui_lang == "TR" else "Select portfolio",
                options=list(range(len(_tum))),
                format_func=lambda i: _plabel(_tum[i]),
                key="pm_inspect_sel",
            )
            _p = _tum[_sec]
            _pid = _p["id"]
            _pos = _poss_all.get(_pid) or PortfolioManager.positions(_pid)
            _nav = _navs_all.get(_pid)
            if _nav is None or (hasattr(_nav, "empty") and _nav.empty):
                _nav = PortfolioManager.nav_history(_pid)
            _pf = (_perfs_all.get(_pid) or PortfolioManager._perf_from_nav(_p, _nav))

            _mi = _pm_meta(_p["horizon"], _p["profile"])
            st.caption(f"**{_mi['ad']}** — {_mi['desc']}  \n"
                       f"Kuruluş: {_p['created_at']} · Başlangıç rejimi: {_p.get('regime_at_start','?')} · "
                       f"Durum: {_p.get('status','aktif')}")
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Nominal", f"%{_pf['nominal']:+.1f}")
            i2.metric("ENAG-Reel", f"%{_pf['reel']:+.1f}")
            i3.metric("XU100'e Göre", f"%{_pf['xu100_rel']:+.1f}")
            i4.metric("Gün", _pf["gun"])
            _rm = PortfolioManager.risk_metrics(_nav if _nav is not None else pd.DataFrame())
            r1, r2, r3, r4 = st.columns(4)
            _fmt = lambda v, s="": (f"{v}{s}" if v is not None else "—")
            r1.metric("Sharpe", _fmt(_rm["sharpe"]), help="Getiri/oynaklık (yıllık, rf=0). >1 iyi, >2 çok iyi")
            r2.metric("Sortino", _fmt(_rm["sortino"]), help="Sadece aşağı oynaklığa göre getiri")
            r3.metric("Max Düşüş", _fmt(_rm["maxdd"], "%"), help="Tepe değerden en derin kayıp")
            r4.metric("IR (vs XU100)", _fmt(_rm["ir"]), help="Endeks-üstü getirinin tutarlılığı")
            if _rm["sharpe"] is None:
                st.caption(f"Risk metrikleri için ≥10 günlük NAV gerekir (şu an {_rm['n_gun']}). Veri biriktikçe dolacak.")

            _render_position_table(_pos)
            if _nav is not None and len(_nav) >= 2:
                st.line_chart(_nav.set_index("date")[["nav"]]
                              .rename(columns={"nav": "Değer (100 baz)"}), height=180)

        score_df = PortfolioManager.shadow_scoreboard()
        if not score_df.empty:
            st.markdown("#### " + ("Kombinasyon Karnesi (tüm gölge geçmişi)"
                                      if ui_lang == "TR" else "Combination Scoreboard"))
            score_df = score_df.rename(columns={
                "Portfoy": "Portföy Sayısı", "Ort_Nominal": "Ort. Nominal %",
                "Ort_Reel": "Ort. ENAG-Reel %", "Ort_XU100_Rel": "Ort. XU100-Göreli %",
                "Ort_Sharpe": "Ort. Sharpe", "Ort_MaxDD": "Ort. MaxDD %",
                "Ort_Gun": "Ort. Gün"})
            st.dataframe(score_df, use_container_width=True, hide_index=True)
            st.caption("Veri biriktikçe hangi vade×profil kombinasyonunun gerçekten "
                       "kazandırdığı burada netleşecek — reel getiriye göre sıralı.")

    # 4) SEKTÖR ROTASYONU
    with tab_sector:
        st.markdown("Para hangi sektöre akıyor? Her sektörün 1 ay / 3 ay momentumu "
                    "dört çeyreğe ayrılır: **Lider** (ikisi de +), **Toparlanan** (kısa vade döndü), "
                    "**Zayıflayan** (kısa vade bozuldu), **Geride** (ikisi de −).")
        if st.button("Sektör Analizini Çalıştır" if ui_lang == "TR" else "Run Sector Analysis",
                     use_container_width=True, key="pm_sector_btn"):
            with st.spinner("Tarama verisi hazırlanıyor..."):
                scan = PortfolioScanner.scan_all()
            rows = {}
            for r in scan:
                if r.error or r.data_rows < 100:
                    continue
                sec = _sector_of(r.ticker)
                rows.setdefault(sec, []).append(r)
            sec_stats = []
            for sec, rs in rows.items():
                if len(rs) < 2:
                    continue
                sec_stats.append({
                    "Sektör": sec,
                    "Hisse": len(rs),
                    "1A Mom %": round(float(np.median([r.momentum_1m for r in rs])), 1),
                    "3A Mom %": round(float(np.median([r.momentum_3m for r in rs])), 1),
                    "Ort. Skor": round(np.mean([r.score for r in rs]), 1),
                    "SMA200 Üstü %": round(np.mean([r.price_above_sma200 for r in rs]) * 100, 0),
                })
            if sec_stats:
                st.session_state["pm_sector_stats"] = sec_stats

        if "pm_sector_stats" in st.session_state:
            sec_stats = st.session_state["pm_sector_stats"]
            sdf = pd.DataFrame(sec_stats)

            # Çeyrek (kadran) grafiği: x=3A, y=1A momentum
            # Eksen aralığı aykırı değere dayanıklı (IQR-kırpma): tek uç sektör
            # (örn. +%31 Çelik) ekseni germesin diye taşanlar kenara sabitlenir,
            # gerçek değerleri etikette/hover'da gösterilir.
            _xs = [s["3A Mom %"] for s in sec_stats]; _ys = [s["1A Mom %"] for s in sec_stats]

            def _robust_range(vals, min_pad):
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = max(q3 - q1, 1.0)
                lo = max(min(vals), q1 - 1.8 * iqr)
                hi = min(max(vals), q3 + 1.8 * iqr)
                pad = max(min_pad, (hi - lo) * 0.16)
                # 0 çizgisi (kadran sınırı) her zaman görünür kalsın
                return min(lo, 0) - pad, max(hi, 0) + pad

            _xlo, _xhi = _robust_range(_xs, 3.0)
            _ylo, _yhi = _robust_range(_ys, 2.0)

            # Etiket yerleşimi: önce yakın komşulardan DIŞA iten yön, çakışırsa
            # alternatif konumlar denenir (yaklaşık piksel modeliyle açgözlü).
            _pts = list(zip(_xs, _ys))
            _PX_W, _PX_H = 1050.0, 450.0        # yaklaşık çizim alanı
            _sx = _PX_W / ((_xhi - _xlo) or 1.0); _sy = _PX_H / ((_yhi - _ylo) or 1.0)

            def _px(p):  # veri koordinatı → yaklaşık piksel
                return (p[0] - _xlo) * _sx, (_yhi - p[1]) * _sy

            def _lbl_box(px, py, nchar, pos):
                w = nchar * 7.5 + 10; h = 18.0
                vert, _, horiz = pos.partition(" ")
                cx = px + (w/2 + 10 if horiz == "right" else -w/2 - 10 if horiz == "left" else 0)
                cy = py + (h/2 + 12 if vert == "bottom" else -h/2 - 12 if vert == "top" else 0)
                return cx - w/2, cy - h/2, cx + w/2, cy + h/2

            def _overlap_area(box, placed):
                tot = 0.0
                for b in placed:
                    w = min(box[2], b[2]) - max(box[0], b[0])
                    h = min(box[3], b[3]) - max(box[1], b[1])
                    if w > 0 and h > 0:
                        tot += w * h
                return tot

            def _pref_order(i):
                xi, yi = _pts[i]
                others = sorted((((p[0]-xi)*_sx)**2 + ((p[1]-yi)*_sy)**2, p)
                                for j, p in enumerate(_pts) if j != i)[:3]
                cx = np.mean([p[0] for _, p in others]) if others else xi - 1
                cy = np.mean([p[1] for _, p in others]) if others else yi - 1
                dx = (xi - cx) * _sx; dy = (yi - cy) * _sy
                vert = "top" if dy >= 0 else "bottom"
                anti = "bottom" if vert == "top" else "top"
                horiz = "right" if dx >= 0 else "left"
                ahoriz = "left" if horiz == "right" else "right"
                first = (f"middle {horiz}" if abs(dx) > abs(dy) * 1.6
                         else f"{vert} {horiz}" if abs(dx) > abs(dy) * 0.5
                         else f"{vert} center")
                cands = [first, f"{vert} center", f"{vert} {horiz}", f"middle {horiz}",
                         f"{vert} {ahoriz}", f"{anti} center", f"middle {ahoriz}",
                         f"{anti} {horiz}", f"{anti} {ahoriz}"]
                return list(dict.fromkeys(cands))

            def _assign_positions(labels):
                placed, out = [], []
                for i in range(len(_pts)):
                    px, py = _px(_pts[i])
                    pos_seq = _pref_order(i)
                    # Boş konum varsa ilk tercihe en yakınını al; hepsi doluysa
                    # en az çakışanı seç (yoğun kümede kaçınılmaz)
                    best_pos, best_ov = pos_seq[0], float("inf")
                    for pos in pos_seq:
                        box = _lbl_box(px, py, len(labels[i]), pos)
                        ov = _overlap_area(box, placed)
                        if ov == 0:
                            best_pos = pos
                            break
                        if ov < best_ov:
                            best_pos, best_ov = pos, ov
                    placed.append(_lbl_box(px, py, len(labels[i]), best_pos))
                    out.append(best_pos)
                return out
            # Ön geçiş: kenara sabitleme (aykırı değer) + etiket metinleri
            _clipped, _labels = [], []
            for s in sec_stats:
                x, y = s["3A Mom %"], s["1A Mom %"]
                _cx = float(np.clip(x, _xlo + (_xhi-_xlo)*0.03, _xhi - (_xhi-_xlo)*0.03))
                _cy = float(np.clip(y, _ylo + (_yhi-_ylo)*0.03, _yhi - (_yhi-_ylo)*0.03))
                _tasti = (_cx != x) or (_cy != y)
                _lbl = s["Sektör"] + (" *" if s["Hisse"] < 4 else "")
                if _tasti:
                    _lbl += f" ({x:+.0f}/{y:+.0f})"
                _clipped.append((_cx, _cy, _tasti)); _labels.append(_lbl)
            _pts = [(c[0], c[1]) for c in _clipped]   # yerleşim, kırpılmış konuma göre
            _positions = _assign_positions(_labels)

            fig = go.Figure()
            for _si, s in enumerate(sec_stats):
                x, y = s["3A Mom %"], s["1A Mom %"]
                if x >= 0 and y >= 0:   color, quad = "#1d6f4e", "Lider"
                elif x < 0 and y >= 0:  color, quad = "#27509e", "Toparlanan"
                elif x >= 0 and y < 0:  color, quad = "#a2701d", "Zayıflayan"
                else:                   color, quad = "#9e2b25", "Geride"
                _az_ornek = s["Hisse"] < 4   # 3 hisselik sektor ortalamasi guvensiz
                _cx, _cy, _tasti = _clipped[_si]
                fig.add_trace(go.Scatter(
                    x=[_cx], y=[_cy], mode="markers+text",
                    marker=dict(size=min(9 + s["Hisse"] * 1.2, 24), color=color,
                                opacity=0.35 if _az_ornek else 0.9,
                                symbol="diamond" if _tasti else "circle",
                                line=dict(width=1, color=_theme()["line"])),
                    text=[_labels[_si]],
                    textposition=_positions[_si],
                    textfont=dict(size=10.5, color=_theme()["muted" ] if _az_ornek else _theme()["ink"]),
                    name=quad, showlegend=False,
                    hovertemplate=f"{s['Sektör']}<br>3A: %{x:+.1f}<br>1A: %{y:+.1f}<br>"
                                  f"Skor: {s['Ort. Skor']}<extra></extra>",
                ))
            fig.add_hline(y=0, line_color="#b8ae9a", line_width=1)
            fig.add_vline(x=0, line_color="#b8ae9a", line_width=1)
            fig.update_layout(
                template=_theme()["plotly"], height=560,
                xaxis_title="3 Aylık Momentum % (medyan)", yaxis_title="1 Aylık Momentum % (medyan)",
                xaxis_range=[_xlo, _xhi],
                yaxis_range=[_ylo, _yhi],
                paper_bgcolor=_theme()["bg"], plot_bgcolor=_theme()["bg"],
                margin=dict(l=40, r=20, t=30, b=40),
                title="Sektör Rotasyon Haritası — sağ üst köşe güçlü",
            )
            for _qx, _qy, _qt in [(0.98, 0.98, "LİDER"), (0.02, 0.98, "TOPARLANAN"),
                                   (0.98, 0.02, "ZAYIFLAYAN"), (0.02, 0.02, "GERİDE")]:
                fig.add_annotation(xref="paper", yref="paper", x=_qx, y=_qy, text=_qt,
                                   showarrow=False, font=dict(size=10, color=_theme()["muted"]),
                                   opacity=0.55)
            st.plotly_chart(fig, use_container_width=True)

            sdf = sdf.sort_values("Ort. Skor", ascending=False)
            st.dataframe(sdf, use_container_width=True, hide_index=True)
            st.caption("⚠️ Sektör büyüklükleri eşit değil (3 hisselik sektör ile 29 hisselik kıyaslanmaz) — "
                       "bu yüzden medyan kullanılır, * işaretli/soluk balonlar 4'ten az hisseli sektörlerdir; "
                       "onların konumunu tek bir hisse belirlemiş olabilir, temkinli oku. "
                       "◆ elmas işaretli sektörler harita aralığının dışına taşan uç değerlerdir — "
                       "kenara sabitlenmiştir, gerçek momentum parantez içindedir.")
            best = sdf.iloc[0]
            st.info(f"💡 En güçlü sektör: **{best['Sektör']}** (ort. skor {best['Ort. Skor']}, "
                    f"hisselerin %{best['SMA200 Üstü %']:.0f}'i uzun vadeli trend üstünde). "
                    f"Portföy önerileri bu tabloyu otomatik yansıtır — skor zaten momentum bazlı.")

    # 5) AYLIK RAPOR
    with tab_report:
        st.markdown("Tüm portföylerin aylık karnesi: nominal + ENAG-reel + XU100-göreli. "
                    "Rapor indirilebilir (Markdown).")
        # Ay seçenekleri: portföylerin var olduğu aylar + son 6 ay
        month_opts = []
        cur = datetime.now()
        for i in range(6):
            m = (cur.replace(day=1) - timedelta(days=i * 28)).strftime("%Y-%m")
            if m not in month_opts:
                month_opts.append(m)
        sel_month = st.selectbox("Rapor ayı" if ui_lang == "TR" else "Report month", month_opts)

        if st.button("Raporu Oluştur" if ui_lang == "TR" else "Generate Report",
                     use_container_width=True, key="pm_report_btn"):
            m_start = f"{sel_month}-01"
            next_m  = (datetime.strptime(m_start, "%Y-%m-%d") + timedelta(days=32)).replace(day=1)
            m_end   = next_m.strftime("%Y-%m-%d")

            lines = [f"# Aylık Portföy Raporu — {sel_month}",
                     f"*Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]

            # ENAG ayı
            enag_m = InflationEngine.rates().get(sel_month)
            if enag_m:
                lines.append(f"**ENAG aylık enflasyon:** %{enag_m[0]} ({enag_m[1]})")
            lines.append("")

            # Portföyler (aktif + arşiv, o ay NAV'ı olanlar)
            ports_all = PortfolioManager.all_portfolios()

            lines.append("## Portföy Performansları")
            lines.append("")
            lines.append("| Portföy | Vade/Profil | Ay içi Nominal % | Ay içi ENAG-Reel % | XU100'e Göre % |")
            lines.append("|---|---|---|---|---|")
            any_row = False
            for p in ports_all:
                nav = PortfolioManager.nav_history(p["id"])
                nav_m = nav[(nav["date"] >= m_start) & (nav["date"] < m_end)]
                if len(nav_m) < 2:
                    continue
                any_row = True
                n0, n1 = float(nav_m["nav"].iloc[0]), float(nav_m["nav"].iloc[-1])
                nom = (n1 / n0 - 1) * 100
                reel = InflationEngine.real_return(nom, nav_m["date"].iloc[0], nav_m["date"].iloc[-1])
                x0, x1 = float(nav_m["xu100_close"].iloc[0]), float(nav_m["xu100_close"].iloc[-1])
                xrel = nom - (x1 / x0 - 1) * 100 if x0 > 0 else 0.0
                lines.append(f"| {p['name']} | {p['horizon']}/{p['profile']} "
                             f"| %{nom:+.1f} | %{reel:+.1f} | %{xrel:+.1f} |")
            if not any_row:
                lines.append("| *(bu ay NAV verisi olan portföy yok)* | | | | |")
            lines.append("")

            # Sinyal istatistikleri (o ay üretilen sinyaller)
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.row_factory = sqlite3.Row
                    sigs = [dict(r) for r in conn.execute(
                        "SELECT signal, result_7d, result_30d, score_version FROM accuracy_validation "
                        "WHERE signal_date >= ? AND signal_date < ?", (m_start, m_end)).fetchall()]
                if sigs:
                    lines.append("## Sinyal Karnesi")
                    lines.append(f"- Üretilen sinyal: **{len(sigs)}**")
                    for col, lbl in [("result_7d", "7 gün"), ("result_30d", "30 gün")]:
                        done = [s for s in sigs if s[col] in ("Basarili", "Basarisiz")]
                        if done:
                            ok = sum(1 for s in done if s[col] == "Basarili")
                            lines.append(f"- {lbl} isabet: **%{ok/len(done)*100:.0f}** ({ok}/{len(done)})")
                    v2n = sum(1 for s in sigs if s.get("score_version") == "v2")
                    lines.append(f"- v2 (momentum) sinyal oranı: {v2n}/{len(sigs)}")
                    lines.append("")
            except Exception:
                pass

            lines.append("---")
            lines.append("*Bu rapor yatırım tavsiyesi değildir. ENAG-reel getiri, ENAG E-TÜFE "
                         "aylık verisiyle gün-oranlı hesaplanır.*")
            report_md = "\n".join(lines)
            st.session_state["pm_last_report"] = (sel_month, report_md)

        if "pm_last_report" in st.session_state:
            rep_month, rep_md = st.session_state["pm_last_report"]
            st.markdown("---")
            st.markdown(rep_md)
            st.download_button(
                "📥 Raporu İndir (.md)" if ui_lang == "TR" else "📥 Download report (.md)",
                data=rep_md.encode("utf-8"),
                file_name=f"portfoy_raporu_{rep_month}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # 6) ENAG VERİ YÖNETİMİ
    with tab_enag:
        st.markdown("ENAG E-TÜFE aylık oranları. Yeni ay açıklandığında buradan ekleyin — "
                    "reel getiri hesapları bu tabloyu kullanır.")
        rates = InflationEngine.rates()
        df_r = pd.DataFrame([{"Ay": k, "Aylık %": v[0], "Kaynak": v[1]} for k, v in sorted(rates.items())])
        st.dataframe(df_r.tail(18), use_container_width=True, hide_index=True)
        ec1, ec2, ec3 = st.columns([2, 2, 1])
        with ec1:
            new_ym = st.text_input("Ay (YYYY-AA)", value=datetime.now().strftime("%Y-%m"))
        with ec2:
            new_pct = st.number_input("Aylık artış %", value=3.0, step=0.01, format="%.2f")
        with ec3:
            st.write("")
            if st.button("Kaydet", key="enag_save") and _guard_write():
                # Girdi doğrulama: geçerli YYYY-AA ve makul aralık (reel-getiri
                # tablosunu saçma değerlere karşı koru)
                import re as _re
                _ym = new_ym.strip()
                if not _re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", _ym):
                    st.error("Ay formatı YYYY-AA olmalı (örn. 2026-07).")
                elif not (-50.0 <= float(new_pct) <= 100.0):
                    st.error("Aylık oran -50 ile 100 arasında olmalı.")
                else:
                    InflationEngine.set_rate(_ym, float(new_pct))
                    st.success(f"{_ym}: %{new_pct} kaydedildi.")
                    st.rerun()


def check_portfolio_alerts(ui_lang):
    portfolio = _history_db.get_portfolio()
    unread_alerts = [a["ticker"] for a in _history_db.get_unread_alerts()]
    history = {r["ticker"]: r for r in _history_db.load_all()}
    
    for item in portfolio:
        tick = item["ticker"]
        if tick in unread_alerts:
            continue
            
        hist = history.get(tick)
        if hist and hist.get("current_price"):
            guncel = hist["current_price"]
            sinyal = hist["signal"]
            
            val = guncel * item["quantity"]
            cost = item["buy_price"] * item["quantity"]
            kar_pct = ((val - cost) / cost * 100) if cost > 0 else 0
            
            if kar_pct <= -10.0:
                msg = f"{tick} pozisyonunuz %{kar_pct:.1f} zararda. Stop-loss seviyenizi kontrol edin." if ui_lang == "TR" else f"{tick} position holds {kar_pct:.1f}% loss. Check stop-loss."
                _history_db.add_alert(tick, msg)
            elif kar_pct >= 20.0:
                msg = f"{tick} pozisyonunuz %{kar_pct:.1f} kârda. Hedefe ulaşıldı." if ui_lang == "TR" else f"{tick} position holds {kar_pct:.1f}% profit. Target reached."
                _history_db.add_alert(tick, msg)
            elif sinyal in ["SAT", "GUCLU SAT"]:
                msg = f"{tick} algoritma tarafından {sinyal} sinyali gördü." if ui_lang == "TR" else f"{tick} triggered {sinyal} signal."
                _history_db.add_alert(tick, msg)

def display_sidebar_alerts(ui_lang):
    _history_db.evaluate_accuracy()
    check_portfolio_alerts(ui_lang)
    unread = _history_db.get_unread_alerts()
    
    if unread:
        for alert in unread:
            st.toast(f"BİLDİRİM: {alert['message']}")
            
        st.markdown("---")
        with st.expander("Aktif Bildirimler" if ui_lang == "TR" else "Active Alerts", expanded=True):
            for alert in unread:
                st.info(f"[{alert['ticker']}] {alert['message']}")
            if st.button("Tümünü Okundu İşaretle" if ui_lang == "TR" else "Mark All Read", key="clear_all_alerts"):
                _history_db.mark_alerts_read([a["id"] for a in unread])
                st.rerun()

# TEMA SİSTEMİ — aynı gazete kimliği, iki baskı
# "gunduz": fildişi kâğıt (varsayılan) · "gece": gece matbaası (sıcak kömür)
THEMES = {
    "gunduz": {
        "ad": "🌞 Gazete (Aydınlık)",
        "bg": "#f7f3ea", "bg2": "#fdfbf5", "panel": "#efe9db",
        "line": "#d8d0c0", "line2": "#e4dccb",
        "ink": "#1a1712", "muted": "#6b6357",
        "up": "#1d6f4e", "down": "#9e2b25", "warn": "#a2701d",
        "accent": "#9e2b25", "navy": "#27509e", "gold": "#8a6d1d",
        "plotly": "plotly_white",
        "font_head": "'Playfair Display', Georgia, serif",
        "font_body": "'Source Serif 4', Georgia, serif",
        "rule": "3px double #1a1712",   # manşet altı çift çizgi
    },
    # Obsidyen — saf grafit, tek nane-yeşili vurgu, İsviçre minimalizmi.
    # Gazete'nin gece kopyası DEĞİL, kendi kimliği olan ayrı bir tema.
    "gece": {
        "ad": "🖤 Obsidyen (Koyu)",
        "bg": "#0e0f12", "bg2": "#15171c", "panel": "#15171c",
        "line": "#23262e", "line2": "#1c1f25",
        "ink": "#f2f3f5", "muted": "#8b93a1",
        "up": "#7ce0c3", "down": "#f2788d", "warn": "#e3c77b",
        "accent": "#7ce0c3", "navy": "#343b49", "gold": "#e3c77b",
        "plotly": "plotly_dark",
        "font_head": "Inter, sans-serif",
        "font_body": "Inter, sans-serif",
        "rule": "1px solid #23262e",    # minimal ince çizgi
    },
}


def _theme() -> dict:
    """Aktif tema paleti. Seçim query param'da saklanır (?tema=gece)."""
    try:
        key = st.session_state.get("tema_sec") or st.query_params.get("tema", "gunduz")
    except Exception:
        key = "gunduz"
    return THEMES.get(key, THEMES["gunduz"])


def _inject_gazete_css():
    """'Gazete' tasarım kimliği — aktif temaya göre iki baskıdan biri.

    Başlıklar Playfair Display, gövde Source Serif 4, veri/etiket Inter.
    Gece Baskısı: config.toml aydınlık kalır, CSS katmanı koyuya boyar.
    """
    t = _theme()
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        font-family: {t['font_body']};
        background: {t['bg']}; color: {t['ink']};
    }}
    [data-testid="stHeader"] {{ background: {t['bg']}; }}
    /* Manşet: çift çizgili gazete başlığı */
    h1 {{
        font-family: {t['font_head']} !important;
        border-bottom: {t['rule']};
        padding-bottom: .35rem;
        letter-spacing: -0.01em;
        color: {t['ink']} !important;
    }}
    h2, h3, h4 {{ font-family: {t['font_head']} !important; color: {t['ink']} !important; }}
    p, li, span, label, .stMarkdown {{ color: {t['ink']}; }}

    /* Metrikler: tabular rakamlar, gazete etiketi */
    [data-testid="stMetricValue"] {{
        font-family: Inter, sans-serif; font-variant-numeric: tabular-nums; color: {t['ink']};
    }}
    [data-testid="stMetricLabel"] {{
        font-family: Inter, sans-serif; text-transform: uppercase;
        letter-spacing: .08em; font-size: .72rem; color: {t['muted']};
    }}

    /* Kenar çubuğu ve paneller */
    [data-testid="stSidebar"] {{ background: {t['panel']}; border-right: 1px solid {t['line']}; }}
    [data-testid="stSidebar"] * {{ color: {t['ink']}; }}
    div[data-testid="stExpander"] {{
        border: 1px solid {t['line']} !important; border-radius: 4px; background: {t['bg2']};
    }}
    div[data-testid="stExpander"] summary {{ font-family: Inter, sans-serif; color: {t['ink']}; }}

    /* Girdi bileşenleri */
    [data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
        background: {t['bg2']} !important; color: {t['ink']} !important;
        border-color: {t['line']} !important;
    }}
    [data-baseweb="popover"] li {{ background: {t['bg2']}; color: {t['ink']}; }}

    /* Tablolar ve butonlar: Inter, ince çerçeve */
    div[data-testid="stDataFrame"] {{ border: 1px solid {t['line']}; }}
    div[data-testid="stDataFrame"] * {{ font-family: Inter, sans-serif; }}
    .stButton>button, .stDownloadButton>button {{
        font-family: Inter, sans-serif; border: 1px solid {t['ink']};
        border-radius: 3px; background: {t['bg']}; color: {t['ink']};
    }}
    .stButton>button:hover {{ background: {t['ink']}; color: {t['bg']}; border-color: {t['ink']}; }}
    .stButton>button[kind="primary"], .stButton>button[data-testid="baseButton-primary"] {{
        background: {t['accent']}; color: #fdfbf5; border-color: {t['accent']};
    }}
    .stTabs [data-baseweb="tab"] {{ font-family: Inter, sans-serif; color: {t['muted']}; }}
    .stTabs [aria-selected="true"] {{ color: {t['accent']} !important; }}
    .stRadio label, .stSelectbox label, .stCheckbox label, .stTextInput label,
    .stNumberInput label {{ font-family: Inter, sans-serif; }}
    hr {{ border-color: {t['line']} !important; }}
    [data-testid="stCaptionContainer"], .stCaption, small {{ color: {t['muted']} !important; }}
    [data-testid="stAlert"] {{ background: {t['panel']}; color: {t['ink']}; }}
    </style>
    """, unsafe_allow_html=True)


def run_app():
    st.set_page_config(
        page_title="Piyasa Defteri — BIST Akıllı Yatırım",
        page_icon="📰", layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_gazete_css()
    # Plotly varsayılan şablonu temaya bağla (eksen/yazı renkleri otomatik uyar)
    try:
        import plotly.io as _pio
        _pio.templates.default = _theme()["plotly"]
    except Exception:
        pass

    # Uygulama açılışında bekleyen sinyalleri kontrol et — GÜNDE 1 KEZ yeterli:
    # sinyal doğrulaması 1/3/7/14/30 GÜN çözünürlüklü, gün içinde tekrarı anlamsız.
    # (Eskiden her oturumda çalışıp ilk sayfayı 10-30 sn geciktiriyordu.)
    if "validation_checked" not in st.session_state:
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(DB_PATH) as _mc:
                _mc.execute("CREATE TABLE IF NOT EXISTS app_meta (k TEXT PRIMARY KEY, v TEXT)")
                _row = _mc.execute("SELECT v FROM app_meta WHERE k='last_signal_check'").fetchone()
            if not _row or _row[0] != today:
                _history_db.check_pending_signals()
                with sqlite3.connect(DB_PATH) as _mc:
                    _mc.execute("INSERT OR REPLACE INTO app_meta (k, v) VALUES ('last_signal_check', ?)", (today,))
                    _mc.commit()
        except Exception:
            pass
        try:
            PortfolioManager.update_navs()  # portföy değerleri her oturumda taze (hızlı: toplu)
        except Exception:
            pass
        st.session_state["validation_checked"] = True

    # TOP BAR — Market Secici (BIST vs US)
    if "selected_market" not in st.session_state:
        st.session_state["selected_market"] = "BIST"

    market_options = ["BIST (Borsa Istanbul)", "US Markets (NASDAQ/NYSE)"]
    market_icons   = ["flag", "globe"]
    _cur_market_idx = 0 if st.session_state["selected_market"] == "BIST" else 1

    if OPTION_MENU_OK:
        selected_market_label = option_menu(
            None,
            market_options,
            icons=market_icons,
            menu_icon="cast",
            default_index=_cur_market_idx,
            orientation="horizontal",
            key="market_selector",
            styles={
                "container":         {"padding": "0!important", "background-color": _theme()["panel"],
                                      "border-radius": "10px", "margin-bottom": "0.5rem"},
                "icon":              {"font-size": "18px"},
                "nav-link":          {"font-size": "15px", "text-align": "center",
                                      "margin": "0px", "color": _theme()["muted"],
                                      "--hover-color": _theme()["line"], "padding": "10px 20px"},
                "nav-link-selected": {"background-color": _theme()["navy"], "color": "#ffffff",
                                      "border-radius": "8px", "font-weight": "700"},
            },
        )
    else:
        selected_market_label = st.radio(
            "Market",
            market_options,
            index=_cur_market_idx,
            horizontal=True,
            key="market_selector_radio",
        )

    active_market = "US" if "US" in selected_market_label else "BIST"
    st.session_state["selected_market"] = active_market

    # SIDEBAR — Tema + Dil + Sayfa Navigasyonu (market'e göre)
    with st.sidebar:
        _tema_mevcut = st.session_state.get("tema_sec") or st.query_params.get("tema", "gunduz")
        tema_sec = st.selectbox(
            "Baskı / Edition",
            options=["gunduz", "gece"],
            index=0 if _tema_mevcut != "gece" else 1,
            format_func=lambda x: THEMES[x]["ad"],
            key="tema_sec_widget",
        )
        if tema_sec != _tema_mevcut:
            st.session_state["tema_sec"] = tema_sec
            try:
                st.query_params["tema"] = tema_sec  # yenilemede kaybolmasın
            except Exception:
                pass
            st.rerun()
        st.session_state["tema_sec"] = tema_sec

        ui_lang = st.selectbox(
            "Dil / Language",
            options=["TR", "EN"], index=0,
            format_func=lambda x: {"TR": "Turkce", "EN": "English"}[x],
            key="global_lang_select"
        )

        # Editör kilidi — yalnız APP_EDIT_KEY secret'ı tanımlıysa gösterilir
        # (halka açık demoda yazma korumasının kilit açma noktası)
        if _get_secret("APP_EDIT_KEY"):
            if st.session_state.get("_edit_unlocked"):
                st.caption("🔓 Editör modu açık")
                if st.button("Kilitle", key="_edit_lock_btn"):
                    st.session_state["_edit_unlocked"] = False
                    st.rerun()
            else:
                with st.expander("🔒 Editör girişi"):
                    _k = st.text_input("Editör anahtarı", type="password", key="_edit_key_in")
                    if st.button("Aç", key="_edit_unlock_btn"):
                        import hmac
                        if hmac.compare_digest(_k, _get_secret("APP_EDIT_KEY")):
                            st.session_state["_edit_unlocked"] = True
                            st.rerun()
                        else:
                            st.error("Anahtar hatalı.")
        st.markdown("---")

        if active_market == "BIST":
            # Not: "Piyasa Ozeti" ve "Sistem Portfolyleri" menüden emekli edildi
            # (Kokpit + Portföy Yöneticisi kapsıyor); render fonksiyonları duruyor.
            pages = [
                "Kokpit", "Portfoy Yoneticisi", "BIST Listesi", "Hisse Analizi",
                "Portfolyum", "Backtest", "Sinyal Takip",
            ]
            page_icons = [
                "newspaper", "wallet2", "list-ul", "search",
                "briefcase", "clock-history", "graph-up-arrow",
            ]
            _default_page = "Kokpit"
        else:  # US
            pages = [
                "US Analiz", "US Backtest",
                "US Stock List", "US Portfolios",
                "Portfolyum", "US Sinyal Takip",
            ]
            page_icons = [
                "search", "clock-history",
                "list-ul", "robot",
                "briefcase", "graph-up-arrow",
            ]
            _default_page = "US Analiz"

        _nav_redirect = False
        if "nav_page" in st.session_state:
            nav = st.session_state.pop("nav_page")
            if nav in pages:
                st.session_state["nav_radio"] = nav
                _nav_redirect = True

        if "nav_radio" not in st.session_state or st.session_state["nav_radio"] not in pages:
            st.session_state["nav_radio"] = _default_page

        default_idx = pages.index(st.session_state["nav_radio"]) if st.session_state["nav_radio"] in pages else 0

        # nav_redirect durumunda option_menu key'ini değiştirerek widget'ı resetle
        _nav_key_suffix = f"_{st.session_state['nav_radio']}" if _nav_redirect else ""

        if OPTION_MENU_OK:
            page = option_menu(
                "Menu" if ui_lang == "EN" else "Menu",
                pages,
                icons=page_icons,
                menu_icon="cast",
                default_index=default_idx,
                key=f"nav_option_menu_{active_market}{_nav_key_suffix}",
                styles={
                    "container":             {"padding": "0!important", "background-color": _theme()["panel"]},
                    "icon":                  {"color": _theme()["navy"], "font-size": "16px"},
                    "nav-link":              {"font-size": "14px", "text-align": "left",
                                              "margin": "0px", "color": _theme()["ink"],
                                              "--hover-color": _theme()["line"]},
                    "nav-link-selected":     {"background-color": _theme()["navy"], "color": "#ffffff"},
                },
            )
            st.session_state["nav_radio"] = page
        else:
            page = st.radio(
                "Menu",
                pages,
                index=default_idx,
                key=f"nav_radio_{active_market}",
            )
            st.session_state["nav_radio"] = page

        display_sidebar_alerts(ui_lang)
        st.markdown("---")
        # Global Disclaimer
        st.markdown(
            f"<div style='background:{_theme()['panel']};border:1px solid #b45309;border-radius:8px;"
            "padding:8px 12px;margin-top:8px;font-size:11px;color:#b45309;text-align:center'>"
            + (
                "⚠️ Bu uygulama <b>yatırım tavsiyesi</b> niteliğinde değildir. "
                "Tüm kararlar kullanıcının sorumluluğundadır."
                if ui_lang == "TR" else
                "⚠️ This app does <b>not</b> constitute investment advice. "
                "All decisions are the user's responsibility."
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    # PAGE ROUTING (safe wrapper — bir sayfa çökerse uygulama çökmez)
    def _safe_render(fn, *args, **kwargs):
        """Sayfa render fonksiyonunu güvenli çalıştır. Hata olursa kullanıcıya göster."""
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            log.error("Sayfa render hatası (%s): %s\n%s", fn.__name__, exc, traceback.format_exc())
            st.error(
                f"Bu sayfada beklenmeyen bir hata oluştu. Lütfen tekrar deneyin.\n\n"
                f"Hata detayı: `{type(exc).__name__}: {exc}`"
                if ui_lang == "TR" else
                f"An unexpected error occurred on this page. Please try again.\n\n"
                f"Error: `{type(exc).__name__}: {exc}`"
            )
            with st.expander("Teknik Detay / Technical Details"):
                st.code(traceback.format_exc(), language="python")

    if active_market == "BIST":
        if page == "Kokpit":
            _safe_render(render_kokpit_page, ui_lang)
        elif page == "Piyasa Ozeti":
            _safe_render(render_dashboard_page, ui_lang)
        elif page == "Portfoy Yoneticisi":
            _safe_render(render_portfolio_manager_page, ui_lang)
        elif page == "BIST Listesi":
            _safe_render(render_bist_list_page, ui_lang)
        elif page == "Hisse Analizi":
            _safe_render(render_analysis_page, ui_lang)
        elif page == "Portfolyum":
            _safe_render(render_portfolio_page, ui_lang)
        elif page == "Backtest":
            _safe_render(render_backtest_page, ui_lang)
        elif page == "Sistem Portfolyleri":
            _safe_render(render_smart_portfolio_page, ui_lang)
        elif page == "Sinyal Takip":
            _safe_render(render_validation_page, ui_lang)
    else:  # US Markets
        if page == "US Analiz":
            _safe_render(render_us_markets_page, ui_lang, mode_override="analysis")
        elif page == "US Backtest":
            _safe_render(render_us_markets_page, ui_lang, mode_override="backtest")
        elif page == "US Stock List":
            _safe_render(render_us_stock_list_page, ui_lang)
        elif page == "US Portfolios":
            _safe_render(render_us_system_portfolios_page, ui_lang)
        elif page == "Portfolyum":
            _safe_render(render_portfolio_page, ui_lang)
        elif page == "US Sinyal Takip":
            _safe_render(render_us_validation_page, ui_lang)

# ENTRY POINT

if __name__ == "__main__":
    import sys
    if "streamlit" in sys.modules or any("streamlit" in a for a in sys.argv):
        run_app()
    else:
        print("=== BIST Smart Investment Assistant — CLI Test ===")
        ticker = input("Stock code (e.g. THYAO): ").strip() or "THYAO"
        s = compute_bist_score(ticker)
        r = RiskEngine.compute(s)
        print(f"\nTicker    : {s.ticker}")
        print(f"Price     : {s.stock.current_price:.2f} TL")
        print(f"Score     : {s.total_score}/100 -> {s.signal}")
        print(f"RSI       : {s.technical.rsi:.1f}")
        print(f"Stop-Loss : {r.stop_loss_normal:.2f} TL")
        print(f"Target 1  : {r.take_profit_1:.2f} TL")
        print(f"R/R Ratio : {r.risk_reward_ratio:.2f}")