"""
BIST Smart Investment Assistant — Real News Engine v4
======================================================
Özellikler:
  ✅ Kaynak güvenilirlik puanlaması (tier sistemi)
  ✅ Resmi terim bonusu (KAP, Bilanço, İhale vb.)
  ✅ Mükerrer haber tespiti (aynı haber → güven artar)
  ✅ Spekülasyon filtresi (sosyal medya dili engellenir)
  ✅ Dil seçeneği (TR / EN)
  ✅ Sahte veri yok — veri yoksa score=50

Kaynaklar:
  Tier 1 (En Güvenilir): Bloomberg HT, NTV Ekonomi, Reuters TR, AA Ekonomi
  Tier 2 (Güvenilir):    Investing.com TR, Finans Gündem, Mynet Finans
  Tier 3 (Ek Kaynak):    Google News, Bing News, Yahoo Finance

Kurulum:
  pip install feedparser
"""

import re
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

try:
    from transformers import pipeline as hf_pipeline
    HF_OK = True
except ImportError:
    HF_OK = False

logger = logging.getLogger("bist.news")

# ─────────────────────────────────────────────────────────
# BERT TÜRKÇE SENTİMENT ANALİZCİ (Singleton)
# ─────────────────────────────────────────────────────────

class BertSentimentAnalyzer:
    """
    savasy/bert-base-turkish-sentiment-cased modeli ile Türkçe sentiment.
    Singleton pattern: model yalnızca bir kez yüklenir (~500 MB, ilk seferinde indirilir).
    transformers + torch kurulu değilse otomatik devre dışı kalır.
    """
    _instance = None
    _pipe     = None
    _ready    = False

    MODEL_ID = "savasy/bert-base-turkish-sentiment-cased"
    # Model çıktıları → bizim etiketlerimiz
    LABEL_MAP = {
        "positive": "olumlu",
        "negative": "olumsuz",
        "neutral":  "notr",
        "POSITIVE": "olumlu",
        "NEGATIVE": "olumsuz",
        "NEUTRAL":  "notr",
        "LABEL_0":  "olumsuz",  # bazı modeller 0/1/2 kullanır
        "LABEL_1":  "notr",
        "LABEL_2":  "olumlu",
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> bool:
        """Modeli yükle. Başarılıysa True döner."""
        if self._ready:
            return True
        if not HF_OK:
            logger.warning("transformers kurulu değil → pip install transformers torch")
            return False
        try:
            logger.info(f"BERT model yükleniyor: {self.MODEL_ID} (ilk seferde ~500 MB indirilir)")
            self._pipe  = hf_pipeline(
                "text-classification",
                model=self.MODEL_ID,
                top_k=1,
                truncation=True,
                max_length=128,
            )
            self._ready = True
            logger.info("BERT model hazır ✅")
            return True
        except Exception as exc:
            logger.error(f"BERT model yüklenemedi: {exc}")
            return False

    def predict(self, text: str) -> str:
        """Metni sınıflandır. 'olumlu' | 'olumsuz' | 'notr' döner."""
        label, _ = self.predict_with_confidence(text)
        return label

    def predict_with_confidence(self, text: str) -> tuple:
        """(etiket, guven_skoru) döner. Guven 0.0-1.0 arasında."""
        if not self._ready:
            return "notr", 0.0
        try:
            out   = self._pipe(text[:512])
            item  = out[0][0] if isinstance(out[0], list) else out[0]
            label = self.LABEL_MAP.get(item["label"], "notr")
            score = float(item["score"])
            return label, score
        except Exception as exc:
            logger.debug(f"BERT predict hata: {exc}")
            return "notr", 0.0


# Global BERT instance — ilk analyze_news() çağrısında yüklenir
# Streamlit ortamında @st.cache_resource ile sarmallanarak hot-reload'da
# modelin yeniden indirilmesi/yüklenmesi önlenir.
try:
    import streamlit as _st

    @_st.cache_resource(show_spinner=False)
    def _get_bert_cached() -> BertSentimentAnalyzer:
        b = BertSentimentAnalyzer()
        b.load()
        return b

    _bert = _get_bert_cached()
except Exception:
    _bert = BertSentimentAnalyzer()

import re as _re

# ─────────────────────────────────────────────────────────
# KATMAN 1 — GÜÇLÜ OVERRIDE (BERT'İ GEÇERSIZ KILAR)
# Regex kalıpları — bu ifadeler bulunursa kesin karar verilir
# ─────────────────────────────────────────────────────────

# Kesin pozitif kalıplar (yüzde X artış, rekor kâr, temettü açıkladı vb.)
_STRONG_POS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"yüzde\s*\d+[\.,]?\d*\s*(artış|büyüme|yükseliş|arttu)",
    r"%\s*\d+[\.,]?\d*\s*(artış|büyüme|yükseliş)",
    r"\d+[\.,]?\d*\s*%\s*(artış|büyüme|yükseliş)",
    r"artış\s+gösterdi",
    r"(rekor\s+)(kâr|kar|satış|ihracat|üretim|gelir)",
    r"(net\s+)?(kâr|kar)\s+(açıkladı|elde etti|kaydetti)",
    r"temettü\s+(açıkladı|dağıtacak|kararı|ödemesi)",
    r"(hedef\s+fiyat|fiyat\s+hedefi)\s+(yükseltildi|artırıldı)",
    r"(güçlü\s+al|al\s+tavsiyesi)",
    r"(kapasitesini|gücünü|üretimini)\s+(artırdı|yükseltti|genişletti)",
    r"mw[a-z']*\s*(kapasiteye|güce)\s+(ulaştı|yükseltti|çıkardı)",
    r"(devreye\s+aldı|üretime\s+geçti|faaliyete\s+açtı)",
    r"ihracat\s+(rekoru|artışı|başarısı)",
    r"(anlaşma|sözleşme|ihale)\s+(imzaladı|kazandı|aldı)",
    r"(lisans|ruhsat)\s+(aldı|kazandı)",
    r"(büyüme|genişleme)\s+(kaydetti|gerçekleştirdi)",
    r"(satışlar|gelir|ciro)\s+\d",           # Satışlar %X artış gibi
    r"(güçlü|kuvvetli)\s+(performans|sonuç|büyüme)",
    r"(tahvil|bono)\s+(başarıyla|üstü talep)",
    r"(borsa|hisse)\s+değer",
    # İngilizce (EN haberleri için)
    r"(record\s+)?(profit|revenue|earnings|sales)\s+(surge|jump|soar|rise)",
    r"(raised?|raised)\s+(target|price\s+target)",
    r"(strong\s+buy|outperform|overweight)",
    r"\d+%\s+(increase|growth|rise|gain)",
    r"(capacity|power)\s+(increase|expansion|growth)",
    r"(new\s+)?(contract|deal|agreement)\s+(signed|won|awarded)",
    r"beat\s+(estimate|expectation)",
]]

# Kesin negatif kalıplar
_STRONG_NEG_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"zarar\s+(açıkladı|bildirdi|yazdı|etti)",
    r"(net\s+)?zarar\s+\d",
    r"iflas\s+(başvurusu|ilan)",
    r"konkordato\s+(ilan|başvuru)",
    r"(kredi\s+notu)\s+(düşürüldü|indirildi|düşürüldü)",
    r"(görevden\s+al|istifa\s+etti)",
    r"(büyük|ağır)\s+(kayıp|zarar|düşüş)",
    r"yüzde\s*\d+[\.,]?\d*\s*(gerileme|düşüş|azalma|kayıp)",
    r"%\s*\d+[\.,]?\d*\s*(gerileme|düşüş|azalma)",
    r"\d+[\.,]?\d*\s*%\s*(gerileme|düşüş|azalma)",
    r"(gelir|satış|ihracat|ciro)\s+(geriledi|düştü|azaldı)",
    r"(beklentinin|tahminlerin)\s+altında\s+kaldı",
    r"(hayal\s+kırıklığı|hayal\s+kırıklığı\s+yarattı)",
    r"(uyarı|uyarı\s+verdi|riskler?\s+artıyor)",
    r"(dava|soruşturma|ceza)\s+(açıldı|uygulandı|kesildi)",
    r"temerrüt",
    r"(üretim\s+durdu|fabrika\s+kapandı)",
    # İngilizce
    r"(net\s+)?loss\s+of\s+\$?\d",
    r"(bankruptcy|insolvency|default)",
    r"(credit\s+rating)\s+(downgraded|cut|lowered)",
    r"(missed?|missed)\s+(estimate|expectation|target)",
    r"\d+%\s+(decline|drop|fall|decrease|loss)",
    r"(profit\s+warning|earnings\s+warning)",
]]

# ─────────────────────────────────────────────────────────
# KATMAN 2 — ANAHTAR KELİME LİSTELERİ (BERT belirsizse)
# ─────────────────────────────────────────────────────────

FIN_POSITIVE = [
    # Büyüme / artış
    "artış", "artırdı", "artıyor", "arttı", "yükseliş", "yükseldi",
    "yükselt", "büyüme", "büyüdü", "büyüyor", "rekor", "zıpladı", "sıçradı",
    "tarihi yüksek", "en yüksek", "güçlü", "kuvvetli", "başarılı",
    # Kazanç / finansal
    "kar", "kâr", "temettü", "net kar", "net kâr", "faaliyet karı",
    "gelir artışı", "ciro artışı", "ihracat artışı", "satış artışı",
    "kar payı", "pozitif", "olumlu", "iyileşme",
    # Enerji sektörü
    "kurulu güç", "kapasite artışı", "devreye aldı", "mw", "yenilenebilir",
    "santral", "rüzgar", "güneş enerjisi", "lisans aldı", "türbin",
    "enerji üretimi", "elektrik üretimi",
    # Yatırım / anlaşma
    "yatırım", "anlaşma", "sözleşme", "ihale kazandı", "sözleşme imzaladı",
    "hedef yükseltildi", "hedef artırıldı", "tavsiye yinelendi",
    "kredi notu artırıldı", "kredi notu yükseltildi",
    # İngilizce
    "growth", "increase", "profit", "revenue", "record", "gain",
    "strong", "beat", "outperform", "upgrade", "buy", "dividend",
    # English financial (expanded)
    "earnings beat", "revenue growth", "raised guidance", "price target raised",
    "analyst upgrade", "bullish", "breakout", "all-time high", "ath",
    "market cap", "cash flow", "buyback", "share repurchase", "guidance raised",
    "exceeds expectations", "better than expected", "blowout quarter",
    "margin expansion", "double upgrade", "overweight", "accumulate",
]

FIN_NEGATIVE = [
    # Kayıp / düşüş
    "zarar", "kayıp", "düşüş", "azaldı", "geriledi", "düştü", "azalma",
    "beklentinin altında", "hayal kırıklığı", "uyarı", "risk",
    "gerileme", "zayıflama", "kötüleşme", "negatif", "olumsuz",
    # Borç / iflас
    "borç", "iflas", "konkordato", "temerrüt", "temerrut",
    "kredi notu düşürüldü", "kredi notu indirildi",
    # Yönetim
    "istifa", "görevden ayrıldı", "görevden alındı", "ayrıldı",
    # Regülasyon
    "ceza", "soruşturma", "dava", "suçlama", "yaptırım",
    # İngilizce
    "loss", "decline", "drop", "fall", "miss", "warn", "risk",
    "downgrade", "sell", "underperform", "bankruptcy", "default",
    # English financial (expanded)
    "earnings miss", "revenue decline", "lowered guidance", "price target cut",
    "analyst downgrade", "bearish", "breakdown", "52-week low",
    "debt concern", "cash burn", "dilution", "share offering", "guidance cut",
    "below expectations", "worse than expected", "disappointing quarter",
    "margin compression", "double downgrade", "underweight", "reduce",
]


def _check_strong_patterns(text: str):
    """
    Kesin pozitif/negatif regex kalıplarını kontrol et.
    Dönüş: 'olumlu' | 'olumsuz' | None (emin değil)
    """
    for pat in _STRONG_POS_PATTERNS:
        if pat.search(text):
            return "olumlu"
    for pat in _STRONG_NEG_PATTERNS:
        if pat.search(text):
            return "olumsuz"
    return None


def _classify_financial_keywords(text: str) -> str:
    """Anahtar kelime tabanlı finans sentiment (güçlü override sonrası devreye girer)."""
    tl = text.lower()
    pos = sum(1 for w in FIN_POSITIVE if w in tl)
    neg = sum(1 for w in FIN_NEGATIVE if w in tl)
    if pos > neg:   return "olumlu"
    if neg > pos:   return "olumsuz"
    return "notr"
# ─────────────────────────────────────────────────────────
# Tier 1: Kurumsal, editöryal denetimli, finans odaklı
# Tier 2: Güvenilir finans/ekonomi siteleri
# Tier 3: Arama motorları, genel haberler

SOURCE_TIERS = {
    # Tier 1 — En yüksek güvenilirlik (ağırlık: 3x)
    # Not: AA Ekonomi ve KAP RSS'leri test'te erişilemez bulundu (404/bağlantı hatası)
    "Bloomberg HT":     {"tier": 1, "weight": 3.0, "label": "Grade A"},
    "NTV Ekonomi":      {"tier": 1, "weight": 2.5, "label": "Grade A"},
    "Habertürk":        {"tier": 1, "weight": 2.5, "label": "Grade A"},
    # Tier 2 — Güvenilir (ağırlık: 2x)
    "Hürriyet":         {"tier": 2, "weight": 2.0, "label": "Grade B"},
    "Milliyet":         {"tier": 2, "weight": 2.0, "label": "Grade B"},
    "Sabah Ekonomi":    {"tier": 2, "weight": 2.0, "label": "Grade B"},
    "Dünya Gazetesi":   {"tier": 2, "weight": 2.2, "label": "Grade B"},
    "Investing.com TR": {"tier": 2, "weight": 2.0, "label": "Grade B"},
    "Finans Gündem":    {"tier": 2, "weight": 2.0, "label": "Grade B"},
    "Ekonomim":         {"tier": 2, "weight": 2.0, "label": "Grade B"},
    "Para Analiz":      {"tier": 2, "weight": 1.8, "label": "Grade B"},
    "Foreks":           {"tier": 2, "weight": 1.8, "label": "Grade B"},
    "CNN Türk Ekonomi": {"tier": 2, "weight": 1.9, "label": "Grade B"},
    # Tier 3 — Ek kaynak (ağırlık: 1x)
    "Google News TR":   {"tier": 3, "weight": 1.0, "label": "Grade C"},
    "Google News EN":   {"tier": 3, "weight": 1.0, "label": "Grade C"},
    "Bing News":        {"tier": 3, "weight": 0.8, "label": "Grade C"},
    "Yahoo Finance":    {"tier": 3, "weight": 0.8, "label": "Grade C"},
    "Yahoo Finance RSS":  {"tier": 2, "weight": 2.0},
    "Google News US":     {"tier": 3, "weight": 1.0},
    "Bing News US":       {"tier": 3, "weight": 0.8},
    "Yahoo Finance US":   {"tier": 2, "weight": 2.2},
}

def _get_weight(source: str) -> float:
    return SOURCE_TIERS.get(source, {}).get("weight", 1.0)

def _get_tier(source: str) -> int:
    return SOURCE_TIERS.get(source, {}).get("tier", 3)

def _get_tier_label(source: str) -> str:
    return SOURCE_TIERS.get(source, {}).get("label", "Grade C")

# ─────────────────────────────────────────────────────────
# SPEKÜLASYON FİLTRESİ
# ─────────────────────────────────────────────────────────
# Bu kelimeler geçen haberler spekülatif sayılır, skordan düşülür

SPECULATION_WORDS = [
    "tahminim", "sanırım", "saniyorum", "bence", "bana göre",
    "duydum ki", "kulağıma geldi", "söylentisi", "söylenti",
    "iddia", "iddiaları", "asılsız", "doğrulanmadı", "kaynaksız",
    "twitter", "tweet", "sosyal medya paylaşımı", "forum",
    "ekşi sözlük", "şikayetvar", "reddit",
    "pump", "dump", "moon", "rocket", "100x",
    "kesin çıkar", "kesinlikle alın", "hemen al", "son fırsat",
    "garantili", "garanti kazanç", "batmaz",
    "manipülasyon", "manipülasyon var", "yapay yükseliş",
]

# ─────────────────────────────────────────────────────────
# RESMİ TERİM BONUSU — güven skoru artırıcılar
# ─────────────────────────────────────────────────────────

OFFICIAL_TERMS = {
    # KAP ve yasal bildirimler (+15 puan bonus)
    "kap bildirimi":       15,
    "özel durum açıklaması": 15,
    "ozel durum aciklamasi": 15,
    "kamuyu aydınlatma":   15,
    "kamuyu aydinlatma":   15,
    "***":                 15,  # KAP başlığı formatı

    # Finansal raporlar (+12 puan)
    "bilanço":             12,
    "bilanco":             12,
    "finansal sonuçlar":   12,
    "finansal sonuclar":   12,
    "faaliyet raporu":     12,
    "mali tablo":          12,
    "yıllık rapor":        12,
    "çeyrek sonuçları":    12,
    "ceyrek sonuclari":    12,

    # Kurumsal eylemler (+10 puan)
    "pay geri alımı":      10,
    "pay geri alimi":      10,
    "sermaye artırımı":    10,
    "sermaye artirimi":    10,
    "temettü dağıtımı":    10,
    "temettu dagitimi":    10,
    "genel kurul kararı":  10,
    "ihale sonucu":        10,
    "sözleşme imzalandı":  10,
    "sozlesme imzalandi":  10,
    "birleşme anlaşması":  10,
    "birlasma anlasması":  10,
    "halka arz":           10,

    # Yönetim değişiklikleri (+8 puan)
    "yönetim kurulu kararı": 8,
    "yonetim kurulu karari":  8,
    "ceo değişikliği":       8,
    "genel müdür":           8,

    # Analist raporları (+6 puan)
    "hedef fiyat":         6,
    "analist raporu":      6,
    "kredi notu":          6,
    "rating":              6,
    "fiyat hedefi":        6,
}

def _official_bonus(title: str) -> int:
    """Başlıkta resmi terim varsa güven bonusu döner."""
    tl    = title.lower()
    bonus = 0
    for term, pts in OFFICIAL_TERMS.items():
        if term in tl:
            bonus = max(bonus, pts)  # En yüksek bonus
    return bonus

# ─────────────────────────────────────────────────────────
# SENTIMENT KELİME BANKASI
# ─────────────────────────────────────────────────────────

POSITIVE_WORDS = [
    "kar", "kâr", "kazanc", "kazanç", "rekor", "buyume", "büyüme",
    "artis", "artış", "yukselis", "yükseliş", "guclu", "güçlü",
    "basari", "başarı", "olumlu", "pozitif", "iyilesme", "iyileşme",
    "toparlanma", "temettu", "temettü", "yatirim", "yatırım",
    "ihracat", "anlasma", "anlaşma", "sozlesme", "sözleşme",
    "ortaklik", "ortaklık", "genisleme", "genişleme", "kapasite",
    "ihale kazandi", "kazandi", "kazandı", "asti", "aştı",
    "yukari", "yukarı", "ralli", "degerlenme", "değerlenme",
    "buyuyor", "büyüyor", "artti", "arttı", "bedelsiz",
    "sermaye artirimi", "sermaye artırımı", "kar payi", "kâr payı",
    "al sinyali", "destek buldu", "yukselebilir", "yükselebilir",
    "hedef fiyat yukari", "rating artirimi", "not artirimi",
    "sözleşme imzalandı", "ihale kazanıldı", "yeni sipariş",
]

NEGATIVE_WORDS = [
    "zarar", "dusus", "düşüş", "kayip", "kayıp", "azalis", "azalış",
    "gerileme", "zayif", "zayıf", "olumsuz", "negatif", "kotu", "kötü",
    "kriz", "dava", "ceza", "sorusturma", "soruşturma", "iflas",
    "iptal", "erteleme", "tasfiye", "baski", "baskı",
    "belirsizlik", "daralma", "geriledi", "cekilme", "çekilme",
    "uyari", "uyarı", "deger kaybi", "değer kaybı", "kaybetti",
    "sat sinyali", "satis baskisi", "satış baskısı", "destek kirdi",
    "dusus trendi", "düşüş trendi", "dusebilir", "düşebilir",
    "hedef fiyat asagi", "rating indirimi", "not indirimi",
    "sozlesme fesih", "sözleşme fesih", "iptal edildi",
]

# ─────────────────────────────────────────────────────────
# ŞİRKET VE DİL HARİTALARI
# ─────────────────────────────────────────────────────────

TICKER_TO_TR = {
    "THYAO": "Türk Hava Yolları", "GARAN": "Garanti Bankası",
    "AKBNK": "Akbank",            "ISCTR": "İş Bankası",
    "YKBNK": "Yapı Kredi",        "SISE":  "Şişecam",
    "KCHOL": "Koç Holding",       "SAHOL": "Sabancı Holding",
    "TCELL": "Turkcell",          "BIMAS": "BİM Mağazalar",
    "ASELS": "Aselsan",           "FROTO": "Ford Otosan",
    "TOASO": "Tofaş",             "TUPRS": "Tüpraş",
    "ENKAI": "Enka İnşaat",       "EKGYO": "Emlak Konut",
    "KOZAA": "Koza Anadolu",      "KOZAL": "Koza Altın",
    "PETKM": "Petkim",            "EREGL": "Ereğli Demir Çelik",
    "ARCLK": "Arçelik",           "VESTL": "Vestel",
    "MGROS": "Migros",            "SOKM":  "Şok Marketler",
    "DOHOL": "Doğan Holding",     "HALKB": "Halkbank",
    "VAKBN": "Vakıfbank",         "PGSUS": "Pegasus",
    "TAVHL": "TAV Havalimanları", "ULKER": "Ülker",
}

TICKER_TO_EN = {
    "THYAO": "Turkish Airlines",  "GARAN": "Garanti BBVA",
    "AKBNK": "Akbank",            "ISCTR": "Is Bankasi",
    "YKBNK": "Yapi Kredi",        "SISE":  "Sisecam",
    "KCHOL": "Koc Holding",       "TCELL": "Turkcell",
    "BIMAS": "BIM stores",        "ASELS": "Aselsan defense",
    "FROTO": "Ford Otosan",       "TOASO": "Tofas",
    "TUPRS": "Tupras refinery",   "EREGL": "Erdemir steel",
    "ARCLK": "Arcelik",           "PGSUS": "Pegasus Airlines",
    "TAVHL": "TAV Airports",      "VESTL": "Vestel electronics",
    "MGROS": "Migros retail",     "PETKM": "Petkim petrochemical",
}

# ─────────────────────────────────────────────────────────
# HISSE TAKMAİSİM / ALIAS HARİTASI
# Başlık eşleşmesini genişletir: ticker kodu veya tam ad eşleşmezse
# bu listeden biri başlıkta geçiyorsa haber "ilgili" sayılır.
# ─────────────────────────────────────────────────────────

TICKER_ALIASES: dict[str, list[str]] = {
    "THYAO": ["thy", "türk hava", "türk havayol", "turkish airlines", "thao"],
    "GARAN": ["garanti", "garanti bbva", "garanti bankası", "garanti bank"],
    "AKBNK": ["akbank"],
    "ISCTR": ["iş bankası", "işbank", "is bankasi", "isbank"],
    "YKBNK": ["yapı kredi", "yapi kredi", "ykb"],
    "SISE":  ["şişecam", "sisecam", "şişe cam"],
    "KCHOL": ["koç holding", "koc holding", "koç"],
    "SAHOL": ["sabancı holding", "sabanci holding", "sabancı"],
    "TCELL": ["turkcell"],
    "BIMAS": ["bim", "bim mağazaları", "bim magazalari"],
    "ASELS": ["aselsan"],
    "FROTO": ["ford otosan", "ford"],
    "TOASO": ["tofaş", "tofas"],
    "TUPRS": ["tüpraş", "tupras", "tüpraş rafin"],
    "ENKAI": ["enka", "enka inşaat"],
    "EKGYO": ["emlak konut", "emlak gyo"],
    "KOZAA": ["koza anadolu", "koza"],
    "KOZAL": ["koza altın", "koza altin"],
    "PETKM": ["petkim"],
    "EREGL": ["ereğli", "eregli", "erdemir", "ereğli demir"],
    "ARCLK": ["arçelik", "arcelik"],
    "VESTL": ["vestel"],
    "MGROS": ["migros"],
    "SOKM":  ["şok", "sok market", "şok market"],
    "DOHOL": ["doğan holding", "dogan holding", "doğan"],
    "HALKB": ["halkbank", "halk bankası", "halk bank"],
    "VAKBN": ["vakıfbank", "vakifbank", "vakıf bank"],
    "PGSUS": ["pegasus", "pegasus hava"],
    "TAVHL": ["tav", "tav havalimanı", "tav airports"],
    "ULKER": ["ülker", "ulker"],
}

US_TICKER_TO_EN = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet Google",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "TSLA": "Tesla",
    "META": "Meta Platforms Facebook", "NFLX": "Netflix",
    "AMD": "AMD Advanced Micro Devices", "INTC": "Intel",
    "CRM": "Salesforce", "ORCL": "Oracle", "ADBE": "Adobe",
    "PYPL": "PayPal", "SQ": "Block Square", "SHOP": "Shopify",
    "UBER": "Uber", "ABNB": "Airbnb", "COIN": "Coinbase",
    "PLTR": "Palantir", "SNOW": "Snowflake", "DDOG": "Datadog",
    "BA": "Boeing", "JPM": "JPMorgan Chase", "BAC": "Bank of America",
    "GS": "Goldman Sachs", "V": "Visa", "MA": "Mastercard",
    "WMT": "Walmart", "COST": "Costco", "HD": "Home Depot",
    "DIS": "Walt Disney", "CMCSA": "Comcast", "PEP": "PepsiCo",
    "KO": "Coca-Cola", "JNJ": "Johnson Johnson", "PFE": "Pfizer",
    "UNH": "UnitedHealth", "MRK": "Merck", "ABBV": "AbbVie",
    "XOM": "Exxon Mobil", "CVX": "Chevron", "LLY": "Eli Lilly",
    "AVGO": "Broadcom", "QCOM": "Qualcomm", "MU": "Micron Technology",
    "ARM": "ARM Holdings", "SMCI": "Super Micro Computer",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8",
}

# ─────────────────────────────────────────────────────────
# VERİ SINIFLARI
# ─────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    url: str            = ""
    sentiment: str      = "notr"
    confidence: float   = 50.0   # 0-100: bu haberin güven skoru
    is_kap: bool        = False
    is_material: bool   = False
    is_speculation: bool = False
    official_bonus: int = 0
    duplicate_count: int = 1     # kaç farklı kaynakta çıktı
    tier: int           = 3


@dataclass
class NewsAnalysisResult:
    score: float          = 50.0
    positive_count: int   = 0
    negative_count: int   = 0
    neutral_count: int    = 0
    total_news: int       = 0
    filtered_count: int   = 0    # spekülasyon nedeniyle filtrelenen
    headlines: list       = field(default_factory=list)
    news_items: list      = field(default_factory=list)
    kap_disclosures: list = field(default_factory=list)
    material_events: list = field(default_factory=list)
    data_sources: list    = field(default_factory=list)
    data_quality: str     = "notr"
    language: str         = "TR"
    warning: str          = ""
    sentiment_engine: str = "kelime_sayma"  # "bert" | "kelime_sayma"


# ─────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────

def _classify_keywords(text: str) -> str:
    """Genel anahtar kelime tabanlı fallback sentiment."""
    t   = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    if pos > neg:  return "olumlu"
    if neg > pos:  return "olumsuz"
    return "notr"


BERT_CONFIDENCE_THRESHOLD = 0.70  # BERT bu güvenin altındaysa finans sözlüğü devreye girer

# ─────────────────────────────────────────────────────────
# TEKNİK FİYAT HAREKETİ FİLTRESİ
# "HEKTS yüzde 3 yükseldi" gibi nötr borsa hareketi haberleri
# Katman 1 regex'i tarafından yanlış etiketlenir. Bu fonksiyon
# bu tür başlıkları tespit ederek Katman 1'i devre dışı bırakır.
# ─────────────────────────────────────────────────────────

_TECHNICAL_MOVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    # "HEKTS yüzde X yükseldi/geriledi" — salt fiyat hareketi
    r"[A-Z]{3,6}\s+(yüzde|%)\s*[\d,\.]+\s*(yükseldi|geriledi|düştü|kazandı|kaybetti)",
    # "X hissesi %Y ile BIST'in en ...sı oldu"
    r"hissesi.{0,20}(en\s+)?(çok\s+)?(yükselen|düşen|kazandıran|kaybettiren)",
    # "BIST 100 yükseliş/düşüş" — endeks haberi
    r"\bBIST\s*-?\s*\d+\b.{0,30}(yükseli|düşüş|gerileme|artış)",
    # "hisse (yüzde|%) X ile ... kapattı" — kapanış haberi
    r"(yüzde|%)\s*[\d,\.]+\s*(ile|oranında).{0,20}(kapattı|tamamladı|seyretti)",
    # "teknik analiz" başlığı
    r"teknik\s+analiz",
]]


def _is_technical_price_move(text: str) -> bool:
    """Başlık salt borsa fiyat hareketi haberi mi? True ise Katman 1 atlanır."""
    for pat in _TECHNICAL_MOVE_PATTERNS:
        if pat.search(text):
            return True
    return False


def _classify(text: str) -> str:
    """
    Hibrit Sentiment Sınıflandırma:
    0. TEKNİK FİYAT HAREKETİ → Katman 1 atlanır (yanlış override önlenir)
    1. BERT çalışıyorsa ve güven >= %70 → BERT kararı geçerli
    2. BERT çalışıyor ama güven < %70 → Finans anahtar kelimelerini dene
       Finans sozlugu netsiz ise BERT kararini kullan
    3. BERT kurulu degilse -> Genel anahtar kelime sayma
    """
    # ── KATMAN 0: TEKNİK FİYAT HAREKETİ KONTROLÜ ──
    # Salt fiyat hareketi haberlerinde Katman 1 regex hatalı override yapar.
    # Bu haberleri direkt BERT/kelime sayıma yönlendir.
    is_tech_move = _is_technical_price_move(text)

    if not is_tech_move:
        # ── KATMAN 1: GÜÇLÜ REGEX OVERRIDE ──
        strong = _check_strong_patterns(text)
        if strong is not None:
            logger.debug(f"Güçlü override: {strong!r} | metin={text[:60]!r}")
            return strong
    else:
        logger.debug(f"Teknik fiyat hareketi → Katman 1 atlandı | metin={text[:60]!r}")

    # ── KATMAN 2: BERT (kurulu ve hazırsa) ──
    if not _bert._ready:
        return _classify_keywords(text)

    bert_label, bert_conf = _bert.predict_with_confidence(text)

    # BERT %70+ emin → BERT'e güven
    if bert_conf >= BERT_CONFIDENCE_THRESHOLD:
        logger.debug(f"BERT kararı: {bert_label} ({bert_conf:.0%})")
        return bert_label

    # ── KATMAN 3: BERT belirsiz → finans anahtar kelimeleri ──
    fin_label = _classify_financial_keywords(text)
    if fin_label != "notr":
        logger.debug(
            f"Hibrit: BERT={bert_label}({bert_conf:.0%}) belirsiz, "
            f"Finans={fin_label}"
        )
        return fin_label

    # Hiçbiri net sonuç vermedi → BERT kararını kabul et
    return bert_label

def _is_speculation(text: str) -> bool:
    tl = text.lower()
    return any(w in tl for w in SPECULATION_WORDS)

def _parse_date(s: str) -> Optional[datetime]:
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=None)
        except (ValueError, AttributeError):
            continue
    return None

def _clean_title(title: str) -> str:
    # "Başlık - Kaynak Adı" → "Başlık"
    cleaned = re.sub(r"\s*[-|]\s*[^-|]{3,50}$", "", title).strip()
    return cleaned if len(cleaned) > 10 else title.strip()

def _normalize_title(title: str) -> str:
    """Mükerrer tespiti için başlığı normalize et."""
    t = title.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:80]

def _is_relevant(title: str, ticker: str, company_tr: str) -> bool:
    """Haberin belirtilen hisse ile ilgili olup olmadığını kontrol eder.
    Sırasıyla: ticker kodu, parantezli ticker, şirket adı, alias listesi.
    """
    tl = title.lower()
    tu = ticker.upper()

    if tu in title.upper():
        return True
    if f"({tu})" in title:
        return True
    if company_tr and company_tr[:6].lower() in tl:
        return True
    # Alias kontrolü
    for alias in TICKER_ALIASES.get(tu, []):
        if alias.lower() in tl:
            return True
    return False

def _compute_confidence(item: NewsItem) -> float:
    """
    Haberin güven skorunu hesapla (0-100).
    Faktörler:
      - Kaynak tier'ı (1=yüksek, 3=düşük)
      - Resmi terim bonusu
      - Mükerrer sayısı (aynı haber → daha güvenilir)
      - Spekülasyon cezası
    """
    base = {1: 80.0, 2: 65.0, 3: 50.0}.get(item.tier, 50.0)

    # Resmi terim bonusu
    base += item.official_bonus * 0.5

    # Mükerrer bonusu: her tekrar +5, max +20
    base += min(20, (item.duplicate_count - 1) * 5)

    # Spekülasyon cezası
    if item.is_speculation:
        base -= 30

    return round(max(0.0, min(100.0, base)), 1)

# ─────────────────────────────────────────────────────────
# RSS ÇEKME ALTYAPISI
# ─────────────────────────────────────────────────────────

def _fetch_rss(
    url: str,
    source_name: str,
    ticker: str,
    cutoff: datetime,
    max_results: int = 30,
) -> list[NewsItem]:
    """Genel RSS çekici."""
    if not FEEDPARSER_OK:
        return []

    company_tr = TICKER_TO_TR.get(ticker.upper(), ticker)
    items      = []

    try:
        resp = None
        for _attempt in range(2):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=12)
                if resp.status_code == 200:
                    break
                logger.warning(f"{source_name} HTTP {resp.status_code} (attempt {_attempt+1})")
            except requests.exceptions.Timeout:
                logger.warning(f"{source_name} timeout (attempt {_attempt+1})")
                continue
            except requests.exceptions.ConnectionError:
                logger.warning(f"{source_name} connection error (attempt {_attempt+1})")
                continue
        if resp is None or resp.status_code != 200:
            return []

        feed = feedparser.parse(resp.content)

        for entry in feed.entries[:60]:
            title = _clean_title(entry.get("title") or "")
            if not title or len(title) < 10:
                continue

            if not _is_relevant(title, ticker, company_tr):
                continue

            pub = _parse_date(entry.get("published", ""))
            if pub and pub < cutoff:
                continue

            tl          = title.lower()
            is_kap      = "kap" in tl or "***" in title
            is_material = any(k in tl for k in [
                "özel durum", "bilanço", "temettü", "sermaye artırımı",
                "pay geri alım", "genel kurul", "halka arz", "ihale",
                "birleşme", "sözleşme", "faaliyet raporu",
            ])

            items.append(NewsItem(
                title=title,
                source=source_name,
                published=pub.isoformat() if pub else "",
                url=entry.get("link", ""),
                sentiment=_classify(title),
                is_kap=is_kap,
                is_material=is_material,
                is_speculation=_is_speculation(title),
                official_bonus=_official_bonus(title),
                tier=_get_tier(source_name),
            ))

        logger.info(f"{source_name}: {ticker} -> {len(items)} haber")

    except Exception as exc:
        logger.warning(f"{source_name} hata: {exc}")

    return items[:max_results]

# ─────────────────────────────────────────────────────────
# KAYNAK FONKSİYONLARI
# ─────────────────────────────────────────────────────────

def _src_haberturk(ticker, cutoff):
    """Habertürk Ekonomi — test'te ✅ doğrulandı."""
    for url in [
        "https://www.haberturk.com/rss/ekonomi.xml",
        "https://www.haberturk.com/rss",
    ]:
        r = _fetch_rss(url, "Habertürk", ticker, cutoff)
        if r: return r
    return []

def _src_ntv(ticker, cutoff):
    """NTV Ekonomi + NTV Para — test'te ✅ doğrulandı."""
    items = []
    for url, name in [
        ("https://www.ntv.com.tr/ekonomi.rss",  "NTV Ekonomi"),
        ("https://www.ntv.com.tr/ntvpara.rss",  "NTV Ekonomi"),
    ]:
        items.extend(_fetch_rss(url, name, ticker, cutoff))
    return items

def _src_hurriyet(ticker, cutoff):
    """Hürriyet Ekonomi RSS — test'te ✅ doğrulandı."""
    items = []
    for url in [
        "https://www.hurriyet.com.tr/rss/ekonomi",
        "https://www.hurriyet.com.tr/rss/anasayfa",
    ]:
        items.extend(_fetch_rss(url, "Hürriyet", ticker, cutoff))
    return items

def _src_milliyet(ticker, cutoff):
    """Milliyet Ekonomi RSS — test'te ✅ doğrulandı."""
    for url in [
        "https://www.milliyet.com.tr/rss/rssNew/ekonomiRss.xml",
    ]:
        r = _fetch_rss(url, "Milliyet", ticker, cutoff)
        if r: return r
    return []

def _src_sabah(ticker, cutoff):
    """Sabah Ekonomi + Finans — test'te ✅ doğrulandı."""
    items = []
    for url in [
        "https://www.sabah.com.tr/rss/ekonomi.xml",
        "https://www.sabah.com.tr/rss/finansekonomihaber.xml",
    ]:
        items.extend(_fetch_rss(url, "Sabah Ekonomi", ticker, cutoff))
    return items

def _src_dunya(ticker, cutoff):
    """Dünya Gazetesi RSS — test'te ✅ doğrulandı."""
    return _fetch_rss("https://www.dunya.com/rss", "Dünya Gazetesi", ticker, cutoff)

def _src_bloomberght(ticker, cutoff):
    """Bloomberg HT — test'te ✅ doğrulandı (yalnızca /rss çalışıyor)."""
    return _fetch_rss("https://www.bloomberght.com/rss", "Bloomberg HT", ticker, cutoff)

def _src_investing(ticker, cutoff):
    """Investing.com TR RSS — test'te ✅ doğrulandı."""
    items = []
    for url in [
        "https://tr.investing.com/rss/news.rss",
        "https://tr.investing.com/rss/news_285.rss",
    ]:
        items.extend(_fetch_rss(url, "Investing.com TR", ticker, cutoff))
    return items

def _src_finansgundemi(ticker, cutoff):
    """Finans Gündem RSS — test'te ✅ doğrulandı (yalnızca /rss çalışıyor)."""
    return _fetch_rss("https://www.finansingundemi.com/rss", "Finans Gündem", ticker, cutoff)

def _src_ekonomim(ticker, cutoff):
    """Ekonomim.com RSS — test'te ✅ doğrulandı."""
    return _fetch_rss("https://www.ekonomim.com/rss", "Ekonomim", ticker, cutoff)

def _src_paraanaliz(ticker, cutoff):
    """Para Analiz RSS — test'te ✅ doğrulandı."""
    items = []
    for url in [
        "https://www.paraanaliz.com/rss",
        "https://www.paraanaliz.com/feed",
    ]:
        items.extend(_fetch_rss(url, "Para Analiz", ticker, cutoff))
    return items

def _src_foreks(ticker, cutoff):
    """Foreks RSS — test'te ✅ doğrulandı."""
    return _fetch_rss("https://www.foreks.com/rss", "Foreks", ticker, cutoff)

def _src_cnnturk(ticker, cutoff):
    """CNN Türk Ekonomi RSS — test'te ✅ doğrulandı."""
    return _fetch_rss(
        "https://www.cnnturk.com/feed/rss/ekonomi/rss.xml",
        "CNN Türk Ekonomi",
        ticker,
        cutoff,
    )

def _src_google_tr(ticker, cutoff):
    if not FEEDPARSER_OK: return []
    company  = TICKER_TO_TR.get(ticker.upper(), ticker)
    aliases  = TICKER_ALIASES.get(ticker.upper(), [])
    items    = []
    # Temel sorgular
    queries  = [
        f'"{company}" borsa hisse',
        f"{ticker} BIST hisse senedi",
        f"{ticker} KAP bildirimi",
        f"{ticker} finansal sonuçlar",
    ]
    # Alias sorguları — ilk 2 alias ile ek arama
    for alias in aliases[:2]:
        queries.append(f'"{alias}" hisse borsa bist')
    for q in queries:
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=tr&gl=TR&ceid=TR:tr"
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:10]:
                t = _clean_title(e.get("title") or "")
                if not t or not _is_relevant(t, ticker, company):
                    continue
                pub  = _parse_date(e.get("published", ""))
                if pub and pub < cutoff: continue
                tl   = t.lower()
                items.append(NewsItem(
                    title=t, source="Google News TR",
                    published=pub.isoformat() if pub else "",
                    url=e.get("link", ""),
                    sentiment=_classify(t),
                    is_kap="kap" in tl or "***" in t,
                    is_material=any(k in tl for k in ["bilanço","temettü","ihale","sermaye"]),
                    is_speculation=_is_speculation(t),
                    official_bonus=_official_bonus(t),
                    tier=3,
                ))
        except Exception as exc:
            logger.warning(f"Google News TR hata: {exc}")
        time.sleep(0.2)
    logger.info(f"Google News TR: {ticker} -> {len(items)} haber")
    return items

# ─────────────────────────────────────────────────────────
# US MARKET KAYNAKLARI
# ─────────────────────────────────────────────────────────

def _src_yahoo_finance_rss(ticker, cutoff):
    """Yahoo Finance RSS - ABD hisseleri için Tier 2 kaynak."""
    if not FEEDPARSER_OK: return []
    items = []
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
        for e in feed.entries[:20]:
            t = _clean_title(e.get("title") or "")
            if not t or len(t) < 10:
                continue
            pub = _parse_date(e.get("published", ""))
            if pub and pub < cutoff:
                continue
            company = US_TICKER_TO_EN.get(ticker.upper(), ticker)
            if not _is_relevant(t, ticker, company):
                continue
            items.append(NewsItem(
                title=t, source="Yahoo Finance RSS",
                published=pub.isoformat() if pub else "",
                url=e.get("link", ""),
                sentiment=_classify(t),
                is_speculation=_is_speculation(t),
                official_bonus=_official_bonus(t),
                tier=2,
            ))
    except Exception as exc:
        logger.warning(f"Yahoo Finance RSS hata: {exc}")
    logger.info(f"Yahoo Finance RSS: {ticker} -> {len(items)} haber")
    return items


def _src_google_en_us(ticker, cutoff):
    """Google News EN for US stocks - Tier 3 kaynak."""
    if not FEEDPARSER_OK: return []
    company = US_TICKER_TO_EN.get(ticker.upper(), ticker)
    items = []
    queries = [
        f'"{company}" stock',
        f'{ticker} earnings quarterly results',
        f'{ticker} stock price target analyst',
    ]
    for q in queries:
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:8]:
                t = _clean_title(e.get("title") or "")
                if not t or len(t) < 10:
                    continue
                pub = _parse_date(e.get("published", ""))
                if pub and pub < cutoff:
                    continue
                if not _is_relevant(t, ticker, company):
                    continue
                items.append(NewsItem(
                    title=t, source="Google News US",
                    published=pub.isoformat() if pub else "",
                    url=e.get("link", ""),
                    sentiment=_classify(t),
                    is_speculation=_is_speculation(t),
                    official_bonus=_official_bonus(t),
                    tier=3,
                ))
            time.sleep(0.2)
        except Exception as exc:
            logger.warning(f"Google News US hata: {exc}")
    logger.info(f"Google News US: {ticker} -> {len(items)} haber")
    return items


def _src_bing_en_us(ticker, cutoff):
    """Bing News EN for US stocks - Tier 3 kaynak."""
    company = US_TICKER_TO_EN.get(ticker.upper(), ticker)
    url = f"https://www.bing.com/news/search?q={quote(company + ' stock earnings')}&format=rss"
    return _fetch_rss(url, "Bing News US", ticker, cutoff, max_results=15)


def _src_yahoo_us(ticker, cutoff):
    """Yahoo Finance yfinance API for US stocks - no .IS suffix."""
    if not YFINANCE_OK: return []
    items = []
    try:
        news = yf.Ticker(ticker.upper()).news or []
        for n in news[:15]:
            ts  = n.get("providerPublishTime", 0)
            pub = datetime.fromtimestamp(ts) if ts else datetime.now()
            if pub < cutoff: continue
            t = (n.get("title") or "").strip()
            if not t: continue
            items.append(NewsItem(
                title=t, source="Yahoo Finance US",
                published=pub.isoformat(),
                url=n.get("link", ""),
                sentiment=_classify(t),
                is_speculation=_is_speculation(t),
                official_bonus=_official_bonus(t),
                tier=2,
            ))
        logger.info(f"Yahoo Finance US: {ticker} -> {len(items)} haber")
    except Exception as exc:
        logger.warning(f"Yahoo Finance US hata: {exc}")
    return items


def _src_google_en(ticker, cutoff):
    if not FEEDPARSER_OK: return []
    en = TICKER_TO_EN.get(ticker.upper())
    if not en: return []
    items = []
    q   = f'"{en}" stock BIST Turkey'
    url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        for e in feed.entries[:8]:
            t = _clean_title(e.get("title") or "")
            if not t: continue
            pub = _parse_date(e.get("published", ""))
            if pub and pub < cutoff: continue
            items.append(NewsItem(
                title=t, source="Google News EN",
                published=pub.isoformat() if pub else "",
                url=e.get("link", ""),
                sentiment=_classify(t),
                is_speculation=_is_speculation(t),
                official_bonus=_official_bonus(t),
                tier=3,
            ))
    except Exception as exc:
        logger.warning(f"Google News EN hata: {exc}")
    logger.info(f"Google News EN: {ticker} -> {len(items)} haber")
    return items

def _src_bing(ticker, cutoff):
    company = TICKER_TO_TR.get(ticker.upper(), ticker)
    url     = f"https://www.bing.com/news/search?q={quote(company+' hisse borsa')}&format=rss"
    return _fetch_rss(url, "Bing News", ticker, cutoff)

def _src_yahoo(ticker, cutoff):
    if not YFINANCE_OK: return []
    items = []
    try:
        # .IS only for BIST tickers; US tickers queried as-is
        yt = ticker.upper()
        if yt not in US_TICKER_TO_EN:
            yt += ".IS"
        news = yf.Ticker(yt).news or []
        for n in news[:15]:
            ts  = n.get("providerPublishTime", 0)
            pub = datetime.fromtimestamp(ts) if ts else datetime.now()
            if pub < cutoff: continue
            t = (n.get("title") or "").strip()
            if not t: continue
            items.append(NewsItem(
                title=t, source="Yahoo Finance",
                published=pub.isoformat(),
                url=n.get("link", ""),
                sentiment=_classify(t),
                is_speculation=_is_speculation(t),
                official_bonus=_official_bonus(t),
                tier=3,
            ))
        logger.info(f"Yahoo Finance: {ticker} -> {len(items)} haber")
    except Exception as exc:
        logger.warning(f"Yahoo Finance hata: {exc}")
    return items

# ─────────────────────────────────────────────────────────
# ANA ANALİZ FONKSİYONU
# ─────────────────────────────────────────────────────────

def analyze_news(
    ticker: str,
    days: int = 30,
    language: str = "TR",        # "TR" | "EN" | "BOTH"
    min_confidence: float = 30.0, # Bu güvenin altındaki haberler skora dahil edilmez
) -> NewsAnalysisResult:
    """
    Çoklu kaynaktan haber çekip gelişmiş sentiment analizi yapar.

    Args:
        ticker         : BIST kodu (ör. 'THYAO')
        days           : Kaç günlük haber (varsayılan 14)
        language       : "TR", "EN" veya "BOTH"
        min_confidence : Skora dahil edilecek minimum güven eşiği (0-100)

    Returns:
        NewsAnalysisResult
    """
    result = NewsAnalysisResult(language=language)
    cutoff = datetime.now() - timedelta(days=days)

    # ── BERT Sentiment — lazy-load (ilk çağrıda ~500 MB indirilir) ──
    bert_loaded = _bert.load()
    if bert_loaded:
        logger.info(f"{ticker} | Sentiment motoru: BERT (savasy/bert-base-turkish-sentiment-cased)")
        result.sentiment_engine = "bert"
    else:
        logger.info(f"{ticker} | Sentiment motoru: Kelime sayma (BERT kurulu değil)")
        result.sentiment_engine = "kelime_sayma"

    # ── Aktif kaynaklar (dil seçeneğine göre) ────────────
    source_fns = []

    if language in ("TR", "BOTH"):
        source_fns += [
            # ── Tier 1 (test'te ✅) ──────────────────────────
            _src_haberturk,     # Bloomberg HT, NTV — doğrulandı
            _src_ntv,           # NTV Ekonomi + NTV Para — doğrulandı
            _src_bloomberght,   # Bloomberg HT — doğrulandı
            # ── Tier 2 (test'te ✅) ──────────────────────────
            _src_hurriyet,      # Hürriyet — doğrulandı
            _src_milliyet,      # Milliyet — doğrulandı
            _src_sabah,         # Sabah Ekonomi + Finans — doğrulandı
            _src_dunya,         # Dünya Gazetesi — doğrulandı
            _src_investing,     # Investing.com TR — doğrulandı
            _src_finansgundemi, # Finans Gündem — doğrulandı
            _src_ekonomim,      # Ekonomim — doğrulandı
            _src_paraanaliz,    # Para Analiz — doğrulandı
            _src_foreks,        # Foreks — doğrulandı
            _src_cnnturk,       # CNN Türk Ekonomi — doğrulandı
            # ── Tier 3 ───────────────────────────────────────
            _src_google_tr,
            _src_bing,
        ]

    if language in ("EN", "BOTH"):
        source_fns += [
            _src_google_en,
            _src_yahoo,
            # US-specific sources
            _src_yahoo_finance_rss,
            _src_google_en_us,
            _src_bing_en_us,
            _src_yahoo_us,
        ]

    if language == "EN":
        source_fns = [
            _src_yahoo_finance_rss,
            _src_google_en_us,
            _src_bing_en_us,
            _src_yahoo_us,
            _src_google_en,
        ]

    # ── Paralel çekme ─────────────────────────────────────
    all_raw: list[NewsItem] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fn, ticker, cutoff): fn.__name__ for fn in source_fns}
        for future in as_completed(futures, timeout=20):
            try:
                items = future.result()
                if items:
                    all_raw.extend(items)
                    src = items[0].source
                    if src not in result.data_sources:
                        result.data_sources.append(src)
            except Exception as exc:
                logger.warning(f"Kaynak exception: {exc}")

    # ── Mükerrer tespiti ve duplicate_count artırımı ──────
    # Başlık benzerliğine göre grupla, her grupta en güvenilir kaynağı tut
    normalized_map: dict[str, list[NewsItem]] = {}
    for item in all_raw:
        key = _normalize_title(item.title)
        normalized_map.setdefault(key, []).append(item)

    deduplicated: list[NewsItem] = []
    for key, group in normalized_map.items():
        # En düşük tier (= en güvenilir) kaynağı seç
        best = min(group, key=lambda x: (x.tier, -_get_weight(x.source)))
        best.duplicate_count = len(group)
        deduplicated.append(best)

    # ── Güven skoru hesapla ───────────────────────────────
    for item in deduplicated:
        item.confidence = _compute_confidence(item)

    # ── Spekülasyon filtrele ──────────────────────────────
    clean_items    = [i for i in deduplicated if not i.is_speculation]
    filtered_count = len(deduplicated) - len(clean_items)
    result.filtered_count = filtered_count

    if filtered_count > 0:
        logger.info(f"{filtered_count} spekülatif haber filtrelendi")

    # ── Minimum güven eşiği uygula ────────────────────────
    scored_items = [i for i in clean_items if i.confidence >= min_confidence]

    if not scored_items:
        result.score        = 50.0
        result.data_quality = "notr"
        result.warning      = (
            f"{ticker} icin yeterli guvenilir haber bulunamadi "
            f"(min_confidence={min_confidence}). "
            "Sentiment skoru notr (50) olarak ayarlandi."
        )
        return result

    # ── Ağırlıklı sentiment skoru ─────────────────────────
    weighted_pos = 0.0
    weighted_neg = 0.0
    total_weight = 0.0

    for item in scored_items:
        # Güven skoru üstel ağırlık: %58→0.34, %80→0.64, %99→0.98
        # Doğrusal yerine karesel: yüksek BERT güveni çok daha fazla etki eder
        w = _get_weight(item.source) * (item.confidence / 100) ** 2

        if item.sentiment == "olumlu":
            weighted_pos += w
            result.positive_count += 1
        elif item.sentiment == "olumsuz":
            weighted_neg += w
            result.negative_count += 1
        else:
            result.neutral_count += 1

        total_weight += w

    result.total_news = len(scored_items)

    # Ağırlıklı oran
    if total_weight > 0:
        raw_score = (weighted_pos / (weighted_pos + weighted_neg) * 100
                     if (weighted_pos + weighted_neg) > 0 else 50.0)
    else:
        raw_score = 50.0

    # Güven düzeltmesi: az haber → 50'ye yakın
    confidence_factor = min(1.0, result.total_news / 10)
    final_score = 50.0 + (raw_score - 50.0) * confidence_factor

    result.score = round(max(0.0, min(100.0, final_score)), 1)

    # ── Çıktıları doldur ──────────────────────────────────
    scored_items.sort(key=lambda x: (x.tier, -x.confidence, x.published), reverse=False)
    scored_items.sort(key=lambda x: x.published, reverse=True)

    result.headlines    = [i.title for i in scored_items]
    result.news_items   = [
        {
            "title":          i.title,
            "source":         i.source,
            "tier_label":     _get_tier_label(i.source),
            "published":      i.published,
            "url":            i.url,
            "sentiment":      i.sentiment,
            "confidence":     i.confidence,
            "duplicate_count": i.duplicate_count,
            "is_kap":         i.is_kap,
            "is_material":    i.is_material,
            "official_bonus": i.official_bonus,
        }
        for i in scored_items
    ]
    result.kap_disclosures = [n for n in result.news_items if n["is_kap"]]
    result.material_events = [n for n in result.news_items if n["is_material"]]

    # Kalite etiketi
    has_tier1 = any(_get_tier(i.source) == 1 for i in scored_items)
    if has_tier1 and result.total_news >= 5:
        result.data_quality = "yuksek"
    elif result.total_news >= 5:
        result.data_quality = "orta"
    elif result.total_news >= 2:
        result.data_quality = "dusuk"
        result.warning = f"Sadece {result.total_news} guvenilir haber bulundu."
    else:
        result.data_quality = "notr"
        result.score  = 50.0
        result.warning = "Yeterli haber yok, skor notr."

    logger.info(
        f"{ticker} | {result.total_news} haber ({filtered_count} filtrelendi) | "
        f"Skor: {result.score} | P/N: {result.positive_count}/{result.negative_count} | "
        f"Kalite: {result.data_quality} | Kaynaklar: {result.data_sources}"
    )
    return result

def analyze_news_for_date(
    ticker: str,
    target_date,
    lookback_days: int = 7,
) -> tuple:
    """
    target_date öncesi lookback_days günlük Google News haberlerine bakarak
    o tarih için sentiment döner. Backtest tarih-bazlı haber filtresi için.

    Google News after:/before: operatörleriyle tarih filtreli sorgu kullanır.
    BERT yerine hızlı keyword matching kullanır (backtest hızı için).

    Returns: (sentiment: 'olumlu' | 'olumsuz' | 'notr', haber_sayisi: int)
    """
    if not FEEDPARSER_OK:
        return "notr", 0

    try:
        if isinstance(target_date, str):
            target_dt = datetime.strptime(target_date[:10], "%Y-%m-%d")
        else:
            target_dt = target_date

        from_dt    = target_dt - timedelta(days=lookback_days)
        after_str  = from_dt.strftime("%Y-%m-%d")
        before_str = target_dt.strftime("%Y-%m-%d")

        company = TICKER_TO_TR.get(ticker.upper(), ticker)

        # US stock detection: if ticker is in US map or has no TR mapping
        is_us = ticker.upper() in US_TICKER_TO_EN
        if is_us:
            company = US_TICKER_TO_EN.get(ticker.upper(), ticker)

        if is_us:
            queries = [
                f'"{company}" stock after:{after_str} before:{before_str}',
                f'{ticker} earnings stock price after:{after_str} before:{before_str}',
            ]
            lang_params = "hl=en&gl=US&ceid=US:en"
        else:
            queries = [
                f'"{company}" borsa hisse after:{after_str} before:{before_str}',
                f'{ticker} BIST hisse after:{after_str} before:{before_str}',
            ]
            lang_params = "hl=tr&gl=TR&ceid=TR:tr"

        sentiments = []
        for q in queries:
            url = (
                f"https://news.google.com/rss/search?"
                f"q={quote(q)}&{lang_params}"
            )
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[:10]:
                    t = _clean_title(e.get("title") or "")
                    if not t or not _is_relevant(t, ticker, company):
                        continue
                    sent = _check_strong_patterns(t) or _classify_financial_keywords(t)
                    sentiments.append(sent)
                time.sleep(0.15)
            except Exception as exc:
                logger.debug(f"analyze_news_for_date sorgu hatası: {exc}")

        if not sentiments:
            return "notr", 0

        pos   = sentiments.count("olumlu")
        neg   = sentiments.count("olumsuz")
        total = len(sentiments)

        # Güçlü negatif: en az 2 negatif haber + negatif çoğunluk
        if neg >= 2 and neg > pos:
            return "olumsuz", total
        elif pos > neg:
            return "olumlu", total
        return "notr", total

    except Exception as exc:
        logger.warning(f"analyze_news_for_date hatası ({ticker}, {target_date}): {exc}")
        return "notr", 0


# ─────────────────────────────────────────────────────────
# STREAMLIT RENDER
# ─────────────────────────────────────────────────────────

def render_news_panel(result: NewsAnalysisResult):
    try:
        import streamlit as st
    except ImportError:
        return

    # Dil / Kalite başlığı
    lang_label = {"TR": "🇹🇷 Türkçe", "EN": "🇬🇧 English", "BOTH": "🇹🇷+🇬🇧"}.get(
        result.language, result.language
    )

    quality_map = {
        "yuksek": ("✅ High quality — Tier 1 sources confirmed", "success"),
        "orta":   ("Medium quality — no Tier 1 source", "warning"),
        "dusuk":  ("Low data", "warning"),
        "notr":   ("No reliable data — sentiment neutral", "info"),
    }
    msg, kind = quality_map.get(result.data_quality, ("", "info"))
    getattr(st, kind)(f"{msg} | {lang_label}")

    # ── Sentiment Motoru Göstergesi ──────────────────────
    if result.sentiment_engine == "bert":
        st.success("🧠 **BERT AI Aktif** — Türkçe bağlam anlayışıyla sentiment analizi yapılıyor")
    else:
        st.info("📝 **Kelime Sayma** — `pip install transformers torch` ile BERT'i aktif et")

    if result.warning:
        st.caption(f"{result.warning}")
    if result.filtered_count:
        st.caption(f"🚫 {result.filtered_count} speculative article(s) filtered out")
    if result.data_sources:
        st.caption(f"Sources: {' · '.join(result.data_sources)}")

    if not result.news_items:
        if not FEEDPARSER_OK:
            st.error("feedparser kurulu değil → pip install feedparser")
        return

    # Metrikler
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total",     result.total_news)
    c2.metric("🟢 Pos",    result.positive_count)
    c3.metric("🔴 Neg",    result.negative_count)
    c4.metric("⬜ Neutral", result.neutral_count)
    c5.metric("🚫 Filtered", result.filtered_count)

    if result.material_events:
        st.success(f"Grade C {len(result.material_events)} official price-sensitive event(s)")

    st.markdown("---")

    # Haber listesi
    use_bert = result.sentiment_engine == "bert"
    for item in result.news_items[:15]:
        sent        = item.get("sentiment", "notr")
        emoji       = {"olumlu": "🟢", "olumsuz": "🔴", "notr": "⬜"}.get(sent, "⬜")
        tier_label  = item.get("tier_label", "Grade C")
        kap_badge   = "**KAP** | " if item.get("is_kap") else ""
        mat_badge   = "Grade C " if item.get("is_material") else ""
        dup         = item.get("duplicate_count", 1)
        dup_badge   = f"🔁×{dup} " if dup > 1 else ""
        conf        = item.get("confidence", 50)
        pub         = item.get("published", "")[:10]

        # BERT etiketi
        sent_label_map = {"olumlu": "OLUMLU", "olumsuz": "OLUMSUZ", "notr": "NOTR"}
        sent_color_map = {"olumlu": "#22c55e", "olumsuz": "#ef4444", "notr": "#94a3b8"}
        sent_label = sent_label_map.get(sent, "NOTR")
        sent_color = sent_color_map.get(sent, "#94a3b8")
        bert_badge = (
            f" &nbsp;<span style='background:{sent_color};color:#fff;"
            f"font-size:10px;padding:1px 6px;border-radius:4px;font-weight:bold'>"
            f"{'BERT: ' if use_bert else ''}{sent_label} {conf:.0f}%</span>"
        )

        st.markdown(
            f"{emoji} {mat_badge}{dup_badge}{kap_badge}"
            f"**{item.get('title', '')}**{bert_badge}  \n"
            f"<span style='font-size:11px;color:#94a3b8'>"
            f"{tier_label} {item.get('source','')} · {pub}"
            f"</span>",
            unsafe_allow_html=True,
        )
        url = item.get("url", "")
        if url:
            st.markdown(
                f"<a href='{url}' target='_blank' "
                f"style='font-size:11px;color:#60a5fa'>Read →</a>",
                unsafe_allow_html=True,
            )
        st.markdown("")


# ─────────────────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ticker = (
        sys.argv[1].upper() if len(sys.argv) > 1
        else input("Ticker (e.g. THYAO): ").strip().upper() or "THYAO"
    )
    lang = sys.argv[2].upper() if len(sys.argv) > 2 else "TR"

    print(f"\nAnalyzing: {ticker} | Language: {lang} | Days: 30")
    print("-" * 60)

    r = analyze_news(ticker, days=30, language=lang)

    print(f"Score        : {r.score}/100")
    print(f"Quality      : {r.data_quality}")
    print(f"Language     : {r.language}")
    print(f"Sources      : {r.data_sources}")
    print(f"Total        : {r.total_news}")
    print(f"Pos / Neg    : {r.positive_count} / {r.negative_count}")
    print(f"KAP items    : {len(r.kap_disclosures)}")
    print(f"Material     : {len(r.material_events)}")
    print(f"Filtered     : {r.filtered_count} (speculative)")
    if r.warning:
        print(f"Warning      : {r.warning}")

    print("\nTop Headlines (with confidence):")
    for item in r.news_items[:8]:
        tier  = item.get("tier_label", "")
        conf  = item.get("confidence", 0)
        dup   = item.get("duplicate_count", 1)
        dup_s = f" [x{dup}]" if dup > 1 else ""
        print(f"  {tier} [{conf:.0f}%]{dup_s} {item['title'][:65]}")
        print(f"       └─ {item['source']} · {item['published'][:10]}")