"""
skor_gunlugu — Canlı bileşik skorun BİLEŞENLERİNİ her gün kalıcı olarak kaydeder.

Neden var (CLAUDE.md > handoff 2.3): canlı skorun ~%40'ı (haber %10, analist
primi %20, değerleme %10) hiç ölçülmedi ve tarihsel olarak ölçülemez — Google
News arşiv vermiyor, analist hedeflerinin geçmişi yok. Tek yol ileriye dönük
kayıt. Eskiden bileşenler yalnız `analysis_history`'ye gidiyordu: Streamlit
Cloud'da /tmp'de yaşıyor (konteynerle ölüyor) ve yalnız kullanıcının elle
baktığı hisseleri içeriyordu — seçilmiş, taraflı, kalıcı olmayan bir örnek.

Burada: robot her BIST kapanışından sonra TÜM tarama evreninin (~480 hisse)
tam bileşik skorunu `compute_bist_score` ile — canlı sistemin kullandığı
fonksiyonun KENDİSİYLE, kopyasıyla değil — hesaplar ve Turso'daki
`skor_gunlugu` tablosuna yazar. Birkaç ay sonra ileri getirilerle
birleştirilince bileşen başına IC çıkar.

Kapsam bilerek YALNIZ BIST: analist hedefi yalnız BIST için otomatik çekiliyor
(US'te prim hep eksik olurdu), haber motoru Türkçe kaynaklı, ve US evreni
(~1400 hisse) günlük süreyi üçe katlardı.

Kullanım:
    python skor_gunlugu.py yaz      # robotun çağırdığı — evreni skorla, Turso'ya yaz
    python skor_gunlugu.py ozet     # tabloda ne var: gün başına satır, kapsama
"""

import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import logging  # noqa: E402
logging.disable(logging.WARNING)   # hisse başı yfinance/haber uyarıları çıktıyı boğuyor

import bist_analyzer as ba  # noqa: E402

PIYASA = "BIST"
ISCI = 8            # ölçüldü: 8 işçi → hisse başı ~1.6s, 16 → ~1.2s ama haber
                    # kaynaklarında boş dönüş başladı (kısıtlama belirtisi)
YAZ_PARTI = 50      # Turso'ya kaç satırda bir yazılır (yarıda ölürse kayıp sınırlı)
MIN_KAPSAMA = 0.80  # evrenin bundan azı skorlanabildiyse koşu HATALI sayılır
MAX_HABERSIZ = 0.90 # satırların bundan fazlası haber_n=0 ise kaynaklar çökmüş olabilir

TABLO_SQL = """
    CREATE TABLE IF NOT EXISTS skor_gunlugu (
        tarih        TEXT,     -- koşu günü (YYYY-MM-DD)
        market       TEXT,
        ticker       TEXT,
        surum        TEXT,     -- SCORING_VERSION: v3/v4 satırları karışmasın
        bar_tarihi   TEXT,     -- fiyat serisinin son günü (tatil/bayat veri ayırt edilsin)
        fiyat        REAL,
        toplam       REAL,
        sinyal       TEXT,
        teknik       REAL,
        sentiment    REAL,
        prim         REAL,
        deger        REAL,
        w_teknik     INTEGER,  -- o gün GERÇEKTE uygulanan ağırlıklar
        w_sentiment  INTEGER,
        w_prim       INTEGER,
        w_deger      INTEGER,
        haber_n      INTEGER,  -- skora giren haber sayısı (0 = haber yok, ağırlık tekniğe gitti)
        haber_pos    INTEGER,
        haber_neg    INTEGER,
        haber_bilgisiz INTEGER,
        duygu_motoru TEXT,     -- bert | kelime_sayma
        kaynaklar    TEXT,     -- haber getiren kaynaklar (KAP düştü mü? — gün gün ayırt edilsin)
        hedef        REAL,     -- analist hedef fiyatı (yoksa NULL)
        analist_n    INTEGER,
        fk           REAL,
        pddd         REAL,
        rs           REAL,     -- endekse göre göreli güç (teknik skora bonus veriyor)
        week52_pos   REAL,
        kaydedildi   TEXT,
        PRIMARY KEY (tarih, market, ticker)
    )
"""

KOLONLAR = ("tarih", "market", "ticker", "surum", "bar_tarihi", "fiyat", "toplam",
            "sinyal", "teknik", "sentiment", "prim", "deger", "w_teknik",
            "w_sentiment", "w_prim", "w_deger", "haber_n", "haber_pos", "haber_neg",
            "haber_bilgisiz", "duygu_motoru", "kaynaklar", "hedef", "analist_n", "fk", "pddd",
            "rs", "week52_pos", "kaydedildi")


def _sayi(v):
    """NaN/inf ve numpy tiplerini Turso'nun kabul edeceği şekle çevir."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _satir(ticker: str, tarih: str) -> tuple:
    """Bir hisse için canlı bileşik skoru hesapla → tablo satırı. Hata yükseltir."""
    s = ba.compute_bist_score(ticker, market=PIYASA)
    if s.signal == "HATA":
        raise RuntimeError(s.stock.error or "veri yok")
    nr = getattr(s, "_news_result", None)
    w = getattr(s, "_agirliklar", (None,) * 4)
    tv = getattr(s, "_tv_target", None) or {}
    df = s.stock.df
    bar = df.index[-1].strftime("%Y-%m-%d") if df is not None and len(df) else None
    return (
        tarih, PIYASA, ticker, ba.SCORING_VERSION, bar,
        _sayi(s.stock.current_price), _sayi(s.total_score), s.signal,
        _sayi(s.teknik_score), _sayi(s.sentiment_score),
        _sayi(s.prim_score), _sayi(s.deger_score),
        *w,
        getattr(nr, "total_news", None), getattr(nr, "positive_count", None),
        getattr(nr, "negative_count", None), getattr(nr, "bilgisiz_count", None),
        getattr(nr, "sentiment_engine", None),
        ",".join(getattr(nr, "data_sources", None) or []) or None,
        _sayi(s.valuation.target_price), tv.get("count"),
        _sayi(s.stock.pe_ratio), _sayi(s.stock.pb_ratio),
        _sayi(s.technical.relative_strength), _sayi(s.technical.week52_position),
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def _yaz(satirlar: list) -> None:
    sql = (f"INSERT OR REPLACE INTO skor_gunlugu ({', '.join(KOLONLAR)}) "
           f"VALUES ({', '.join('?' * len(KOLONLAR))})")
    ba._PMDB.execute_batch([(sql, r) for r in satirlar])


def yaz() -> int:
    tarih = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Skor günlüğü | {tarih} | {PIYASA} ===")

    # Robotun US turu (21:30 UTC) BIST işi yapmaz; kapanıştaki tur yazdı.
    only = (os.getenv("ROBOT_ONLY_MARKET") or "").strip().upper()
    if only and only != PIYASA:
        print(f"    kısıtlı tur ({only}) — skor günlüğü yalnız {PIYASA}, atlanıyor")
        return 0

    # Robotla aynı ilke: yapılandırılmış olmak erişilebilir olmak değildir,
    # ve CI'da lokal SQLite'a yazmak konteynerle birlikte çöpe yazmaktır.
    if os.getenv("GITHUB_ACTIONS") and not ba._PMDB.is_cloud():
        print("HATA: CI'da Turso yapılandırılmamış — yazılacak kalıcı yer yok.")
        return 1
    try:
        ba._PMDB.execute(TABLO_SQL)
    except Exception as exc:
        print(f"HATA: skor_gunlugu tablosu hazırlanamadı — {type(exc).__name__}: {exc}")
        return 1

    evren = ba.get_scan_universe(PIYASA)
    # Paylaşılan önbellekleri iş parçacıkları başlamadan ısıt: yoksa ilk 8
    # hisse aynı anda TradingView hedef sorgusunu ve endeks indirmesini
    # tetikler (8 kopya istek, olası kısıtlama).
    ba.fetch_tv_analyst_targets()
    ba._fetch_benchmark_cached(ba._default_index(PIYASA))
    print(f"    evren: {len(evren)} hisse, {ISCI} işçi")

    t0 = time.time()
    bekleyen, yazilan, hatali = [], 0, []
    with ThreadPoolExecutor(ISCI) as ex:
        isler = {ex.submit(_satir, t, tarih): t for t in evren}
        for i, fut in enumerate(as_completed(isler), 1):
            t = isler[fut]
            try:
                bekleyen.append(fut.result())
            except Exception as exc:
                hatali.append((t, str(exc)[:60]))
            if len(bekleyen) >= YAZ_PARTI:
                _yaz(bekleyen)     # TursoUnavailable burada yükselir → koşu kırmızı
                yazilan += len(bekleyen)
                bekleyen = []
            if i % 100 == 0:
                print(f"    {i}/{len(evren)} ({time.time() - t0:.0f}s)")
    if bekleyen:
        _yaz(bekleyen)
        yazilan += len(bekleyen)

    sure = time.time() - t0
    print(f"    yazıldı: {yazilan}/{len(evren)} hisse, {len(hatali)} hata, {sure:.0f}s")
    for t, e in hatali[:10]:
        print(f"      {t}: {e}")

    # Çıkış kodu gerçeği söylemeli (bkz. robotun 2026-07-31 olayı). İki sessiz
    # bozulma biçimi yakalanır: evrenin büyük kısmı skorlanamadı, ya da haber
    # kaynakları CI IP'sini kısıtladı ve her hisse "haber yok" döndü — ikincisi
    # sessizce yanlış bir duygu günlüğü üretir, tam da ölçmeye çalıştığımız şey.
    sorun = []
    if not evren or yazilan < MIN_KAPSAMA * len(evren):
        sorun.append(f"kapsama düşük ({yazilan}/{len(evren)})")
    if yazilan:
        r = ba._PMDB.execute(
            "SELECT SUM(CASE WHEN COALESCE(haber_n,0)=0 THEN 1 ELSE 0 END) AS h, "
            "SUM(CASE WHEN kaynaklar LIKE '%KAP%' THEN 1 ELSE 0 END) AS k, "
            "COUNT(*) AS n FROM skor_gunlugu WHERE tarih=? AND market=?",
            (tarih, PIYASA))["rows"][0]
        n = max(r.get("n") or 1, 1)
        oran = (r.get("h") or 0) / n
        kap = (r.get("k") or 0) / n
        print(f"    haberi olmayan hisse: %{oran * 100:.0f} | KAP'tan veri gelen: %{kap * 100:.0f}")
        if oran > MAX_HABERSIZ:
            sorun.append(f"haber kaynakları boş dönüyor (%{oran * 100:.0f} haber_n=0)")
        # KAP, ölçülmüş iki sinyalin (geri alım +, bedelsiz −) TEK girdisi.
        # Hiç gelmemesi koşuyu kırmızı YAPMAZ — borsapy yolu bilinen kırılgan
        # bir dış kaynak, kalıcı kırmızı sinyali köreltir (ENAG gibi) — ama
        # o günün duygu satırları "KAP'sız" diye ayırt edilebilsin diye
        # yüksek sesle söylenir; `kaynaklar` kolonu da aynı bilgiyi taşır.
        if kap == 0:
            print("    UYARI: KAP hiçbir hissede veri getirmedi — ölçülmüş KAP "
                  "sinyalleri bugün girdisiz (borsapy yolu düşmüş olabilir)")
    if sorun:
        print(f"=== Skor günlüğü SORUNLU: {'; '.join(sorun)} ===")
        return 1
    print("=== Skor günlüğü tamamlandı ===")
    return 0


def ozet() -> int:
    ba._PMDB.execute(TABLO_SQL)
    rows = ba._PMDB.execute("""
        SELECT tarih, market, surum, COUNT(*) AS n,
               AVG(CASE WHEN haber_n > 0 THEN 1.0 ELSE 0 END) AS haberli,
               AVG(CASE WHEN hedef IS NOT NULL THEN 1.0 ELSE 0 END) AS hedefli,
               AVG(CASE WHEN fk IS NOT NULL THEN 1.0 ELSE 0 END) AS fkli,
               AVG(CASE WHEN kaynaklar LIKE '%KAP%' THEN 1.0 ELSE 0 END) AS kapli,
               AVG(toplam) AS ort_toplam
        FROM skor_gunlugu GROUP BY tarih, market, surum ORDER BY tarih DESC LIMIT 30
    """)["rows"]
    if not rows:
        print("skor_gunlugu boş.")
        return 0
    print(f"{'tarih':10} {'piy':4} {'sürüm':5} {'n':>4} {'haberli':>8} "
          f"{'KAP':>5} {'hedefli':>8} {'F/K':>6} {'ort.skor':>8}")
    for r in rows:
        print(f"{r['tarih']:10} {r['market']:4} {r['surum']:5} {r['n']:>4} "
              f"{r['haberli'] * 100:>7.0f}% {r['kapli'] * 100:>4.0f}% "
              f"{r['hedefli'] * 100:>7.0f}% "
              f"{r['fkli'] * 100:>5.0f}% {r['ort_toplam']:>8.1f}")
    return 0


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else ""
    if komut == "yaz":
        sys.exit(yaz())
    if komut == "ozet":
        sys.exit(ozet())
    print(__doc__)
    sys.exit(2)
