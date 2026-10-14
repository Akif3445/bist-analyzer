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

VERİ KASASI (2026-08-31): son 2 yıl KİLİTLİ. Aşağıdaki komutların HİÇBİRİ
veri_kasasi.json'daki kesim tarihinden sonrasını görmez; kilit varsayılandır,
bayrak gerektirmez (bkz. VERİ KASASI bölümü). Bu yüzden bu dosyadaki komutlar
2026-08-31 öncesinde yayımlanmış sayılardan FARKLI sonuç verir — pencere artık
2 yıl kısa. Karşılaştırmayı kasa-öncesi sayılarla değil, kasa altındaki yeni
taban çizgisiyle yap.

Kullanım: python pipeline_backtest.py dondur [yil=5]        # evren+panel dondur
          python pipeline_backtest.py run   [yil_sayisi=5]   # tam simülasyon
          python pipeline_backtest.py karsilastir [yil=5]    # yönetilen vs yönetimsiz
          python pipeline_backtest.py sweep [yil_sayisi=5]   # yönetim kuralı taraması
          python pipeline_backtest.py validate [yil=8]        # örneklem-dışı doğrulama
          python pipeline_backtest.py kasa                    # kasa durumu
          python pipeline_backtest.py kasa-ac [yil=8] EVET    # REZERV DÖNEM — tek sefer
"""

import os as _os
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

# PİYASA ETKİSİ (slippage) — bilinen yanlılık listesindeki 5. madde.
# COST_RT düz bir yüzdedir ve işlem BÜYÜKLÜĞÜNÜ hiç dikkate almaz. Gerçekte
# alım fiyatı yukarı, satış fiyatı aşağı iter; etki, işlemin o hissenin
# günlük hacmine oranıyla büyür. Portföy ince hacimli isimler tutuyor
# (OZATD, OZSUB, ODINE...), yani bu kalem teorik değil.
#
# Karekök yasası: etki ≈ K × sqrt(işlem / günlük hacim). K=0.02 ile
#   hacmin  %1'i  → %0.20
#   hacmin %10'u  → %0.63
#   hacmin %50'si → %1.41
# Literatürdeki ampirik aralığın muhafazakâr ucu; kesin bir sabit değil,
# büyüklük mertebesi tahmini olarak okunmalı.
SLIP_K = 0.02


def _piyasa_etkisi(feats, ticker, tarih, islem_tl):
    """Tek işlemin piyasa etkisi (oran). Hacim bilinmiyorsa ceza uygulanır."""
    if islem_tl <= 0:
        return 0.0
    f = feats.get(ticker) if feats else None
    if f is None:
        return SLIP_K
    try:
        tv = f["tlvol"]
        ix = tv.index.searchsorted(tarih, side="right") - 1
        adv = float(tv.iloc[ix]) if ix >= 0 else 0.0
    except Exception:
        adv = 0.0
    if adv <= 0:
        return SLIP_K
    return SLIP_K * min(1.0, islem_tl / adv) ** 0.5
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


# VERİ KASASI (2026-08-31) — geliştirmenin GÖREMEDİĞİ rezerv dönem
#
# Bugüne dek her ölçüm 8 yıllık pencerenin TAMAMINI gördü. Deflated Sharpe
# sonucu bunun bedelini sayıyla söylüyor: ~695 konfigürasyon denendiği için
# DSR 0.68 < 0.95, yani backtest TEK BAŞINA kenarın varlığını kuramıyor.
# Kritik sonuç şu: aynı veri üzerinde daha fazla test yapmak güveni ARTIRMAZ,
# her yeni deneme N'i büyütüp DSR'yi düşürür.
#
# Bu yüzden pencerenin son 2 yılı KİLİTLENİR. Tüm geliştirme kalan 6 yılda
# yapılır; kasa Ekim'de BİR KEZ açılır.
#
# İDDİANIN SINIRI — abartmamak şart: rezerv dönem "hiç bakılmamış veri"
# DEĞİLDİR. Bu dosyadaki 8 yıllık tablolar onu zaten kapsıyordu ve iki
# parametre kararı (takas_farki 12→6, skor_cikis 43→39) onu içeren
# pencerelerde alındı. Kasanın verdiği garanti daha dar ama yine de değerli:
# 2026-08-31'DEN SONRAKİ hiçbir karar ona uydurulmuş olamaz. Gerçekten
# kirlenmemiş kanıt hâlâ ileri zamandır (canlı gölge kohortlar) ve
# parametrelerin görmediği piyasalardır (US transfer testi).
#
# Mekanizma bilerek "unutulamaz" kuruldu:
#
#   1. Kilit VARSAYILANDIR, opsiyonel değil. Hiçbir komuta bayrak eklemek
#      gerekmez; hepsi penceresini _endeks_penceresi() üzerinden alır.
#   2. Panel yüklenirken de kesim tarihinde budanır (_kasa_panel). İki
#      katmanlı: gözden kaçan bir kod yolu bile kasadaki günlere erişemez.
#   3. Açmak tek yönlüdür ve açık onay ister; açılış veri_kasasi.json'a
#      tarih/gerekçe/sonuçla yazılır. Bir kez görülen veri bir daha kasa
#      olamaz, dolayısıyla bu dosya sonucun ne kadar temiz olduğunun kaydı.
#
# Pencere semantiği: BAŞLANGIÇ bugünden geriye sayılmaya devam eder, yalnız
# SON kasa kesiminde biter. Böylece yayımlanmış sonuçların başlangıç tarihleri
# değişmez ve karşılaştırılabilir kalır — kilitliyken `8` = 2018-08 → 2024-08
# (yani 6 yıllık geliştirme penceresi), `karsilastir 8` hâlâ aynı krizleri
# içerir. Alternatif (kesimden geriye saymak) 2016-08 → 2024-08 verirdi ve
# bu dosyadaki hiçbir sayıyla kıyaslanamazdı.
# Yol BETİĞE görelidir, çalışma dizinine değil: kasa dosyası bulunamazsa kilit
# sessizce KALKAR ve bu, sessizce yanlış sonuç üreten türden bir hatadır.
# Başka bir dizinden çalıştırmak yeterliydi.
KASA_DOSYA = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "veri_kasasi.json")
_KASA_ACIK = False          # yalnızca `kasa-ac` komutu True yapar
_KASA_META = None


def _kasa_meta() -> dict:
    global _KASA_META
    if _KASA_META is None:
        import json
        if _os.path.exists(KASA_DOSYA):
            with open(KASA_DOSYA, encoding="utf-8") as f:
                _KASA_META = json.load(f)
        else:
            _KASA_META = {}
            print(f"UYARI: {KASA_DOSYA} bulunamadı — VERİ KASASI KİLİDİ YOK, "
                  "bu koşu rezerv dönemi de görüyor.")
    return _KASA_META


def _kasa_tarihi():
    """Kilitliyken veri kesim tarihi (Timestamp), kasa açıkken None."""
    if _KASA_ACIK:
        return None
    k = _kasa_meta().get("kesim")
    return pd.Timestamp(k) if k else None


def _kasa_panel(closes, vols):
    """Paneli kasa kesiminde budar — ikinci güvence katmanı.

    Tüm göstergeler geriye dönük (SMA, ATR, rolling hacim), dolayısıyla
    kuyruğu kesmek kesimden ÖNCEKİ hiçbir değeri değiştirmez. Budanmış panel
    için AYRI bir parmak izi basılır: donmuş panelin izi tüm veriyi kapsar,
    kasa altındaki bir sonuç ise başka bir veri kümesinden gelir ve çıktıda
    ayırt edilebilmesi gerekir.
    """
    import hashlib
    kes = _kasa_tarihi()
    if kes is None:
        return closes, vols
    c = {t: s[s.index <= kes] for t, s in closes.items()}
    c = {t: s for t, s in c.items() if len(s) > 0}
    v = {t: vols[t][vols[t].index <= kes] for t in c if t in vols}
    iz = hashlib.sha256(
        "|".join(f"{t}:{len(c[t])}:{c[t].iloc[-1]:.4f}"
                 for t in sorted(c)).encode()).hexdigest()[:12]
    print(f"[KASA] panel {kes:%Y-%m-%d} tarihinde budandı → {len(c)} hisse "
          f"| kasa parmak izi {iz}")
    return c, v


def _endeks_penceresi(sembol, years):
    """İndeks kapanışı + simülasyon penceresi, kasa kesimi uygulanmış.

    Tüm komutlar penceresini buradan alır. Tek kapı olması şart: kasa
    kilidini her komuta ayrı ayrı eklemek, er geç bir komutta unutulur ve
    kilit sessizce delinir (bkz. robot piyasa izolasyonundaki tuzak).
    """
    ix = yf.download(sembol, period=f"{years+1}y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(ix.columns, pd.MultiIndex):
        ix.columns = ix.columns.get_level_values(0)
    seri = ix["Close"].dropna()
    kes = _kasa_tarihi()
    if kes is None:
        start = seri.index[-1] - pd.DateOffset(years=years)
        dates = seri.index[seri.index >= start]
    else:
        # PENCERE İKİ UCUNDAN DA SABİT (2026-09-13 düzeltmesi).
        #
        # Kasa yalnız SONU sabitlemişti; başlangıç "bugünden geriye" sayılmaya
        # devam ediyordu. Sonuç: pencere her gün bir gün kayıyor ve iki hafta
        # sonra aynı komut başka bir sayı veriyordu. Ölçüldü — `us 8`
        # 2026-08-31'de 2018-08-31→2024-08-29 (72 rebalans, NAV 4.55) iken
        # 2026-09-13'te 2018-09-11→2024-08-30 (71 rebalans, NAV 4.58) oldu.
        # Dosyaya yazılmış bir sayının iki hafta sonra tekrarlanamaması, tam
        # da donmuş panelin engellemek için var olduğu şey.
        #
        # Çözüm: başlangıcı da KESİME çıpala. `years` anlamını korur —
        # "kasa olmasaydı kullanacağım pencere" — ve rezerv yıl düşülür:
        #   years=8, rezerv=2 → 2024-08-31 − 6 yıl = 2018-08-31
        # Bu, yayımlanmış 8 yıllık sonuçların başlangıç tarihinin AYNISIDIR,
        # yani hem karşılaştırılabilirlik hem tekrarlanabilirlik korunur.
        rezerv = float(_kasa_meta().get("rezerv_yil", 2))
        gelistirme = max(1.0, years - rezerv)
        start = kes - pd.DateOffset(months=int(round(gelistirme * 12)))
        seri = seri[seri.index <= kes]
        dates = seri.index[(seri.index >= start) & (seri.index <= kes)]
        if len(dates) < 60:
            raise SystemExit(
                f"Kasa kesimi ({kes:%Y-%m-%d}) sonrası pencerede yalnız "
                f"{len(dates)} gün kaldı — daha uzun bir yıl sayısı ver.")
        print(f"[KASA] pencere {dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d} "
              f"({(dates[-1]-dates[0]).days/365.25:.1f} yıl) | son "
              f"{_kasa_meta().get('rezerv_yil', 2)} yıl KİLİTLİ")
    return seri, dates


def _download_panel(tickers, years, tekrar=2, ek=".IS"):
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
                    t = s[:-len(ek)] if ek else s
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

    syms = [t + ek for t in tickers]
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
        return _kasa_panel(d["closes"], d["vols"])

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
    return _kasa_panel(closes, vols)


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


def _pick_at(date, feats, ret_panel, max_pos=None, kur=None):
    """Bugünkü propose() mantığının point-in-time kopyası → {ticker: weight}.

    max_pos: pozisyon sayısı (varsayılan MAX_POS). Parametreleştirildi çünkü
    pozisyon SAYISI bugüne dek hiç taranmamıştı — 6 değeri bir varsayımdı."""
    if max_pos is None:
        max_pos = MAX_POS
    # Kurulum parametreleri de dışarıdan verilebilir. Hiçbiri bugüne dek
    # taranmamıştı; "makul görünen" varsayımlardı ve bu dosyada o ifade
    # defalarca yanlış çıktı (kontrarian skor, düşüş freni, takas_farki...).
    k = kur or {}
    _min_vol = k.get("min_tl_vol", MIN_TL_VOL)
    _skor_min = k.get("score_min", SCORE_MIN)
    _sek_cap = k.get("sector_cap", SECTOR_CAP)
    _kor_lim = k.get("corr_limit", CORR_LIMIT)
    # Sektör kaynağı dışarıdan verilebilir. _sector_of yalnız BIST ve US
    # sektörlerini biliyor; üçüncü bir piyasada HER hisse "Diğer"e düşer, tek
    # sektör olur ve sector_cap=2 portföyü 6 yerine 2 pozisyona indirir —
    # sessizce, hata vermeden, bambaşka bir stratejiyi ölçerek.
    _sek_map = k.get("sektor")
    # Sıralama anahtarı dışarıdan verilebilir (momentum kontrolü için). Verilirse
    # sistemin skoru HİÇ kullanılmaz — score_min eşiği de dahil — yani kalan her
    # şey (likidite, SMA200, sektör, korelasyon, ATR ağırlık) aynı makine, yalnız
    # "hangi hisse daha iyi" sorusunun cevabı değişir.
    _sira = k.get("sira")
    # Sağlamlık testi (2026-10-07, ön kayıt bilanco_esikler.json):
    #   "haric"  → bu hisseler aday listesine hiç girmez (A: filtre)
    #   "saglik" → {hisse: 0-1}; verilirse sıralama 0.9×teknik + 10×sağlamlık
    #              (bilinmiyorsa 0.5) + 0.4×3A momentum olur (B: puan)
    # İkisi de verilmezse davranış önceki ile birebir aynıdır.
    _haric = k.get("haric") or set()
    _saglik = k.get("saglik")
    cands = []
    for t, f in feats.items():
        if t in _haric:
            continue
        cl = f["close"]
        ix = cl.index.searchsorted(date, side="right") - 1
        if ix < 260:
            continue
        d = cl.index[ix]
        if (date - d).days > 7:      # o civarda veri yoksa (delist/askı) alma
            continue
        price, score = float(cl.iloc[ix]), float(f["score"].iloc[ix])
        sma, tlv, atr = float(f["sma200"].iloc[ix] or 0), float(f["tlvol"].iloc[ix] or 0), float(f["atr_pct"].iloc[ix])
        if _sira is not None:
            rank = _sira(cl, ix)
            if tlv < _min_vol or price <= 0 or not sma or price < sma or rank is None:
                continue
            cands.append((t, rank, atr))
            continue
        if tlv < _min_vol or price <= 0 or not sma or price < sma or score < _skor_min:
            continue
        mom3 = (price / float(cl.iloc[max(0, ix-63)]) - 1) * 100
        if _saglik is not None:
            sg = _saglik.get(t)
            sg = 0.5 if sg is None or not np.isfinite(sg) else sg
            cands.append((t, 0.9 * score + 10.0 * sg + mom3 * 0.4, atr))
        else:
            cands.append((t, score + mom3 * 0.4, atr))
    cands.sort(key=lambda x: -x[1])

    picks, sec_n, kept = [], {}, []
    win = ret_panel.loc[:date].tail(120)
    for t, _rank, atr in cands:
        sec = _sek_map.get(t, "Diğer") if _sek_map else _sector_of(t)
        if sec_n.get(sec, 0) >= _sek_cap:
            continue
        if kept and t in win.columns:
            cors = win[kept].corrwith(win[t]).abs()
            if (cors > _kor_lim).any():
                continue
        picks.append((t, atr))
        kept.append(t) if t in win.columns else None
        sec_n[sec] = sec_n.get(sec, 0) + 1
        if len(picks) >= max_pos:
            break
    if not picks:
        return {}
    inv = np.array([1 / max(a, 0.5) for _, a in picks])
    w = inv / inv.sum()
    w = np.minimum(w, 0.30); w = w / w.sum()
    return {t: float(wi) for (t, _a), wi in zip(picks, w)}


def _walk(weights_by_date, closes, dates, feats=None, portfoy_tl=0.0):
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
            # portfoy_tl=0 → etki kapalı, davranış eskisiyle BİREBİR aynı
            if portfoy_tl > 0:
                etki = 0.0
                for t in set(new_w) | set(cur_w):
                    d = abs(new_w.get(t, 0) - cur_w.get(t, 0))
                    if d > 0:
                        etki += d / 2 * _piyasa_etkisi(feats, t, d1, portfoy_tl * v * d)
                v *= (1 - etki)
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
    xu, dates = _endeks_penceresi("XU100.IS", years)
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
    # Ampirik persentil de basılır: "bandın içinde" bir hakeme yetmez, nerede
    # olduğu sorulur. Bandın alt ucundaki bir sonuçla üst ucundaki aynı cümleyle
    # raporlanmamalı — kasa altındaki koşu tam da bu ayrımı gerektirdi.
    _yzd = float((rng_final < nav_sys.iloc[-1]).mean() * 100)
    print(f"\nRastgele dağılımda sistemin yeri: son NAV {nav_sys.iloc[-1]:.2f} | "
          f"medyan {np.median(rng_final):.2f} | "
          f"rastgele %5-%95 bandı [{np.percentile(rng_final,5):.2f}, {np.percentile(rng_final,95):.2f}] "
          f"→ %{_yzd:.0f} persentil ({(rng_final < nav_sys.iloc[-1]).sum()}/{len(rng_final)} tohum) "
          f"{'>%95 (rastgeleden anlamlı iyi)' if beat else '— bandın içinde'}")

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

# Gerçek ATR / yalnız-kapanıştan ATR oranı. Panel High/Low taşımadığı için
# _precompute'taki ATR gün içi aralığı görmez ve küçük çıkar. Ölçüldü
# (2026-09-27, 40 BIST hissesi, son 120 gün): medyan 2.01, %25-%75 [1.94, 2.12]
# — dar ve tutarlı. Canlıdaki k×ATR stopunu backtest'e aynı mesafeyle taşımak
# için gerekir; olmasa stop iki kat sıkı olur ve maliyeti abartılırdı.
ATR_OLCEK = 2.0


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


def _walk_managed(weights_by_date, closes, dates, feats, par, portfoy_tl=0.0):
    """Aylık hedefler + GÜNLÜK yönetim (trailing stop, skor çıkışı, tavan,
    düşüş freni, takas eşiği). Nakit gerçek pozisyondur (getirisi 0).

    Canlıda olup burada VARSAYILAN OLARAK OLMAYAN iki mekanizma seçenekle
    açılabilir (1.4 devir ölçümü, 2026-09-27); ikisi de kapalıyken davranış
    öncekiyle birebir aynıdır:
      par["atr_stop"]    = k  → alışta giriş − k×ATR sabit stop (canlı:
                                PortfolioManager.STOP_MULTS). Panel yalnız
                                kapanış taşıdığı için ATR ATR_OLCEK ile büyütülür.
      par["gunluk_alim"] = True → boştaki nakit her gün değerlendirilir
                                (canlı: PortfolioDoctor adım 4); kapalıyken nakit
                                ay başına kadar bekler.
    s.attrs["tutma"] (çıkışlarda iş günü cinsinden elde tutma listesi) ve
    s.attrs["devir"] (tek yön devir toplamı) her koşuda yazılır — NAV'a dokunmaz.
    """
    atr_k  = float(par.get("atr_stop", 0) or 0)
    gunluk = bool(par.get("gunluk_alim", False))
    giris, stop_px, son_satis = {}, {}, {}
    tutmalar, devir_top = [], 0.0
    if gunluk:
        from bist_analyzer import PortfolioDoctor as _PD
        _aday_min = float(_PD.KURAL["min_aday_skor"])
        _bekle = int(_PD.KURAL["bekleme_gun"])
        _gun_max = int(_PD.KURAL["gunluk_islem"])
        _ix = pd.DatetimeIndex(dates)
        _hz = lambda k: pd.DataFrame({t: f[k] for t, f in feats.items()}).reindex(
            _ix, method="ffill", limit=5)
        G_S, G_C, G_SMA, G_TV = _hz("score"), _hz("close"), _hz("sma200"), _hz("tlvol")
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

        w_bas = {t: st["w"] for t, st in cur.items()}   # devir/tutma defteri için

        # 2) GÜNLÜK YÖNETİM
        satilan_w = 0.0
        satilanlar = {}      # ticker -> satılan ağırlık (piyasa etkisi için)
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
            # a0) sabit ATR stopu (yalnız atr_stop > 0 iken; canlıdaki "Stop kesildi")
            if atr_k and t in stop_px and float(p) <= stop_px[t]:
                satilan_w += cur[t]["w"]
                satilanlar[t] = satilanlar.get(t, 0) + cur[t]["w"]
                del cur[t]; continue
            # a) takip eden stop
            if cur[t]["peak"] > 0 and float(p) <= cur[t]["peak"] * (1 - trail):
                satilan_w += cur[t]["w"]
                satilanlar[t] = satilanlar.get(t, 0) + cur[t]["w"]
                del cur[t]; continue
            # b) skor çıkışı
            sk = _skor_at(feats, t, d1)
            if sk is not None and sk < skor_c:
                satilan_w += cur[t]["w"]
                satilanlar[t] = satilanlar.get(t, 0) + cur[t]["w"]
                del cur[t]; continue
            # c) pozisyon tavanı
            if cur[t]["w"] > tavan:
                satilan_w += cur[t]["w"] - tavan
                satilanlar[t] = satilanlar.get(t, 0) + (cur[t]["w"] - tavan)
                cur[t]["w"] = tavan
        if satilan_w > 0:
            v *= (1 - COST_RT / 2 * satilan_w)     # tek yön maliyet
            islem_maliyeti_toplam += COST_RT / 2 * satilan_w
            if portfoy_tl > 0:
                _e = 0.0
                for _t, _w in satilanlar.items():
                    _e += _w * _piyasa_etkisi(feats, _t, d1, portfoy_tl * v * _w)
                v *= (1 - _e)
                islem_maliyeti_toplam += _e

        # 3) Rebalans günü: takas eşiğiyle hedefe yaklaş
        while ri < len(rebal) and rebal[ri] <= d1:
            hedef = weights_by_date[rebal[ri]]
            # Fren "nakit yüksekse serbest bırakılsın" diye denendi (2026-08-18)
            # ve ÖLÇÜLDÜ: reel CAGR %15.6 → %11.7, MaxDD %-25.2 → %-39.5.
            # Frendeyken yüksek nakit "pozisyonlar düşen piyasada stoplandı"
            # demek; oraya geri girmek zarar ettiriyor. Fren işini yapıyor,
            # dokunulmadı. Sezgi yine yanıldı, ölçüm düzeltti.
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
            if portfoy_tl > 0:
                _e = 0.0
                for _t in set(yeni_w) | set(cur):
                    _d = abs(yeni_w.get(_t, 0) - cur.get(_t, {}).get("w", 0))
                    if _d > 0:
                        _e += _d / 2 * _piyasa_etkisi(feats, _t, d1, portfoy_tl * v * _d)
                v *= (1 - _e)
                islem_maliyeti_toplam += _e
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

        # 4) GÜNLÜK NAKİT DEĞERLENDİRME (yalnız gunluk_alim) — PortfolioDoctor
        # adım 4'ün kopyası: skor >= min_aday_skor, SMA200 üstü, likit, yakın
        # zamanda satılmamış, sektör tavanı; günde en fazla gunluk_islem alım.
        if gunluk and v >= nav_peak * (1 - fren):
            nakit = 1.0 - sum(st["w"] for st in cur.values())
            if nakit >= 0.08:
                sk, pr, sm, tv = G_S.iloc[i], G_C.iloc[i], G_SMA.iloc[i], G_TV.iloc[i]
                uygun = sk[(sk >= _aday_min) & (pr > sm) & (tv >= MIN_TL_VOL)].dropna()
                sek_n = {}
                for t in cur:
                    sek_n[_sector_of(t)] = sek_n.get(_sector_of(t), 0) + 1
                alinan = 0
                for t in uygun.sort_values(ascending=False).index:
                    if alinan >= _gun_max or nakit < 0.08:
                        break
                    if t in cur or i - son_satis.get(t, -10**6) < _bekle:
                        continue
                    sec = _sector_of(t)
                    if sek_n.get(sec, 0) >= SECTOR_CAP:
                        continue
                    w_y = min(nakit, tavan, 1.0 / MAX_POS)
                    v *= (1 - COST_RT / 2 * w_y)
                    islem_maliyeti_toplam += COST_RT / 2 * w_y
                    cur[t] = {"w": w_y, "peak": float(pr[t])}
                    nakit -= w_y
                    sek_n[sec] = sek_n.get(sec, 0) + 1
                    alinan += 1

        # Defter: giren/çıkan pozisyonlar ve günün tek yön devri
        for t in set(w_bas) - set(cur):
            if t in giris:
                tutmalar.append(i - giris.pop(t))
            son_satis[t] = i
            stop_px.pop(t, None)
        for t in set(cur) - set(w_bas):
            giris[t] = i
            if atr_k:
                try:
                    _f = feats[t]
                    _j = _f["close"].index.searchsorted(d1, side="right") - 1
                    _px = float(_f["close"].iloc[_j])
                    _atr = float(_f["atr_pct"].iloc[_j]) / 100.0 * ATR_OLCEK
                    stop_px[t] = _px * (1 - atr_k * _atr)
                except Exception:
                    pass
        devir_top += sum(abs(cur.get(t, {}).get("w", 0) - w_bas.get(t, 0))
                         for t in set(w_bas) | set(cur)) / 2

        nav.append(v)

    s = pd.Series(nav, index=dates)
    s.attrs["maliyet"] = islem_maliyeti_toplam
    s.attrs["tutma"] = tutmalar
    s.attrs["devir"] = devir_top
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

    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
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

    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
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


def pozisyon(years=8, katman=5):
    """POZİSYON SAYISI TARAMASI — MAX_POS hiç test edilmemişti.

    İki ayrı soruyu birlikte yanıtlar, çünkü ikisi zıt yönde çeker:

      1. GETİRİ: pozisyon sayısı artınca en iyi fikirler seyreliyor mu?
         Yönetimli/yönetimsiz getiri, MaxDD ve Sharpe her N için ölçülür,
         ayrıca walk-forward katmanlarında kaç kez varsayılanı geçtiği.

      2. ŞANS: aynı evrenden N hisse çekilen 50 rastgele portföyün son
         NAV dağılımı ne kadar geniş? Bu, "portföyün sonucu ne kadar
         kendi becerisi, ne kadar çekiliş" sorusunun doğrudan ölçüsü.
         Canlıda kullanıcının iki portföyü arasında 38 puan fark çıktı;
         bu bölüm o farkın N ile nasıl daraldığını gösterir.

    Bu ikisi birlikte okunmalı: N büyüdükçe beklenen getiri bir miktar
    düşebilir ama kötü bir çekiliş yapma ihtimali de düşer. Hangisinin
    daha değerli olduğu risk tercihidir — ölçüm ikisini de verir.
    """
    t0 = time.time()
    print(f"=== Pozisyon Sayısı Taraması | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})

    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
    rebal_dates = pd.date_range(dates[0], dates[-1], freq="MS")
    rebal_dates = [dates[pd.Index(dates).searchsorted(d)] for d in rebal_dates if d <= dates[-1]]
    xu_n = xu.loc[dates] / xu.loc[dates].iloc[0]
    print(f"Panel {len(feats)} hisse | {dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d} "
          f"({time.time()-t0:.0f}s)\n")

    denenecek = [4, 6, 8, 10, 12, 15]

    print("BÖLÜM 1 — Sistem performansı, pozisyon sayısına göre")
    print(f"{'N':>3} {'reel CAGR':>11} {'MaxDD':>8} {'Sharpe':>7} {'yönetimli reel':>15} "
          f"{'yön. MaxDD':>11} {'yön. Sharpe':>12}")
    print("-" * 74)
    wbd_cache = {}
    for n in denenecek:
        wbd = {}
        for d in rebal_dates:
            w = _pick_at(d, feats, ret_panel, max_pos=n)
            if w:
                wbd[d] = w
        wbd_cache[n] = wbd
        su = _stats(_walk(wbd, closes, dates), xu_n, "_")
        sm = _stats(_walk_managed(wbd, closes, dates, feats, dict(MGMT_VARSAYILAN)), xu_n, "_")
        isaret = " *" if n == MAX_POS else ""
        print(f"{n:>3} {su['reel_yillik']:>10.1f}% {su['maxdd']:>7.1f}% {su['sharpe']:>7.2f} "
              f"{sm['reel_yillik']:>14.1f}% {sm['maxdd']:>10.1f}% {sm['sharpe']:>12.2f}{isaret}")
    print("(* mevcut varsayılan)\n")

    print("BÖLÜM 2 — ŞANS FAKTÖRÜ: aynı evrenden N hisselik 50 rastgele portföy")
    print(f"{'N':>3} {'%5 dilim':>10} {'medyan':>9} {'%95 dilim':>11} "
          f"{'bant genişliği':>16} {'en kötü/en iyi':>16}")
    print("-" * 72)
    all_t = sorted(feats)
    for n in denenecek:
        finaller = []
        for seed in range(RANDOM_SEEDS):
            rng = np.random.default_rng(seed)
            wbd_r = {}
            for d in rebal_dates:
                elig = [t for t in all_t if feats[t]["close"].index.searchsorted(d) > 260]
                if len(elig) >= n:
                    sel = rng.choice(elig, n, replace=False)
                    wbd_r[d] = {t: 1.0 / n for t in sel}
            finaller.append(_walk(wbd_r, closes, dates).iloc[-1])
        f = np.array(finaller)
        p5, p50, p95 = np.percentile(f, [5, 50, 95])
        print(f"{n:>3} {p5:>10.1f} {p50:>9.1f} {p95:>11.1f} {p95/p5:>15.1f}x "
              f"{f.min()/f.max():>15.2f}")
    print()
    print("Okuma: bant genişliği (%95 ÷ %5) düştükçe sonuç 'hangi hisseleri")
    print("çektiğine' daha az bağlı demektir. Beklenen getiri ile bu daralma")
    print("arasındaki denge, pozisyon sayısı kararının özüdür.\n")

    # BÖLÜM 3 — tek pencere optimumuna GÜVENME. Bu projede tam bu tür
    # sonuçlar üç kez çürüdü (takas_farki etkileşimi, max_pozisyon, trailing).
    # wbd'ler yukarıda zaten kuruldu; katmanlara bölmek ucuz.
    n = len(dates)
    sinir = [int(n * i / (katman + 1)) for i in range(katman + 2)]
    print(f"BÖLÜM 3 — WALK-FORWARD: eğitimde seçilen N, testte tutuyor mu ({katman} katman)")
    print(f"{'katman':<8} {'test dönemi':<20} {'eğitim seçti':>13} "
          f"{'test varsayılan':>16} {'test seçilen':>14}")
    print("-" * 78)
    kazanma, secimler = 0, []
    for k in range(katman):
        eg_dates = dates[:sinir[k + 1]]
        te_dates = dates[sinir[k + 1]:sinir[k + 2]]
        en_iyi, en_iyi_g = None, -1e18
        for nn in denenecek:
            w_eg = {d: w for d, w in wbd_cache[nn].items() if d <= eg_dates[-1]}
            if not w_eg:
                continue
            g = _ozet(_walk_managed(w_eg, closes, eg_dates, feats, dict(MGMT_VARSAYILAN)))[0]
            if g > en_iyi_g:
                en_iyi, en_iyi_g = nn, g
        if en_iyi is None:
            continue
        def _te(nn):
            w = {d: x for d, x in wbd_cache[nn].items()
                 if eg_dates[-1] < d <= te_dates[-1]}
            return _ozet(_walk_managed(w, closes, te_dates, feats,
                                       dict(MGMT_VARSAYILAN)))[0] if w else float("nan")
        t_v, t_s = _te(MAX_POS), _te(en_iyi)
        fark = (t_s - t_v) / max(abs(t_v), 1e-9) * 100
        tuttu = (en_iyi != MAX_POS) and fark > 5.0
        if tuttu:
            kazanma += 1
        secimler.append(en_iyi)
        print(f"{k+1:<8} {te_dates[0]:%Y-%m}→{te_dates[-1]:%Y-%m}      {en_iyi:>13} "
              f"{t_v:>15.1f}% {t_s:>13.1f}% {'  ✓' if tuttu else '  ·'}")
    print("-" * 78)
    onay = len(secimler) >= 3 and kazanma > len(secimler) / 2
    print(f"Eğitimin seçimleri: {secimler} | {kazanma}/{len(secimler)} katmanda tuttu")
    print(f"KARAR: {'DEĞİŞTİR → ' + str(max(set(secimler), key=secimler.count)) if onay else 'DEĞİŞTİRME (MAX_POS=%d kalsın)' % MAX_POS}")
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")


# TRANSFER TESTİ PİYASALARI (2026-09-13)
#
# us_transfer bir kopyala-yapıştır ile üçüncü piyasaya genişletilebilirdi ve
# bu, bu projede birkaç kez canımızı yakan hata sınıfının ta kendisi olurdu
# (bkz. MGMT_VARSAYILAN'ın elle kopyalanmış hâli, _walk vs _walk_managed).
# Tek fonksiyon, piyasa başına yalnız YAPISAL çeviri tablosu:
#
#   ek          yfinance sembol eki (.L, .DE — US'te yok)
#   endeks      karşılaştırma endeksi
#   maliyet     round-trip işlem maliyeti (KOMİSYON YAPISI, ayar değil)
#   hacim_esigi likidite tabanı, o piyasanın KENDİ para biriminde
#   tv          TradingView pazar adı (evreni dondururken)
#
# Davranışsal parametrelerin hiçbiri burada yok ve OLMAMALI: skor_cikis,
# takas_farki, trailing_pct, max_pozisyon, MAX_POS, sector_cap, corr_limit,
# score_min hepsi BIST'te seçilmiş değerlerinde kalır. Testin bütün kanıt
# değeri bundan geliyor.
TRANSFER_PIYASA = {
    "US": {
        "ad": "ABD", "tv": "america", "ek": "", "endeks": "^GSPC",
        "endeks_ad": "S&P 500", "maliyet": 0.002, "hacim_esigi": 50e6,
        "birim": "USD", "limit": 600,
        "not": "US nominal USD, enflasyon düzeltmesi yok.",
    },
    "UK": {
        # LSE fiyatları PENİ (GBp) cinsinden gelir: AZN 11708 = 117.08 sterlin.
        # Getiriler ölçekten bağımsız olduğu için bu sonuçları etkilemez, AMA
        # hacim eşiğini etkiler — eşik de peni cinsinden verilmeli.
        # 100_000_000 peni = 1M sterlin/gün.
        "ad": "Birleşik Krallık", "tv": "uk", "ek": ".L", "endeks": "^FTSE",
        "endeks_ad": "FTSE 100", "maliyet": 0.006, "hacim_esigi": 100e6,
        "birim": "GBp", "limit": 700,
        # Maliyet BIST'ten YÜKSEK ve bu bilerek: %0.5 damga vergisi (stamp
        # duty) alışta alınır, üstüne komisyon/spread. Yanlış tarafa yuvarlamak
        # sonucu BİZE KARŞI bozar — güvenli yön budur.
        "not": "UK nominal GBP. Maliyet %0.6 (damga vergisi dahil), BIST'in "
               "üstünde — bilerek muhafazakâr.",
    },
}


def _transfer_evren(piyasa, cfg):
    """Piyasanın evrenini TV'den çeker ve DONDURUR (ticker + SEKTÖR).

    Sektörün de donması şart: _sector_of yalnız BIST/US biliyor, üçüncü bir
    piyasada her hisse "Diğer" olur, tek sektör kalır ve sector_cap portföyü
    6 yerine 2 pozisyona düşürürdü — sessizce, hata vermeden, bambaşka bir
    stratejiyi ölçerek. Ayrıca evren dosyası BIST/US'teki gibi
    tekrarlanabilirlik çapasıdır (bkz. TEKRARLANABİLİRLİK ÇAPASI).
    """
    import json
    dosya = f"backtest_evren_{piyasa.lower()}.json"
    if _os.path.exists(dosya):
        with open(dosya, encoding="utf-8") as f:
            m = json.load(f)
        print(f"{piyasa} evreni DOSYADAN: {len(m['tickers'])} hisse "
              f"(donduruldu {m.get('tarih')})")
        return m["tickers"], m.get("sektor", {})

    from tradingview_screener import Query
    _n, df = (Query().set_markets(cfg["tv"])
              .select("name", "close", "average_volume_30d_calc", "sector")
              .limit(cfg["limit"]).get_scanner_data())
    df["_hacim"] = df["close"] * df["average_volume_30d_calc"]
    df = df[df["_hacim"] >= cfg["hacim_esigi"]].sort_values("_hacim", ascending=False)

    tickers, sektor = [], {}
    for _, r in df.iterrows():
        ham = str(r.get("name") or "").strip()
        if not ham:
            continue
        # LSE sembolleri sonda nokta taşıyabilir (BP., RR.) — yfinance'te
        # "BP..L" diye bir sembol yok, "BP.L" var.
        t = ham.rstrip(".")
        if not t or not any(c.isalpha() for c in t):
            continue
        tickers.append(t)
        sek = r.get("sector")
        sektor[t] = sek if isinstance(sek, str) and sek else "Diğer"
    tickers = list(dict.fromkeys(tickers))

    meta = {"tarih": datetime.now().strftime("%Y-%m-%d"), "piyasa": piyasa,
            "kaynak": f"TradingView {cfg['tv']}, hacme göre azalan",
            "hacim_esigi": cfg["hacim_esigi"], "birim": cfg["birim"],
            "not": "Tekrarlanabilirlik çapası — bu listeyi değiştirmek geçmiş "
                   "sonuçları karşılaştırılamaz kılar.",
            "tickers": tickers, "sektor": sektor}
    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"{piyasa} evreni DONDURULDU -> {dosya} ({len(tickers)} hisse, "
          f"eşik {cfg['hacim_esigi']/1e6:.0f}M {cfg['birim']})")
    return tickers, sektor


def _transfer_panel(piyasa, years):
    """Transfer piyasasının donmuş paneli (yoksa indir+dondur), kasa budamalı.

    _transfer'dan çıkarıldı: momentum kontrolü de aynı panelleri okuyor ve
    panel yükleme mantığının iki kopyası, bu dosyanın defalarca kaydettiği
    sürüklenme hatası sınıfıdır.
    """
    import hashlib
    import pickle
    cfg = TRANSFER_PIYASA[piyasa]
    pn_dosya = f"backtest_panel_{piyasa.lower()}_{years}y.pkl"
    sektor = {}
    if _os.path.exists(pn_dosya):
        with open(pn_dosya, "rb") as f:
            d = pickle.load(f)
        closes, vols = d["closes"], d["vols"]
        sektor = d.get("sektor", {})
        print(f"{piyasa} panel ÖNBELLEKTEN: {len(closes)} hisse | "
              f"parmak izi {d['iz']}")
    else:
        uni, sektor = _transfer_evren(piyasa, cfg)
        closes, vols = _download_panel(uni, years, ek=cfg["ek"])
        iz = hashlib.sha256("|".join(f"{t}:{len(closes[t])}" for t in sorted(closes))
                            .encode()).hexdigest()[:12]
        with open(pn_dosya, "wb") as f:
            pickle.dump({"closes": closes, "vols": vols, "iz": iz,
                         "sektor": sektor}, f)
        print(f"{piyasa} panel yazıldı -> {pn_dosya} | {len(closes)} hisse | iz {iz}")

    # Kasa bu panelde de geçerli: pencere zaten _endeks_penceresi ile kesiliyor
    # ama panelin kendisini de budamak ikinci güvence katmanı (BIST tarafındaki
    # _panel_hazirla ile aynı gerekçe).
    closes, vols = _kasa_panel(closes, vols)
    return closes, vols, sektor


def _transfer(piyasa="US", years=8):
    """TRANSFER TESTİ — mekânda örneklem-dışı doğrulama.

    Zamanda örneklem-dışı test (walk-forward) yıllar ister; mekânda olan
    istemez. Aynı kurulum mantığı tamamen farklı bir piyasada da çalışıyorsa,
    bu "yöntem genelleniyor" demektir ve beklemeden öğrenilir.

    KRİTİK KURAL: davranışsal parametrelere DOKUNULMAZ. skor_cikis,
    takas_farki, trailing_pct, max_pozisyon, MAX_POS, sector_cap, corr_limit
    — hepsi BIST'te seçilmiş değerlerle kalır. Ayarlarsam iyi bir sonuç
    hiçbir şey kanıtlamaz; testin bütün değeri parametrelerin BAŞKA bir
    piyasada seçilmiş olmasından geliyor. Yalnızca YAPISAL/BİRİM çevirileri
    yapılır ve hepsi TRANSFER_PIYASA tablosunda görünür durur.
    """
    cfg = TRANSFER_PIYASA[piyasa]
    t0 = time.time()
    print(f"=== {cfg['ad'].upper()} TRANSFER TESTİ | {years} yıl | "
          f"{datetime.now():%Y-%m-%d %H:%M} ===")
    print("Davranışsal parametreler BIST'ten AYNEN alındı — hiçbiri bu piyasa "
          "için ayarlanmadı.")
    print()

    closes, vols, sektor = _transfer_panel(piyasa, years)

    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})
    endeks, dates = _endeks_penceresi(cfg["endeks"], years)
    dates = list(dates)
    reb = pd.date_range(dates[0], dates[-1], freq="MS")
    reb = [dates[pd.Index(dates).searchsorted(d)] for d in reb if d <= dates[-1]]

    # US yolu sektörü _sector_of'tan almaya DEVAM eder (yayımlanmış US sayıları
    # oradan üretildi, kaynağı değiştirmek onları sessizce oynatırdı); donmuş
    # sektör haritası olan piyasalar kendi haritasını kullanır.
    kur = {"min_tl_vol": cfg["hacim_esigi"]}
    if sektor:
        kur["sektor"] = sektor
    wbd = {}
    for d in reb:
        w = _pick_at(d, feats, ret_panel, kur=kur)
        if w:
            wbd[d] = w
    print(f"Panel {len(feats)} hisse | {len(wbd)}/{len(reb)} rebalans | "
          f"{dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d} ({time.time()-t0:.0f}s)")
    if sektor:
        _kap = len({sektor.get(t, "Diğer") for t in feats})
        print(f"Sektör haritası donmuş evrenden: {_kap} ayrı sektör "
              f"(tek sektöre düşseydi sector_cap portföyü 2'ye indirirdi)")
    print()

    global COST_RT
    _eski_cost = COST_RT
    COST_RT = cfg["maliyet"]
    try:
        nav_u = _walk(wbd, closes, dates)
        nav_m = _walk_managed(wbd, closes, dates, feats, dict(MGMT_VARSAYILAN))
        endeks_n = endeks.loc[dates] / endeks.loc[dates].iloc[0]
        rng_finals = []
        all_t = sorted(feats)
        for seed in range(RANDOM_SEEDS):
            rng = np.random.default_rng(seed)
            wr = {}
            for d in reb:
                elig = [t for t in all_t if feats[t]["close"].index.searchsorted(d) > 260]
                if len(elig) >= MAX_POS:
                    wr[d] = {t: 1.0 / MAX_POS for t in rng.choice(elig, MAX_POS, replace=False)}
            rng_finals.append(_walk(wr, closes, dates).iloc[-1])
    finally:
        COST_RT = _eski_cost

    def _oz(nav, ad):
        tot = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
        yrs = (nav.index[-1] - nav.index[0]).days / 365.25
        cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
        dd = ((nav / nav.cummax()) - 1).min() * 100
        r = nav.pct_change().dropna()
        sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        print(f"{ad:<26} toplam %{tot:>8.1f} | CAGR %{cagr:>6.1f} | "
              f"MaxDD %{dd:>6.1f} | Sharpe {sh:>5.2f}")
        return nav.iloc[-1]
    print("=" * 84)
    f_u = _oz(nav_u, f"YÖNETİMSİZ ({piyasa})")
    f_m = _oz(nav_m, f"YÖNETİLEN ({piyasa})")
    _oz(endeks_n, cfg["endeks_ad"])
    rf = np.array(rng_finals)
    print(f"{'Rastgele-6 (50 tohum)':<26} medyan son NAV {np.median(rf):.2f} | "
          f"%5-%95 [{np.percentile(rf,5):.2f}, {np.percentile(rf,95):.2f}]")
    print("=" * 84)
    # AMPİRİK persentil ile bayrağı AYNI şeyden türet. Önceki hâli ampirik
    # persentili yazdırıp bayrağı np.percentile'ın ARA DEĞERLİ eşiğinden
    # üretiyordu; 50 tohumla bu ikisi ayrışıyor ve çıktı kendi kendisiyle
    # çelişiyordu ("%94 persentil ✅ >%95"). Bir hakem bunu ilk bakışta görür.
    _yzd = (rf < f_u).mean() * 100
    print(f"Sistem son NAV {f_u:.2f} -> rastgele dağılımında %{_yzd:.0f} "
          f"persentil ({(rf < f_u).sum()}/{len(rf)} tohumu geçti) "
          f"{'✅ >=%95' if _yzd >= 95 else '⚠️ %95 ALTI'}")
    print()
    print(f"Not: {cfg['not']} Davranışsal parametreler")
    print("BIST'te seçildi ve burada DEĞİŞTİRİLMEDİ.")
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")


def us_transfer(years=8):
    """US transfer testi — ince sarmalayıcı, bkz. _transfer."""
    _transfer("US", years)


def dsr(years=8):
    """DEFLATED SHARPE RATIO — çoklu test düzeltmesi.

    Bailey & López de Prado (2014) ve Arnott/Harvey/Markowitz'in backtest
    protokolünün merkezî uyarısı: bir araştırmacı yüzlerce konfigürasyon
    dener ve EN İYİSİNİ raporlar. Deneme sayısı arttıkça, tesadüfen yüksek
    Sharpe bulma olasılığı hızla büyür. Dolayısıyla raporlanan Sharpe
    şişkindir ve deneme sayısına göre indirilmelidir.

    Bu projede ~700 parametre konfigürasyonu denendi. O yüzden "Sharpe 0.88"
    ham hâliyle savunulamaz; asıl soru şu: SIFIR kenarı olan bir stratejiyi
    bu kadar çok deneseydik, şans eseri 0.88 görme olasılığımız neydi?

    Hesap:
      SR0 = sqrt(V[SR]) × [(1-γ)Z⁻¹(1-1/N) + γZ⁻¹(1-1/(Ne))]
        = N denemede beklenen EN YÜKSEK Sharpe (kenar yokken)
      DSR = Φ[(SR - SR0)·√(T-1) / √(1 - γ₃·SR + (γ₄-1)/4·SR²)]
        = gerçek kenarın SR0'ı aştığına dair olasılık

    V[SR] uydurulmuyor: bu projede fiilen taranan konfigürasyonların
    Sharpe'larından ampirik olarak hesaplanıyor.
    """
    from scipy import stats as _st
    t0 = time.time()
    print(f"=== Deflated Sharpe Ratio | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})
    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
    reb = pd.date_range(dates[0], dates[-1], freq="MS")
    reb = [dates[pd.Index(dates).searchsorted(d)] for d in reb if d <= dates[-1]]
    wbd = {}
    for d in reb:
        w = _pick_at(d, feats, ret_panel)
        if w:
            wbd[d] = w

    # 1) Denenen konfigürasyonların Sharpe DAĞILIMI (V[SR] için ampirik temel)
    izgara = {
        "trailing_pct": [10, 15, 20, 25, 100],
        "dusus_freni":  [10, 15, 20, 100],
        "max_pozisyon": [20, 25, 30, 40],
        "takas_farki":  [0, 3, 6, 9, 12, 20],
        "skor_cikis":   [30, 35, 39, 43, 47, 52],
    }
    gunluk_sr = []
    for ad, degerler in izgara.items():
        for d0 in degerler:
            par = dict(MGMT_VARSAYILAN); par[ad] = float(d0)
            nav = _walk_managed(wbd, closes, dates, feats, par)
            r = _reel_seri(nav).pct_change().dropna()
            if r.std() > 0:
                gunluk_sr.append(float(r.mean() / r.std()))
    N = len(gunluk_sr)
    V_SR = float(np.var(gunluk_sr, ddof=1))
    print(f"Panel {len(feats)} hisse | {N} konfigürasyon tarandı ({time.time()-t0:.0f}s)")
    print(f"Deneme Sharpe'larının varyansı V[SR] = {V_SR:.3e} (günlük birimde)")
    print()

    # 2) En iyi (mevcut varsayılan) konfigürasyonun serisi
    nav = _walk_managed(wbd, closes, dates, feats, dict(MGMT_VARSAYILAN))
    r = _reel_seri(nav).pct_change().dropna()
    T = len(r)
    SR = float(r.mean() / r.std())
    g3 = float(_st.skew(r)); g4 = float(_st.kurtosis(r, fisher=False))
    print(f"Mevcut yapılandırma: günlük SR {SR:.4f} → yıllık {SR*np.sqrt(252):.2f}")
    print(f"Gözlem T={T} | çarpıklık {g3:+.2f} | basıklık {g4:.2f} (normal=3)")
    print()

    # 3) N denemede beklenen EN YÜKSEK Sharpe (kenar yokken)
    EULER = 0.5772156649015329
    for N_var in (N, 100, 300, 695):
        z1 = _st.norm.ppf(1 - 1.0 / N_var)
        z2 = _st.norm.ppf(1 - 1.0 / (N_var * np.e))
        SR0 = np.sqrt(V_SR) * ((1 - EULER) * z1 + EULER * z2)
        pay = (SR - SR0) * np.sqrt(T - 1)
        payda = np.sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR ** 2)
        DSR = float(_st.norm.cdf(pay / payda))
        etiket = "bu koşuda" if N_var == N else "tüm oturum" if N_var == 695 else ""
        print(f"  N={N_var:<5} SR0(beklenen en yüksek) = {SR0*np.sqrt(252):>5.2f} yıllık | "
              f"DSR = {DSR:.4f} {'✅ anlamlı' if DSR > 0.95 else '⚠️ ANLAMSIZ'} {etiket}")
    print()
    z0 = (SR - 0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR ** 2)
    print(f"Karşılaştırma — düzeltmesiz PSR (SR*=0): {_st.norm.cdf(z0):.4f}")
    print()
    print("Okuma: DSR > 0.95 ise Sharpe, denenen konfigürasyon sayısı hesaba")
    print("katıldıktan SONRA da anlamlıdır. Altındaysa raporlanan Sharpe,")
    print("çoklu testin ürettiği şansla açıklanabilir demektir.")
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")


def sanstest(years=8, katman=5):
    """SİSTEM vs RASTGELE — her dönemde ayrı ayrı.

    Projenin en güçlü sonucu "sistemin son NAV'ı rastgele-6 dağılımının %95
    persentilinin üstünde" idi. Ama bu TEK pencereden geliyordu ve "o dönem
    şanslıydınız" itirazına açıktı. Burada pencere dönemlere bölünür ve her
    dönemde sistem, aynı dönemde çekilmiş 50 rastgele portföyle kıyaslanır.

    Her dönem kendi içinde 100'e tabanlanır; portföy dönem başında sıfırlanmaz
    (sürekli koşudan dilim alınır), böylece giriş maliyeti tekrarlanmaz.
    """
    t0 = time.time()
    print(f"=== Sistem vs Rastgele, dönem dönem | {years} yıl | "
          f"{datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})
    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
    reb = pd.date_range(dates[0], dates[-1], freq="MS")
    reb = [dates[pd.Index(dates).searchsorted(d)] for d in reb if d <= dates[-1]]
    wbd = {}
    for d in reb:
        w = _pick_at(d, feats, ret_panel)
        if w:
            wbd[d] = w
    nav_sys = _walk(wbd, closes, dates)

    all_t = sorted(feats)
    rng_navs = []
    for seed in range(RANDOM_SEEDS):
        rng = np.random.default_rng(seed)
        wbd_r = {}
        for d in reb:
            elig = [t for t in all_t if feats[t]["close"].index.searchsorted(d) > 260]
            if len(elig) >= MAX_POS:
                sel = rng.choice(elig, MAX_POS, replace=False)
                wbd_r[d] = {t: 1.0 / MAX_POS for t in sel}
        rng_navs.append(_walk(wbd_r, closes, dates))
    print(f"Panel {len(feats)} hisse | {RANDOM_SEEDS} rastgele tohum "
          f"({time.time()-t0:.0f}s)")
    print()

    n_all = len(dates)
    sinir = [int(n_all * i / (katman + 1)) for i in range(katman + 2)]
    print(f"{'dönem':<22} {'sistem':>9} {'rastgele medyan':>16} "
          f"{'%95 dilim':>11} {'persentil':>11}  sonuç")
    print("-" * 84)
    ustte = 0
    for k in range(katman + 1):
        b, e = sinir[k], sinir[k + 1] - 1
        sg = (nav_sys.iloc[e] / nav_sys.iloc[b] - 1) * 100
        rg = np.array([(n.iloc[e] / n.iloc[b] - 1) * 100 for n in rng_navs])
        pct = float((rg < sg).mean() * 100)
        p95 = float(np.percentile(rg, 95))
        if sg > p95:
            ustte += 1
        print(f"{dates[b]:%Y-%m}→{dates[e]:%Y-%m}      %{sg:>7.1f} %{np.median(rg):>14.1f} "
              f"%{p95:>9.1f} {pct:>10.0f}%  {'%95 ÜSTÜ' if sg > p95 else ''}")
    print("-" * 84)
    print(f"Sistem {ustte}/{katman + 1} dönemde rastgelenin %95 persentilinin üstünde.")
    print("Ölçüt: getiri > np.percentile(rastgele, 95) (ara değerli eşik). Ampirik")
    print("persentil kolonu 50 tohumla 2 puan adımlıdır, bu yüzden %94 yazan bir")
    print("satır bu eşiği yine de geçebilir — çelişki değil, iki farklı tanım.")
    print("Not: dönemler bağımsız değil (aynı sürekli koşunun dilimleri) ve")
    print("rastgele portföyler her dönemde aynı tohumlarla üretiliyor — yani")
    print("bu bir kesin anlamlılık testi değil, tutarlılık göstergesidir.")
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")


def kurulum(years=8, katman=5):
    """PORTFÖY KURULUM parametrelerini tarar — hiçbiri test edilmemişti.

    MAX_POS taraması bu varsayımların ne kadar körü körüne taşındığını
    gösterdi. Buradakiler de aynı sınıftan: score_min (aday eşiği),
    sector_cap (aynı sektörden kaç hisse), corr_limit (korelasyon elemesi),
    min_tl_vol (likidite tabanı). Hepsi "makul görünen" sayılardı.

    Kabul, walkforward ile aynı: katmanların ÇOĞUNDA ve >%5 farkla kazanmalı.
    """
    t0 = time.time()
    print(f"=== Kurulum Parametreleri | {years} yıl | {katman} katman | "
          f"{datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})
    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
    reb = pd.date_range(dates[0], dates[-1], freq="MS")
    reb = [dates[pd.Index(dates).searchsorted(d)] for d in reb if d <= dates[-1]]
    xu_n = xu.loc[dates] / xu.loc[dates].iloc[0]
    n_all = len(dates)
    sinir = [int(n_all * i / (katman + 1)) for i in range(katman + 2)]
    print(f"Panel {len(feats)} hisse ({time.time()-t0:.0f}s)")
    print()

    taranacak = {
        "score_min":  ([40, 45, 50, 55, 60], SCORE_MIN),
        "sector_cap": ([1, 2, 3], SECTOR_CAP),
        "corr_limit": ([0.70, 0.80, 0.85, 0.95, 1.01], CORR_LIMIT),
        "min_tl_vol": ([10e6, 20e6, 50e6, 100e6], MIN_TL_VOL),
    }
    for ad, (degerler, varsayilan) in taranacak.items():
        print(f"--- {ad} (varsayılan {varsayilan:g}) ---")
        wbd_c = {}
        for d0 in degerler:
            w_all = {}
            for d in reb:
                w = _pick_at(d, feats, ret_panel, kur={ad: d0})
                if w:
                    w_all[d] = w
            wbd_c[d0] = w_all
            sm = _stats(_walk_managed(w_all, closes, dates, feats,
                                      dict(MGMT_VARSAYILAN)), xu_n, "_")
            im = " *" if d0 == varsayilan else ""
            print(f"  {ad}={d0:<10g} reel CAGR %{sm['reel_yillik']:>6.1f} | "
                  f"MaxDD %{sm['maxdd']:>6.1f} | Sharpe {sm['sharpe']:>5.2f} | "
                  f"{len(w_all)} rebalans{im}")
        kazanma, secimler = 0, []
        for k in range(katman):
            eg, te = dates[:sinir[k+1]], dates[sinir[k+1]:sinir[k+2]]
            en_iyi, en_g = None, -1e18
            for d0 in degerler:
                w_eg = {d: w for d, w in wbd_c[d0].items() if d <= eg[-1]}
                if not w_eg:
                    continue
                g = _ozet(_walk_managed(w_eg, closes, eg, feats, dict(MGMT_VARSAYILAN)))[0]
                if g > en_g:
                    en_iyi, en_g = d0, g
            if en_iyi is None:
                continue
            def _te(d0):
                w = {d: x for d, x in wbd_c[d0].items() if eg[-1] < d <= te[-1]}
                return _ozet(_walk_managed(w, closes, te, feats,
                                           dict(MGMT_VARSAYILAN)))[0] if w else float("nan")
            t_v, t_s = _te(varsayilan), _te(en_iyi)
            fark = (t_s - t_v) / max(abs(t_v), 1e-9) * 100
            if en_iyi != varsayilan and fark > 5.0:
                kazanma += 1
            secimler.append(en_iyi)
        onay = len(secimler) >= 3 and kazanma > len(secimler) / 2
        print(f"  → eğitimin seçimleri {secimler} | {kazanma}/{len(secimler)} katmanda tuttu")
        print(f"  KARAR: {'DEĞİŞTİR' if onay else 'DEĞİŞTİRME'}")
        print()
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")


def slippage(years=8):
    """PİYASA ETKİSİ: kenar hangi portföy büyüklüğüne kadar ayakta kalıyor?

    Backtest bugüne dek işlem maliyetini düz yüzde saydı ve işlem BÜYÜKLÜĞÜNÜ
    hiç dikkate almadı. Portföy ince hacimli isimler tuttuğu için bu, "gerçek
    parayla uygulanabilir mi" sorusunun cevabını gizliyordu.

    Burada portföy büyüklüğü artırılır ve her seviyede kenarın ne kadarının
    hayatta kaldığı ölçülür. Sonuç, ölçek kapasitesinin büyüklük mertebesi
    tahminidir — kesin bir eşik değil, ama "sınırsız" olmadığını gösterir.
    """
    t0 = time.time()
    print(f"=== Piyasa Etkisi / Ölçek Kapasitesi | {years} yıl | "
          f"{datetime.now():%Y-%m-%d %H:%M} ===")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})
    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
    reb = pd.date_range(dates[0], dates[-1], freq="MS")
    reb = [dates[pd.Index(dates).searchsorted(d)] for d in reb if d <= dates[-1]]
    wbd = {}
    for d in reb:
        w = _pick_at(d, feats, ret_panel)
        if w:
            wbd[d] = w
    xu_n = xu.loc[dates] / xu.loc[dates].iloc[0]
    print(f"Panel {len(feats)} hisse | {len(wbd)} rebalans | SLIP_K={SLIP_K} "
          f"({time.time()-t0:.0f}s)")
    print()

    print(f"{'portföy':>12} {'yönetimsiz reel':>17} {'yönetimli reel':>16} "
          f"{'yön. Sharpe':>12} {'kayıp':>9}")
    print("-" * 72)
    taban_m = None
    for tl in (0, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000):
        su = _stats(_walk(wbd, closes, dates, feats, tl), xu_n, "_")
        sm = _stats(_walk_managed(wbd, closes, dates, feats,
                                  dict(MGMT_VARSAYILAN), portfoy_tl=tl), xu_n, "_")
        if taban_m is None:
            taban_m = sm["reel_yillik"]
        ad = "etki YOK" if tl == 0 else f"{tl/1e6:,.1f}M TL"
        kayip = "" if tl == 0 else f"{sm['reel_yillik']-taban_m:+8.1f}p"
        print(f"{ad:>12} {su['reel_yillik']:>16.1f}% {sm['reel_yillik']:>15.1f}% "
              f"{sm['sharpe']:>12.2f} {kayip:>9}")
    print()
    print("Okuma: 'etki YOK' satırı bugüne kadar raporlanan rakamdır. Aşağı")
    print("indikçe kenarın ne kadarının işlem maliyetine gittiği görülür.")
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")


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

    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
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
        "skor_cikis":   [35, 39, 43, 47, 52],
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

    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
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


# DEVİR ÖLÇÜMÜ (2026-09-27, açık madde 1.4) — canlıdaki yönetim, test edilen yönetim mi?
#
# Canlı ölçüm: yönetilen BIST portföylerinde medyan elde tutma skor_cikis
# 43→39 sonrasında da ~8 iş günü, haftalık tek yön devir %40-50. En büyük
# çıkış nedeni sabit ATR stopu (29 Ağu sonrası 516 satışın 180'i). Oysa
# _walk_managed'de sabit ATR stopu YOK ve alım yalnız ay başında yapılıyor;
# canlı sistem ise boşalan nakdi her gün yeniden yatırıyor. Yani doğrulanmış
# sayılar (takas_farki, skor_cikis) canlıda koşan sistemin değil, ondan iki
# mekanizma eksik bir sistemin sayıları.
#
# ÖN KAYITLI HİPOTEZ: bu iki mekanizma devri artırır ve getiriyi düşürür.
# Dört yapılandırma, hepsi önceden sabit — tarama DEĞİL, canlı sistemin
# kendisini ölçmek: taban / +ATR stop (3.5, orta-uzun vadenin canlı değeri) /
# +günlük alım / ikisi birden (canlıya en yakın).
def devir(years=8):
    t0 = time.time()
    print(f"=== DEVİR ÖLÇÜMÜ | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    print("Hipotez: sabit ATR stopu + günlük alım devri artırır, getiriyi düşürür.\n")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})
    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
    reb = pd.date_range(dates[0], dates[-1], freq="MS")
    reb = [dates[pd.Index(dates).searchsorted(d)] for d in reb if d <= dates[-1]]
    wbd = {d: w for d in reb if (w := _pick_at(d, feats, ret_panel))}
    xu_n = xu.loc[dates] / xu.loc[dates].iloc[0]
    yil = (dates[-1] - dates[0]).days / 365.25

    yap = [("taban (test edilen)", {}),
           ("+ ATR stop 3.5", {"atr_stop": 3.5}),
           ("+ günlük alım", {"gunluk_alim": True}),
           ("+ ikisi (canlıya yakın)", {"atr_stop": 3.5, "gunluk_alim": True})]
    satir = []
    print("=" * 118)
    for ad, ek in yap:
        nav = _walk_managed(wbd, closes, dates, feats, {**MGMT_VARSAYILAN, **ek})
        st = _stats(nav, xu_n, ad)
        tut = np.array(nav.attrs["tutma"])
        satir.append((ad, st, nav.attrs["devir"] / yil * 100,
                      float(np.median(tut)) if len(tut) else float("nan"),
                      len(tut) / yil, nav.attrs["maliyet"] * 100, nav.iloc[-1]))
    print("=" * 118)
    print(f"\n{'Yapılandırma':<26} {'son NAV':>8} {'reel/yıl':>9} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'devir/yıl':>10} {'tutma(gün)':>11} {'çıkış/yıl':>10} {'maliyet':>8}")
    for ad, st, dv, tm, cy, ml, fn in satir:
        print(f"{ad:<26} {fn:>8.2f} {st['reel_yillik']:>+8.1f}% {st['maxdd']:>6.1f}% "
              f"{st['sharpe']:>7.2f} {dv:>9.0f}% {tm:>11.0f} {cy:>10.0f} {ml:>7.1f}%")
    print("\nNot: canlıda ayrıca Ayı rejim kapısı, 5 gün min. tutma, kısmi kâr alma ve")
    print("günlük takas var; bunlar modellenmedi. 'ikisi' satırı canlıya YAKIN, eşit değil.")
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")


# SAĞLAMLIK TESTİ (2026-10-07) — ders notu kriterleri skorlamada işe yarar mı?
#
# Ön kayıt: bilanco_esikler.json → "_on_kayit" (testten ÖNCE yazıldı). İki
# deneme, başka varyant yok:
#   A (filtre): sağlamlık skoru < 0.50 olan hisse aday listesine girmez
#   B (puan)  : sıralama = 0.9×teknik + 10×sağlamlık + 0.4×3A momentum
# Başarı (birincil iz YÖNETİLEN): temele göre (a) reel yıllık getiri yüksek VE
# 7 takvim yılının ≥4'ünde getiri yüksek, YA DA (b) MaxDD ≥5 puan iyi VE reel
# yıllık getiri en fazla 2 puan düşük. Kriz dönemleri ve yönetimsiz iz tanıdır.
def saglik_testi(years=8):
    import bilanco as bl
    from bist_analyzer import _sector_of
    t0 = time.time()
    print(f"=== SAĞLAMLIK TESTİ | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    print("Ön kayıt: bilanco_esikler.json _on_kayit (A filtre <0.50, B puan 10 ağırlık)\n")
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})
    xu, dates = _endeks_penceresi("XU100.IS", years)
    dates = list(dates)
    reb = pd.date_range(dates[0], dates[-1], freq="MS")
    reb = [dates[pd.Index(dates).searchsorted(d)] for d in reb if d <= dates[-1]]

    tum = bl.tum_oranlar()
    sh = {t: _sector_of(t) for t in tum["ticker"].unique()}
    print(f"Mali tablo: {tum['ticker'].nunique()} hisse, {len(tum)} hisse-çeyrek ({time.time()-t0:.0f}s)")

    wbd = {"temel": {}, "A_filtre": {}, "B_puan": {}}
    hariç_say, skorlu_say, temelde_zayif = [], [], 0
    for d in reb:
        s = bl.saglik_anlik(tum, d, sh)
        haric = {t for t, v in s.items() if np.isfinite(v) and v < 0.50}
        hariç_say.append(len(haric))
        skorlu_say.append(int(s.notna().sum()))
        for ad, kur in (("temel", None), ("A_filtre", {"haric": haric}),
                        ("B_puan", {"saglik": s.to_dict()})):
            w = _pick_at(d, feats, ret_panel, kur=kur)
            if w:
                wbd[ad][d] = w
        temelde_zayif += sum(1 for t in wbd["temel"].get(d, {}) if t in haric)
    n_temel = sum(len(w) for w in wbd["temel"].values())
    print(f"Ayda ortalama {np.mean(skorlu_say):.0f} skorlu hisse, {np.mean(hariç_say):.0f}'i "
          f"filtrede eleniyor | temel seçimlerin %{100 * temelde_zayif / max(n_temel, 1):.0f}'i "
          f"zayıf bilançolu ({time.time()-t0:.0f}s)\n")

    xu_n = xu.loc[dates] / xu.loc[dates].iloc[0]
    sonuc = {}
    print("=" * 118)
    for ad in wbd:
        for iz in ("yönetimsiz", "yönetilen"):
            nav = (_walk(wbd[ad], closes, dates) if iz == "yönetimsiz" else
                   _walk_managed(wbd[ad], closes, dates, feats, dict(MGMT_VARSAYILAN)))
            st = _stats(nav, xu_n, f"{ad} / {iz}")
            sonuc[(ad, iz)] = (nav, st)
    print("=" * 118)

    # Rastgele-6 yeri (yönetimsiz iz) — run ile aynı çekiliş
    all_t = sorted(feats)
    rf = []
    for seed in range(RANDOM_SEEDS):
        rng = np.random.default_rng(seed)
        wr = {}
        for d in reb:
            elig = [t for t in all_t if feats[t]["close"].index.searchsorted(d) > 260]
            if len(elig) >= MAX_POS:
                wr[d] = {t: 1 / MAX_POS for t in rng.choice(elig, MAX_POS, replace=False)}
        rf.append(_walk(wr, closes, dates).iloc[-1])
    rf = np.array(rf)

    def yillik(nav):
        return nav.groupby(nav.index.year).agg(lambda x: x.iloc[-1] / x.iloc[0] - 1) * 100

    def donem(nav, a, b):
        x = nav[(nav.index >= a) & (nav.index <= b)]
        return (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else np.nan

    print(f"\n{'':22}{'reel/yıl':>9}{'MaxDD':>8}{'Sharpe':>8}{'rastgele %':>11}"
          f"{'2018-19 kriz':>13}{'2020 Covid':>11}")
    for (ad, iz), (nav, st) in sonuc.items():
        print(f"{ad + ' / ' + iz:22}{st['reel_yillik']:>+8.1f}%{st['maxdd']:>7.1f}%{st['sharpe']:>8.2f}"
              f"{(rf < nav.iloc[-1]).mean() * 100:>10.0f}%"
              f"{donem(nav, '2018-08-01', '2019-12-31'):>+12.1f}%{donem(nav, '2020-01-01', '2020-06-30'):>+10.1f}%")

    print("\nYÖNETİLEN iz, takvim yılı getirileri (%):")
    yt = pd.DataFrame({ad: yillik(sonuc[(ad, "yönetilen")][0]) for ad in wbd})
    print(yt.round(1).to_string())

    print("\n--- ÖN KAYITLI KARAR (birincil iz: yönetilen) ---")
    tb = sonuc[("temel", "yönetilen")][1]
    for ad in ("A_filtre", "B_puan"):
        st = sonuc[(ad, "yönetilen")][1]
        iyi_yil = int((yt[ad] > yt["temel"]).sum())
        a = st["reel_yillik"] > tb["reel_yillik"] and iyi_yil >= 4
        b = st["maxdd"] >= tb["maxdd"] + 5 and st["reel_yillik"] >= tb["reel_yillik"] - 2
        print(f"{ad}: reel {st['reel_yillik']:+.1f}% vs {tb['reel_yillik']:+.1f}% | daha iyi yıl "
              f"{iyi_yil}/{len(yt)} | MaxDD {st['maxdd']:.1f}% vs {tb['maxdd']:.1f}% → "
              f"{'BAŞARILI (' + ('a' if a else 'b') + ')' if (a or b) else 'BAŞARISIZ'}")
    print(f"\nTamamlandı ({time.time()-t0:.0f}s). Deneme sayısı 2 (A, B); başka varyant denenmez.")


# MOMENTUM KONTROLÜ (2026-09-27) — "sistem bir momentum aracı mı?"
#
# Dört piyasa-dönemin hepsi "orada o zaman momentum ödedi mi?" sorusuyla aynı
# hizada: US 2018-24 kazanıyor, UK ve BIST 2018-24 kaybediyor, BIST 2024-26
# (kasa) kazanıyordu. Endeks ve rastgele-6 kontrolleri bunu ayıramaz; ders
# kitabı momentum portföyü ayırır.
#
# ÖN KAYITLI HİPOTEZ (kod koşmadan önce yazıldı, CLAUDE.md'de de):
#   H: sistem bir momentum faktör aracıdır. Öyleyse her üç piyasada
#     (1) sistem ile MOM-6'nın aylık getiri korelasyonu >= 0.6,
#     (2) MOM-6'nın rastgele-6 persentil SIRALAMASI sisteminkiyle aynı,
#     (3) sistemin MOM-6'ya göre alfası anlamsız (|t| < 2).
#   Çürütme: herhangi bir piyasada anlamlı POZİTİF alfa, ya da düşük korelasyon.
#
# Kontroller STRATEJİ DEĞİLDİR: hangisi daha iyi çıkarsa çıksın benimsenmez;
# benimsemek bir deneme daha demektir ve DSR'yi düşürür.
#
# İki kontrol, rolleri önceden belli:
#   MOM-6 (BİRİNCİL)  ders kitabı 12-1 momentum (Jegadeesh-Titman): son 12
#                     ayın getirisi, son 1 ay hariç. Yalnız likidite filtresi,
#                     en iyi 6, eşit ağırlık. Ayarlanmış hiçbir şey yok.
#   MOM-kurulum       sistemin kendi makinesi (likidite, SMA200, sektör,
#                     korelasyon, ATR ağırlık) ama sıralama 12-1 momentum.
#                     Sistemle farkı YALNIZ skor fonksiyonudur — "skorumuz ders
#                     kitabı momentumdan fazlasını biliyor mu?" sorusunun cevabı.
def _mom_12_1(cl, ix):
    """12-1 ay momentumu (252 → 21 işlem günü önce). Yetersiz veride None."""
    if ix < 252:
        return None
    p0, p1 = float(cl.iloc[ix - 252]), float(cl.iloc[ix - 21])
    if not (p0 > 0 and p1 > 0) or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return p1 / p0 - 1


def _mom_ders_kitabi(date, feats, min_vol, n=None):
    """MOM-6: likidite filtresi + 12-1 momentum sırası + eşit ağırlık. Başka hiçbir şey."""
    n = n or MAX_POS
    aday = []
    for t, f in feats.items():
        cl = f["close"]
        ix = cl.index.searchsorted(date, side="right") - 1
        if ix < 260 or (date - cl.index[ix]).days > 7:
            continue
        if float(f["tlvol"].iloc[ix] or 0) < min_vol:
            continue
        m = _mom_12_1(cl, ix)
        if m is not None:
            aday.append((t, m))
    aday.sort(key=lambda x: -x[1])
    sec = [t for t, _ in aday[:n]]
    return {t: 1.0 / len(sec) for t in sec} if sec else {}


def _aylik(nav, reb):
    """NAV'ı rebalans günlerinde örnekle → ay-ay getiri (çakışmasız)."""
    noktalar = [d for d in reb if d in nav.index] + [nav.index[-1]]
    s = nav.loc[sorted(set(noktalar))]
    return s.pct_change().dropna()


def _alfa(y, x):
    """OLS y = a + b·x → (a, t_a, b). Aylık, çakışmasız gözlem."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    dof = max(len(y) - 2, 1)
    s2 = (e @ e) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta[0], beta[0] / np.sqrt(cov[0, 0]), beta[1]


def momentum(years=8):
    global COST_RT
    t0 = time.time()
    print(f"=== MOMENTUM KONTROLÜ | {years} yıl | {datetime.now():%Y-%m-%d %H:%M} ===")
    print("Ön kayıtlı hipotez: sistem bir momentum aracıdır → korelasyon >= 0.6,")
    print("persentil sıralaması aynı, MOM-6'ya göre alfa |t| < 2 (üç piyasada da).")
    print()

    ozet = []
    for piyasa in ("BIST", "US", "UK"):
        print(f"--- {piyasa} " + "-" * 70)
        if piyasa == "BIST":
            closes, vols = _panel_hazirla(years)
            endeks_sym, maliyet, min_vol, kur = "XU100.IS", COST_RT, MIN_TL_VOL, {}
        else:
            cfg = TRANSFER_PIYASA[piyasa]
            closes, vols, sektor = _transfer_panel(piyasa, years)
            endeks_sym, maliyet, min_vol = cfg["endeks"], cfg["maliyet"], cfg["hacim_esigi"]
            kur = {"min_tl_vol": min_vol}
            if sektor:
                kur["sektor"] = sektor   # _transfer ile birebir aynı kural

        feats = _precompute(closes, vols)
        ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})
        _endeks, dates = _endeks_penceresi(endeks_sym, years)
        dates = list(dates)
        reb = pd.date_range(dates[0], dates[-1], freq="MS")
        reb = [dates[pd.Index(dates).searchsorted(d)] for d in reb if d <= dates[-1]]

        w_sis, w_mom, w_mk = {}, {}, {}
        for d in reb:
            a = _pick_at(d, feats, ret_panel, kur=kur)
            b = _mom_ders_kitabi(d, feats, min_vol)
            c = _pick_at(d, feats, ret_panel, kur={**kur, "sira": _mom_12_1})
            if a: w_sis[d] = a
            if b: w_mom[d] = b
            if c: w_mk[d] = c

        _eski = COST_RT
        COST_RT = maliyet
        try:
            nav_s = _walk(w_sis, closes, dates)
            nav_m = _walk(w_mom, closes, dates)
            nav_k = _walk(w_mk, closes, dates)
            rng_navs = []
            all_t = sorted(feats)
            for seed in range(RANDOM_SEEDS):   # _transfer/run ile birebir aynı çekiliş
                rng = np.random.default_rng(seed)
                wr = {}
                for d in reb:
                    elig = [t for t in all_t if feats[t]["close"].index.searchsorted(d) > 260]
                    if len(elig) >= MAX_POS:
                        wr[d] = {t: 1.0 / MAX_POS for t in rng.choice(elig, MAX_POS, replace=False)}
                rng_navs.append(_walk(wr, closes, dates))
        finally:
            COST_RT = _eski
        rf = np.array([n.iloc[-1] for n in rng_navs])
        nav_r = pd.concat(rng_navs, axis=1).mean(axis=1)

        def _satir(nav, ad):
            yrs = (nav.index[-1] - nav.index[0]).days / 365.25
            cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
            dd = ((nav / nav.cummax()) - 1).min() * 100
            yzd = (rf < nav.iloc[-1]).mean() * 100
            print(f"  {ad:<22} son NAV {nav.iloc[-1]:>7.2f} | CAGR %{cagr:>6.1f} | "
                  f"MaxDD %{dd:>6.1f} | rastgele içinde %{yzd:>3.0f} persentil")
            return yzd
        p_s = _satir(nav_s, "SİSTEM (yönetimsiz)")
        p_m = _satir(nav_m, "MOM-6 (ders kitabı)")
        p_k = _satir(nav_k, "MOM-kurulum")
        print(f"  {'Rastgele-6 (50 tohum)':<22} medyan {np.median(rf):>7.2f} | "
              f"%5-%95 [{np.percentile(rf, 5):.2f}, {np.percentile(rf, 95):.2f}]")

        m_s, m_m, m_k, m_r = (_aylik(n, reb) for n in (nav_s, nav_m, nav_k, nav_r))
        ix = m_s.index.intersection(m_m.index).intersection(m_r.index).intersection(m_k.index)
        m_s, m_m, m_k, m_r = m_s[ix], m_m[ix], m_k[ix], m_r[ix]
        kor = float(np.corrcoef(m_s, m_m)[0, 1])
        kor_k = float(np.corrcoef(m_s, m_k)[0, 1])
        # Rastgele-6 ortalaması evrenin eşit ağırlıklı "piyasası" yerine geçer;
        # iki taraftan da düşülünce kalan, momentuma maruziyetin ötesindeki seçim.
        a, ta, b = _alfa(m_s - m_r, m_m - m_r)
        a_k, ta_k, b_k = _alfa(m_s - m_k, m_m - m_r)   # yalnız skor fonksiyonunun katkısı
        ortak = np.mean([len(set(w_sis[d]) & set(w_mom.get(d, {}))) / len(w_sis[d])
                         for d in w_sis])
        print(f"  aylık korelasyon: sistem~MOM-6 {kor:+.2f} | sistem~MOM-kurulum {kor_k:+.2f} "
              f"| {len(ix)} ay")
        print(f"  sistem-rastgele = a + b·(MOM-6-rastgele): a %{a*100:+.2f}/ay "
              f"(t={ta:+.2f}), b={b:.2f}")
        print(f"  sistem-MOM-kurulum farkı, MOM-6'ya göre: a %{a_k*100:+.2f}/ay (t={ta_k:+.2f})")
        print(f"  ortak hisse: sistemin seçtiklerinin ortalama %{ortak*100:.0f}'i MOM-6'da da var")
        print()
        ozet.append((piyasa, p_s, p_m, p_k, kor, a, ta, b, a_k, ta_k, ortak))

    print("=" * 96)
    print(f"{'Piyasa':<6} {'sis %':>6} {'MOM6 %':>7} {'MOMk %':>7} {'kor':>6} "
          f"{'alfa/ay':>8} {'t':>6} {'beta':>5} {'skor alfa':>10} {'t':>6} {'ortak':>6}")
    for p, ps, pm, pk, k, a, ta, b, ak, tak, o in ozet:
        print(f"{p:<6} {ps:>6.0f} {pm:>7.0f} {pk:>7.0f} {k:>+6.2f} {a*100:>+7.2f}% "
              f"{ta:>+6.2f} {b:>5.2f} {ak*100:>+9.2f}% {tak:>+6.2f} {o*100:>5.0f}%")
    print("=" * 96)
    k1 = all(r[4] >= 0.6 for r in ozet)
    sira_s = [r[0] for r in sorted(ozet, key=lambda r: -r[1])]
    sira_m = [r[0] for r in sorted(ozet, key=lambda r: -r[2])]
    k2 = sira_s == sira_m
    k3 = all(abs(r[6]) < 2 for r in ozet)
    pozitif = [r[0] for r in ozet if r[6] >= 2]
    print(f"(1) korelasyon >= 0.6 her piyasada : {'EVET' if k1 else 'HAYIR'}")
    print(f"(2) persentil sırası aynı          : {'EVET' if k2 else 'HAYIR'} "
          f"(sistem {' > '.join(sira_s)} | MOM-6 {' > '.join(sira_m)})")
    print(f"(3) alfa |t| < 2 her piyasada      : {'EVET' if k3 else 'HAYIR'}")
    if pozitif:
        print(f"HİPOTEZ ÇÜRÜDÜ: anlamlı pozitif alfa -> {', '.join(pozitif)}")
    elif k1 and k2 and k3:
        print("HİPOTEZ TUTTU: sistem, ölçülebildiği kadarıyla bir momentum aracı.")
    else:
        print("KARIŞIK: hipotezin bir kısmı tuttu — tabloya bak, tek cümleye indirme.")
    print(f"Tamamlandı ({time.time()-t0:.0f}s)")


def kasa_durum():
    """Kasanın durumu: nerede kesiliyor, ne kilitli, daha önce açıldı mı."""
    meta = _kasa_meta()
    if not meta.get("kesim"):
        print(f"{KASA_DOSYA} yok → kasa KURULU DEĞİL, tüm veri geliştirmeye açık.")
        return
    kes = pd.Timestamp(meta["kesim"])
    gecmis = meta.get("acilis_kaydi", [])
    print("=" * 78)
    print("VERİ KASASI")
    print("=" * 78)
    print(f"Kesim tarihi   : {kes:%Y-%m-%d}")
    print(f"Kuruldu        : {meta.get('kuruldu', '?')}")
    print(f"Rezerv         : son {meta.get('rezerv_yil', 2)} yıl "
          f"({kes:%Y-%m-%d} → bugün) — hiçbir komut buraya bakmaz")
    print(f"Durum          : {'AÇILDI' if gecmis else 'KİLİTLİ'}"
          f"{' (' + str(len(gecmis)) + ' açılış)' if gecmis else ''}")
    for a in gecmis:
        print(f"   • {a.get('tarih')} | {a.get('pencere')} | {a.get('gerekce', '')}")
    # Pencere İKİ UCUNDAN DA kesime çıpalı (bkz. _endeks_penceresi): takvim
    # ilerledikçe kaymasın diye. `8` = kesim − (8 − rezerv) yıl → kesim.
    rez = float(meta.get("rezerv_yil", 2))
    bas = kes - pd.DateOffset(months=int(round(max(1.0, 8 - rez) * 12)))
    print(f"\nGeliştirme penceresi: `8` → {bas:%Y-%m-%d} → {kes:%Y-%m-%d} "
          f"({(kes - bas).days / 365.25:.1f} yıl), takvimden bağımsız sabit.")
    print(f"Kural: başlangıç = kesim − (yıl − {rez:.0f}) yıl. Bu, yayımlanmış 8 yıllık")
    print("sonuçların başlangıç tarihiyle aynıdır; aynı komut her gün aynı sayıyı verir.")
    print("\nAçmak için:  python pipeline_backtest.py kasa-ac 8 EVET")


def _saglik_kasa_kolu(reb, feats, ret_panel, closes, dates, xu_n, su):
    """H2 (bilanco_esikler.json _on_kayit_2), kasa açılışında BİR KEZ koşar.

    A filtresi yönetimsiz izde düşüşü azaltıyor mu? Kural, skor ve eşik
    _on_kayit ile aynı; karar satırları ön kayıttan birebir. `su` temel
    yönetimsiz izin _stats sonucudur (aynı koşudan).
    """
    import bilanco as bl
    from bist_analyzer import _sector_of
    tum = bl.tum_oranlar()
    sh = {t: _sector_of(t) for t in tum["ticker"].unique()}
    wbd, haric_say = {}, []
    for d in reb:
        s = bl.saglik_anlik(tum, d, sh)
        haric = {t for t, v in s.items() if np.isfinite(v) and v < 0.50}
        haric_say.append(len(haric))
        w = _pick_at(d, feats, ret_panel, kur={"haric": haric})
        if w:
            wbd[d] = w
    print(f"\n--- H2: bilanço filtresi, yönetimsiz iz (ön kayıt _on_kayit_2) ---")
    print(f"Ayda ortalama {np.mean(haric_say):.0f} hisse filtrede eleniyor | "
          f"{len(wbd)}/{len(reb)} rebalans")
    st = _stats(_walk(wbd, closes, dates), xu_n, "A_filtre / yönetimsiz")
    if su["maxdd"] > -20:
        karar = "ÖLÇÜLEMEDİ"
    elif st["maxdd"] >= su["maxdd"] + 5 and st["reel_yillik"] >= su["reel_yillik"] - 2:
        karar = "BAŞARILI"
    else:
        karar = "BAŞARISIZ"
    print(f"H2: MaxDD {st['maxdd']:.1f}% vs {su['maxdd']:.1f}% | reel/yıl "
          f"{st['reel_yillik']:+.1f}% vs {su['reel_yillik']:+.1f}% → {karar}")
    return {"karar": karar, "a_filtre_yonetimsiz": st,
            "ort_elenen": float(np.mean(haric_say))}


def kasa_ac(years=8, onay=""):
    """REZERV DÖNEM TESTİ — kasayı açar. Tek seferlik ve geri alınamaz.

    Ekim sunumunun tek örneklem-dışı sayısı burada üretilir: donmuş
    parametrelerle, hiçbir kararın etkilemediği son 2 yıl. Bu yüzden komut
    açık onay ister ve açılışı dosyaya yazar — "kaç kez bakıldı" sorusunun
    cevabı kayıtta durmalı, hafızada değil.

    Açıldıktan SONRA parametre değiştirmek sonucu geçersiz kılar: o noktadan
    itibaren rezerv dönem de sıradan bir eğitim penceresidir.
    """
    global _KASA_ACIK
    import json
    meta = _kasa_meta()
    if not meta.get("kesim"):
        print(f"{KASA_DOSYA} yok — açılacak kasa yok.")
        return
    kes = pd.Timestamp(meta["kesim"])
    gecmis = meta.get("acilis_kaydi", [])
    print("=" * 78)
    print("VERİ KASASI — REZERV DÖNEM TESTİ (tek seferlik, geri alınamaz)")
    print("=" * 78)
    print(f"Kesim {kes:%Y-%m-%d} | kuruldu {meta.get('kuruldu', '?')} | "
          f"önceki açılış: {len(gecmis)}")
    if onay != "EVET":
        print("\nBu komut kasayı AÇAR. Rezerv dönem bir kez görüldükten sonra bir")
        print("daha 'hiç bakılmamış veri' değildir; sonraki hiçbir koşu aynı kanıt")
        print("değerini taşımaz. Ekim'de, sistem donmuşken çalıştırılmalıdır.")
        print(f"\n    python pipeline_backtest.py kasa-ac {years} EVET")
        return
    if gecmis:
        print("\nUYARI: kasa DAHA ÖNCE açılmış "
              f"({', '.join(str(a.get('tarih')) for a in gecmis)}). Bu koşu "
              "örneklem-dışı DEĞİLDİR — sonucu öyle sunma.")

    _KASA_ACIK = True                      # bundan sonra panel budanmaz
    t0 = time.time()
    closes, vols = _panel_hazirla(years)
    feats = _precompute(closes, vols)
    ret_panel = pd.DataFrame({t: f["ret"] for t, f in feats.items()})

    xu, tum = _endeks_penceresi("XU100.IS", years)
    dates = [d for d in tum if d > kes]
    if len(dates) < 60:
        print(f"Rezerv dönemde yalnız {len(dates)} işlem günü var — test anlamsız.")
        return
    # İlk rebalans pencerenin İLK günü olmalı: ay başına beklemek portföyü
    # haftalarca boş bırakır ve rezerv dönemin başını ölçüm dışı kılar.
    reb = pd.date_range(dates[0], dates[-1], freq="MS")
    reb = sorted({dates[0], *(dates[pd.Index(dates).searchsorted(d)]
                              for d in reb if d <= dates[-1])})

    wbd = {}
    for d in reb:
        w = _pick_at(d, feats, ret_panel)
        if w:
            wbd[d] = w
    print(f"\nPanel {len(feats)} hisse | {len(wbd)}/{len(reb)} rebalans | "
          f"{dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d} ({time.time()-t0:.0f}s)")
    print(f"Donmuş parametreler: {MGMT_VARSAYILAN} | MAX_POS={MAX_POS} "
          f"SCORE_MIN={SCORE_MIN} SECTOR_CAP={SECTOR_CAP} "
          f"CORR_LIMIT={CORR_LIMIT} MIN_TL_VOL={MIN_TL_VOL:,}\n")

    xu_n = xu.loc[dates] / xu.loc[dates].iloc[0]
    nav_u = _walk(wbd, closes, dates)
    nav_m = _walk_managed(wbd, closes, dates, feats, dict(MGMT_VARSAYILAN))

    # Rastgele-6 kontrol: projenin en güçlü iddiası ("rastgeleden anlamlı iyi")
    # asıl burada sınanır — rezerv dönemde de tutuyor mu?
    all_t = sorted(feats)
    rng_navs = []
    for seed in range(RANDOM_SEEDS):
        rng = np.random.default_rng(seed)
        wbd_r = {}
        for d in reb:
            elig = [t for t in all_t
                    if feats[t]["close"].index.searchsorted(d) > 260]
            if len(elig) >= MAX_POS:
                wbd_r[d] = {t: 1 / MAX_POS for t in rng.choice(elig, MAX_POS,
                                                               replace=False)}
        rng_navs.append(_walk(wbd_r, closes, dates))
    rng_final = np.array([n.iloc[-1] for n in rng_navs])
    nav_rng = pd.concat(rng_navs, axis=1).mean(axis=1)

    print("=" * 118)
    su = _stats(nav_u, xu_n, "YÖNETİMSİZ (seçim)")
    sm = _stats(nav_m, xu_n, "YÖNETİLEN")
    sr = _stats(nav_rng, xu_n, "Rastgele-6 (ortalama)")
    sx = _stats(xu_n, xu_n, "XU100")
    print("=" * 118)
    yuzde = float((rng_final < nav_u.iloc[-1]).mean() * 100)
    print(f"\nRastgele dağılımda seçim modelinin yeri: son NAV {nav_u.iloc[-1]:.2f} | "
          f"rastgele %5-%95 bandı [{np.percentile(rng_final,5):.2f}, "
          f"{np.percentile(rng_final,95):.2f}] → {yuzde:.0f}. yüzdelik")
    # H2 ana sonuçtan SONRA ve ayrı korumayla: bir hata ana kasa sonucunu
    # ve açılış kaydını asla düşürmemeli.
    try:
        h2 = _saglik_kasa_kolu(reb, feats, ret_panel, closes, dates, xu_n, su)
    except Exception as e:
        print(f"\nH2 koşamadı: {e!r}")
        h2 = {"karar": "HATA", "hata": repr(e)}
    print("\nBu sayı bir kez üretilir. Sonucu beğenmeyip parametre değiştirmek,")
    print("rezerv dönemi sıradan bir eğitim penceresine çevirir — kasanın")
    print("tüm değeri buna direnmekte.")

    kayit = {
        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pencere": f"{dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d}",
        "gun": len(dates),
        "yil_argumani": years,
        "panel_hisse": len(feats),
        "parametreler": {**{k: MGMT_VARSAYILAN[k] for k in sorted(MGMT_VARSAYILAN)},
                         "MAX_POS": MAX_POS, "SCORE_MIN": SCORE_MIN,
                         "SECTOR_CAP": SECTOR_CAP, "CORR_LIMIT": CORR_LIMIT,
                         "MIN_TL_VOL": MIN_TL_VOL},
        "sonuc": {"yonetimsiz": su, "yonetilen": sm, "rastgele6_ort": sr,
                  "xu100": sx, "rastgele_yuzdelik": yuzde,
                  "h2_bilanco_filtresi": h2},
    }
    meta.setdefault("acilis_kaydi", []).append(kayit)
    meta["durum"] = "acildi"
    with open(KASA_DOSYA, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"\nAçılış {KASA_DOSYA} dosyasına yazıldı | süre {time.time()-t0:.0f}s")


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
    elif _cmd == "us":
        _transfer("US", yrs)
    elif _cmd == "uk":
        _transfer("UK", yrs)
    elif _cmd == "transfer":
        # Genel biçim: transfer <PIYASA> [yil]
        _p = sys.argv[2].upper() if len(sys.argv) > 2 else "US"
        if _p not in TRANSFER_PIYASA:
            raise SystemExit(f"Bilinmeyen piyasa: {_p} "
                             f"(tanimli: {', '.join(TRANSFER_PIYASA)})")
        _transfer(_p, int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    elif _cmd == "dsr":
        dsr(yrs)
    elif _cmd == "sanstest":
        sanstest(yrs)
    elif _cmd == "kurulum":
        kurulum(yrs)
    elif _cmd == "slippage":
        slippage(yrs)
    elif _cmd == "kasa":
        kasa_durum()
    elif _cmd == "kasa-ac":
        # Onay argümanı ZORUNLU — yanlışlıkla açılabilen bir kasa, kasa değildir.
        kasa_ac(int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                sys.argv[3] if len(sys.argv) > 3 else "")
    elif _cmd == "pozisyon":
        pozisyon(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif _cmd == "saglik":
        saglik_testi(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif _cmd == "devir":
        devir(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif _cmd == "momentum":
        momentum(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    else:
        print(__doc__)
