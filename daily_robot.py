"""
daily_robot — Günlük otonom bakım robotu (GitHub Actions'tan çalışır).

Uygulama açılmasa bile sistemin gözü piyasada kalır:
  1. Aktif portföylerin NAV'ını günceller (performans defteri kesintisiz)
  2. Risk alarmlarını kontrol eder (stop kırılımı / hedef / portföy freni)
     → pm_alerts tablosuna yazar, uygulama açılınca bant olarak görünür
  3. Haftalık işleri aksatmaz: gölge portföy seti + analist hedef fotoğrafı
  4. Piyasa rejimini günlük kaydeder (pm_regime_history — ileride analiz için)

Gereksinim: TURSO_DATABASE_URL + TURSO_AUTH_TOKEN ortam değişkenleri
(GitHub repo Secrets). Turso yoksa çalışmayı reddeder — CI konteynerinin
geçici diskine yazmak veri kaybı demektir.

Elle çalıştırma: python daily_robot.py
"""

import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Elle koşuda (Windows konsolu cp1254) çıktıdaki emoji/ok karakterleri
# UnicodeEncodeError veriyordu. Actions'ta zaten UTF-8, orada etkisiz.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import bist_analyzer as ba  # noqa: E402


def _rejim_tablosu() -> None:
    """pm_regime_history şeması: PK (date) → PK (date, market).

    Eski şema günde TEK satır tutuyordu. İki piyasa yazılınca ikincisi
    birincisini eziyordu; pratikte US rejimi BIST'inkini siliyordu, yani
    rejim geçmişi hem eksik hem de hangi piyasaya ait olduğu belirsizdi.

    SQLite'ta PRIMARY KEY yerinde değiştirilemez, tablo yeniden kurulur.
    Mevcut satırların hepsi compute_market_regime() VARSAYILANIYLA, yani
    BIST rejimiyle yazılmıştı → market='BIST' olarak taşınır (kayıpsız).
    'vix' kolonu US için eklendi; BIST'te NULL kalır (usdtry_1m'in US'te
    NULL kalması gibi) — iki piyasanın stres göstergesi farklı büyüklükler
    olduğu için tek kolona bindirmek analizi yanıltırdı.

    Adımlar tek execute_batch'te gönderilir. Yarıda kesilse bile kendini
    onarır: veri v2'ye kopyalandıktan sonra düşer, sonraki koşu boş tabloyu
    yeniden kurup INSERT OR IGNORE ile devam eder.
    """
    ba._PMDB.execute("""
        CREATE TABLE IF NOT EXISTS pm_regime_history (
            date TEXT PRIMARY KEY, regime TEXT, score INTEGER,
            breadth REAL, usdtry_1m REAL
        )
    """)
    kolonlar = {c.get("name") for c in ba._PMDB.execute(
        "SELECT name FROM pragma_table_info('pm_regime_history')")["rows"]}
    if "market" in kolonlar:
        return
    ba._PMDB.execute_batch([
        ("""CREATE TABLE IF NOT EXISTS pm_regime_history_v2 (
                date TEXT, market TEXT, regime TEXT, score INTEGER,
                breadth REAL, usdtry_1m REAL, vix REAL,
                PRIMARY KEY (date, market)
            )""", ()),
        ("INSERT OR IGNORE INTO pm_regime_history_v2 "
         "(date, market, regime, score, breadth, usdtry_1m) "
         "SELECT date, 'BIST', regime, score, breadth, usdtry_1m "
         "FROM pm_regime_history", ()),
        ("DROP TABLE pm_regime_history", ()),
        ("ALTER TABLE pm_regime_history_v2 RENAME TO pm_regime_history", ()),
    ])
    # ASCII ok: Windows konsolu (cp1254) '→' karakterini kodlayamıyor ve
    # elle koşuda UnicodeEncodeError veriyor.
    print("       pm_regime_history: (date) -> (date, market) semasina tasindi")


def main() -> int:
    print(f"=== Günlük Robot | {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    # Geç tur (US kapanışı sonrası) yalnız US'ü yönetir — BIST işleri akşam
    # turunda yapıldı. Boşsa tam tur koşulur.
    _only = (os.getenv("ROBOT_ONLY_MARKET") or "").strip().upper()
    _piyasalar = (_only,) if _only in ("BIST", "US") else ("BIST", "US")
    if _only:
        print(f"    (kısıtlı tur: yalnız {_only})")

    if not ba._PMDB.is_cloud():
        print("HATA: Turso yapılandırılmamış (TURSO_DATABASE_URL / TURSO_AUTH_TOKEN).")
        print("CI ortamında lokal SQLite'a yazmak anlamsız — çıkılıyor.")
        return 1

    # 1) NAV güncelle — yalnız işlenen piyasa(lar)
    # Kısıtsız çağrı (market=None) her iki piyasayı yazıyordu; 15:40 turunda bu,
    # US NAV'ını seans ORTASINDAKİ fiyatla kaydetmek demekti. Kararları kapanışa
    # taşıyıp performans defterini yarım günün fiyatıyla tutmak tutarsız olurdu.
    for _mkt in _piyasalar:
        try:
            ba.PortfolioManager.update_navs(market=_mkt)
            n_port = len(ba.PortfolioManager.active_portfolios(market=_mkt))
            print(f"[1/4]  [{_mkt}] NAV güncellendi ({n_port} aktif portföy)")
        except Exception as exc:
            print(f"[1/4]  [{_mkt}] NAV hatası: {exc}")

    # 2) Haftalık işler (kendi hafta korumaları var — gerekmiyorsa dokunmaz)
    try:
        snap = ba.PortfolioManager.snapshot_tv_targets()
        print(f"[2/4] Analist hedef fotoğrafı: {snap} hisse "
              f"({'yeni çekildi' if snap else 'bu hafta zaten var'})")
    except Exception as exc:
        print(f"[2/4] Hedef fotoğrafı hatası: {exc}")

    # Gölge + kontrol setleri — her piyasa için ayrı (BIST ve US bağımsız
    # kohortlar biriktirir; biri hata verse diğeri koşmaya devam etmeli)
    #
    # DİKKAT: burası eskiden kısıtlı turda (`_only` doluyken) HİÇ koşmuyordu
    # (`_piyasalar if not _only else ()`). Erken tur kısıtsız olduğu için o
    # zaman zararsızdı; artık her iki tur da kısıtlı, dolayısıyla o hâliyle
    # haftalık kohortlar hiçbir zaman oluşmazdı — gölge deneyinin tamamı
    # sessizce dururdu. Mükerrer koşma riski yok: hafta+piyasa bazlı
    # eksik_g/eksik_k kontrolü zaten idempotentlik sağlıyor.
    week = datetime.now().strftime("%G-W%V")
    for _mkt in _piyasalar:
        try:
            rows = ba._PMDB.execute(
                "SELECT COUNT(*) AS c FROM pm_portfolios WHERE kind='golge' AND horizon != 'kontrol' "
                "AND COALESCE(market,'BIST')=? AND name LIKE ?",
                (_mkt, f"%{week}%"))["rows"]
            rows_k = ba._PMDB.execute(
                "SELECT COUNT(*) AS c FROM pm_portfolios WHERE kind='golge' AND horizon = 'kontrol' "
                "AND COALESCE(market,'BIST')=? AND name LIKE ?",
                (_mkt, f"%{week}%"))["rows"]
            eksik_g = not rows or rows[0].get("c", 0) == 0
            eksik_k = not rows_k or rows_k[0].get("c", 0) == 0
            if eksik_g or eksik_k:
                print(f"      [{_mkt}] Bu haftanın gölge/kontrol seti eksik — tarama başlıyor...")
                regime = ba.compute_market_regime(_mkt)
                scan = ba.PortfolioScanner.scan_all(force=True, market=_mkt)
                n_g = ba.PortfolioManager.ensure_shadow_batch(regime, scan, market=_mkt) if eksik_g else 0
                n_k = ba.PortfolioManager.ensure_control_batch(scan, market=_mkt) if eksik_k else 0
                print(f"      [{_mkt}] Oluşturuldu: {n_g} gölge + {n_k} kontrol "
                      f"(rejim: {regime['regime']} {regime['score']}/7)")
                # US'te Ayı rejimi TÜM vadeleri kapatır (kalibrasyon: IC -0.063,
                # t=-3.24) — set boş kalırsa bu bir hata değil, kasıtlı frendir
                if _mkt == "US" and n_g == 0 and eksik_g:
                    print(f"      [{_mkt}] Gölge set boş — rejim/filtre engeli "
                          f"(rejim {regime['regime']}). Beklenen davranış.")
            else:
                print(f"      [{_mkt}] Gölge+kontrol set {week} mevcut "
                      f"({rows[0]['c']}+{rows_k[0]['c']})")
        except Exception as exc:
            print(f"      [{_mkt}] Gölge set hatası: {exc}")
    try:
        ba.PortfolioManager.auto_archive_shadows()
    except Exception as exc:
        print(f"      Arşivleme hatası: {exc}")

    # 2.9) GÜNLÜK PORTFÖY YÖNETİMİ — yönetilen portföyleri gözden geçirir,
    # gerekli çıkış/budama/alım işlemlerini uygular (gerekçeleriyle kaydeder).
    # NOT: US seansı 15:40 UTC'de HENÜZ AÇIK — US kararları o anki fiyatla
    # verilir. Kapanış sonrası tur için workflow'a ayrı bir geç koşu eklendi.
    for _mkt in _piyasalar:
        try:
            _rapor = ba.PortfolioDoctor.daily_manage(_mkt)
            _isl = sum(len(r["aksiyonlar"]) for r in _rapor)
            if not _rapor:
                print(f"[2.9]  [{_mkt}] Yönetilen portföy yok")
                continue
            print(f"[2.9]  [{_mkt}] {len(_rapor)} portföy incelendi, {_isl} aksiyon")
            for r in _rapor:
                for a in r["aksiyonlar"]:
                    print(f"        {r['portfoy'][:28]:28} | {a['aksiyon']:6} "
                          f"{a['ticker']:7} %{a['agirlik']:.0f} — {a['gerekce'][:60]}")
        except Exception as exc:
            print(f"[2.9]  [{_mkt}] Yönetim hatası: {exc}")

    # 2.95) YÖNETİM DENETİMİ — kuralların işe yarayıp yaramadığını ölçer,
    # hiçbir şeyi değiştirmez. Eşik aşılır VE örnek yeterliyse öneri üretir;
    # uygulamak editör onayına bağlıdır (bkz. CLAUDE.md > Denetim).
    for _mkt in _piyasalar:
        try:
            _d = ba.PortfolioAudit.run(_mkt)
            _m = _d["metrikler"]
            if _m:
                print(f"[2.95] [{_mkt}] Denetim: " + " | ".join(
                    f"{k}={v[0]:.2f}(n={v[1]})" for k, v in _m.items()))
            for _b in _d["bulgular"]:
                print(f"       ⚠️ {_b['mesaj'][:110]}")
                print(f"          öneri: {_b['parametre']} {_b['mevcut']} → {_b['onerilen']}")
        except Exception as exc:
            print(f"[2.95] [{_mkt}] Denetim hatası: {exc}")

    # 3) Risk alarmları
    try:
        alerts = []
        for _m in _piyasalar:
            alerts.extend(ba.PortfolioManager.check_risk_alerts(_m))
        print(f"[3/4] Risk kontrolü: {len(alerts)} yeni alarm")
        for a in alerts:
            print(f"      {a}")
    except Exception as exc:
        print(f"[3/4] Risk kontrolü hatası: {exc}")

    # 2.7) ENAG hatırlatıcısı — geçen ayın verisi eksikse duyuruyu tespit et
    # Not: haber başlıkları YILLIK değeri verir, sitenin SSL'i bozuk (525) —
    # bu yüzden sayı OTOMATİK YAZILMAZ; site düzelirse dener, yoksa
    # kullanıcıya "değeri gir" alarmı bırakır. Elle giriş her zaman esas.
    try:
        import re as _re
        from urllib.parse import quote as _q
        onceki_ay = (datetime.now().replace(day=1) - __import__("datetime").timedelta(days=1)).strftime("%Y-%m")
        rates = ba.InflationEngine.rates()
        kayit = rates.get(onceki_ay)
        eksik = (kayit is None) or ("tahmin" in (kayit[1] or ""))
        if eksik and datetime.now().day >= 4:
            yazildi = False
            try:  # 1) site (çoğunlukla 525, ama denemesi bedava)
                import requests as _rq
                r = _rq.get("https://enagrup.org/", timeout=20,
                            headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    m = _re.search(r"ayl[ıi]k[^%]{0,60}%\s*(\d{1,2})[.,](\d{1,2})",
                                   _re.sub(r"<[^>]+>", " ", r.text), _re.IGNORECASE)
                    if m:
                        val = float(f"{m.group(1)}.{m.group(2)}")
                        if 0 < val < 25:
                            # Kazınmış değer 'resmi' DEĞİL — doğrulanana dek
                            # ayırt edilebilir kalsın (aşağıdaki DOĞRULA alarmı
                            # ile birlikte). Kullanıcı UI'dan onaylayınca 'resmi' olur.
                            ba.InflationEngine.set_rate(onceki_ay, val,
                                                        kaynak="otomatik-dogrulanmadi")
                            ba._PMDB.execute(
                                "INSERT INTO pm_alerts (created_at, pid, ticker, tip, mesaj) VALUES (?,?,?,?,?)",
                                (datetime.now().strftime("%Y-%m-%d %H:%M"), 0, "*", "ENAG",
                                 f"ℹ️ ENAG {onceki_ay} aylık %{val} siteden otomatik alındı — ENAG Verisi sekmesinden DOĞRULA."))
                            yazildi = True
                            print(f"[2.7]  ENAG {onceki_ay}: %{val} siteden alındı (doğrulama alarmı bırakıldı)")
            except Exception:
                pass
            if not yazildi:  # 2) duyuru tespiti → hatırlatma alarmı (ayda 1)
                aylar_tr = ["", "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
                            "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]
                ay_adi = aylar_tr[int(onceki_ay[5:])]
                import requests as _rq
                xml = _rq.get(f"https://news.google.com/rss/search?q={_q('ENAG ' + ay_adi + ' enflasyon')}"
                              f"&hl=tr&gl=TR&ceid=TR:tr", timeout=20).text
                duyuru_var = ay_adi in xml.lower() and "enag" in xml.lower()
                onceki_alarm = ba._PMDB.execute(
                    "SELECT COUNT(*) AS c FROM pm_alerts WHERE tip='ENAG' AND mesaj LIKE ?",
                    (f"%{onceki_ay}%",))["rows"]
                if duyuru_var and (not onceki_alarm or onceki_alarm[0].get("c", 0) == 0):
                    ba.PortfolioManager._init_alerts()
                    ba._PMDB.execute(
                        "INSERT INTO pm_alerts (created_at, pid, ticker, tip, mesaj) VALUES (?,?,?,?,?)",
                        (datetime.now().strftime("%Y-%m-%d %H:%M"), 0, "*", "ENAG",
                         f"📰 ENAG {ay_adi} ayı enflasyonunu açıkladı — AYLIK değeri "
                         f"@ENAGRUP'tan bakıp ENAG Verisi sekmesine gir ({onceki_ay})."))
                    print(f"[2.7]  ENAG {onceki_ay} eksik — hatırlatma alarmı bırakıldı")
                else:
                    print(f"[2.7]  ENAG {onceki_ay} eksik, duyuru henüz tespit edilemedi")
        else:
            print(f"[2.7]  ENAG {onceki_ay} kayıtlı ✓")
    except Exception as exc:
        print(f"[2.7]  ENAG kontrol hatası: {exc}")

    # 3.5) Aylık parametre kararlılık koşusu (Roadmap-D) — ayda 1, ayın ilk günlerinde
    try:
        ay = datetime.now().strftime("%Y-%m")
        ba._PMDB.execute("CREATE TABLE IF NOT EXISTS calib_history "
                         "(run_date TEXT, metric TEXT, value REAL, PRIMARY KEY (run_date, metric))")
        rows = ba._PMDB.execute(
            "SELECT COUNT(*) AS c FROM calib_history WHERE run_date LIKE ?", (f"{ay}%",))["rows"]
        if not rows or rows[0].get("c", 0) == 0:
            print("[3.5]  Bu ayın kalibrasyon koşusu yok — başlıyor (~3 dk)...")
            import weight_calibration as wc
            wc.stability()
        else:
            print(f"[3.5]  Kalibrasyon {ay} zaten koşulmuş")
    except Exception as exc:
        print(f"[3.5]  Kalibrasyon hatası: {exc}")

    # 4) Rejim günlüğü — YALNIZ işlenen piyasa(lar) için
    try:
        _rejim_tablosu()
    except Exception as exc:
        print(f"[4/4] Rejim tablosu hazırlanamadı: {exc}")
    for _mkt in _piyasalar:
        try:
            regime = ba.compute_market_regime(_mkt)
            ba._PMDB.execute(
                "INSERT OR REPLACE INTO pm_regime_history "
                "(date, market, regime, score, breadth, usdtry_1m, vix) VALUES (?,?,?,?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d"), _mkt, regime["regime"], regime["score"],
                 regime.get("breadth"), regime.get("usdtry_1m"), regime.get("vix")))
            print(f"[4/4]  [{_mkt}] Rejim kaydı: {regime['regime']} ({regime['score']}/7)")
        except Exception as exc:
            print(f"[4/4]  [{_mkt}] Rejim kaydı hatası: {exc}")

    print("=== Robot tamamlandı ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
