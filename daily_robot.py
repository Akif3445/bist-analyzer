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

import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import bist_analyzer as ba  # noqa: E402


def main() -> int:
    print(f"=== Günlük Robot | {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    if not ba._PMDB.is_cloud():
        print("HATA: Turso yapılandırılmamış (TURSO_DATABASE_URL / TURSO_AUTH_TOKEN).")
        print("CI ortamında lokal SQLite'a yazmak anlamsız — çıkılıyor.")
        return 1

    # 1) NAV güncelle
    try:
        ba.PortfolioManager.update_navs()
        n_port = len(ba.PortfolioManager.active_portfolios())
        print(f"[1/4] NAV güncellendi ({n_port} aktif portföy)")
    except Exception as exc:
        print(f"[1/4] NAV hatası: {exc}")

    # 2) Haftalık işler (kendi hafta korumaları var — gerekmiyorsa dokunmaz)
    try:
        snap = ba.PortfolioManager.snapshot_tv_targets()
        print(f"[2/4] Analist hedef fotoğrafı: {snap} hisse "
              f"({'yeni çekildi' if snap else 'bu hafta zaten var'})")
    except Exception as exc:
        print(f"[2/4] Hedef fotoğrafı hatası: {exc}")

    try:
        week = datetime.now().strftime("%G-W%V")
        rows = ba._PMDB.execute(
            "SELECT COUNT(*) AS c FROM pm_portfolios WHERE kind='golge' AND horizon != 'kontrol' AND name LIKE ?",
            (f"%{week}%",))["rows"]
        rows_k = ba._PMDB.execute(
            "SELECT COUNT(*) AS c FROM pm_portfolios WHERE kind='golge' AND horizon = 'kontrol' AND name LIKE ?",
            (f"%{week}%",))["rows"]
        eksik_g = not rows or rows[0].get("c", 0) == 0
        eksik_k = not rows_k or rows_k[0].get("c", 0) == 0
        if eksik_g or eksik_k:
            print("      Bu haftanın gölge/kontrol seti eksik — tarama başlıyor (~2 dk)...")
            regime = ba.compute_market_regime()
            scan = ba.PortfolioScanner.scan_all(force=True)
            n_g = ba.PortfolioManager.ensure_shadow_batch(regime, scan) if eksik_g else 0
            n_k = ba.PortfolioManager.ensure_control_batch(scan) if eksik_k else 0
            print(f"      Oluşturuldu: {n_g} gölge + {n_k} kontrol (rejim: {regime['regime']})")
        else:
            print(f"      Gölge+kontrol set {week} mevcut ({rows[0]['c']}+{rows_k[0]['c']})")
        ba.PortfolioManager.auto_archive_shadows()
    except Exception as exc:
        print(f"      Gölge set hatası: {exc}")

    # 3) Risk alarmları
    try:
        alerts = ba.PortfolioManager.check_risk_alerts()
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

    # 4) Rejim günlüğü
    try:
        regime = ba.compute_market_regime()
        ba._PMDB.execute("""
            CREATE TABLE IF NOT EXISTS pm_regime_history (
                date TEXT PRIMARY KEY, regime TEXT, score INTEGER,
                breadth REAL, usdtry_1m REAL
            )
        """)
        ba._PMDB.execute(
            "INSERT OR REPLACE INTO pm_regime_history (date, regime, score, breadth, usdtry_1m) VALUES (?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d"), regime["regime"], regime["score"],
             regime.get("breadth"), regime.get("usdtry_1m")))
        print(f"[4/4] Rejim kaydı: {regime['regime']} ({regime['score']}/7)")
    except Exception as exc:
        print(f"[4/4] Rejim kaydı hatası: {exc}")

    print("=== Robot tamamlandı ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
