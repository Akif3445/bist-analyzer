"""KAP tarihsel bildirim toplayıcı — doğrulamanın ön şartı.

NEDEN VAR
---------
Haber katmanı bugüne dek DOĞRULANAMIYORDU: Google News arşiv vermiyor (en çok
haber yapılan 10 BIST isminde 8'i sıfır makale döndü) ve borsapy yalnız son 20
bildirimi veriyor. Doğrulanamayan bir bileşen skorun bir parçası olamaz — bu
projede o hatayı zaten bir kez yaptık.

KAP'ın kendi sorgu ucu bunu çözüyor:

    POST https://www.kap.org.tr/tr/api/disclosure/members/byCriteria

Ölçüldü (2026-09-26): 2018'e kadar geçmiş var, mkkMemberOidList boş
bırakıldığında TÜM piyasa geliyor, bir gün ~0.2 saniyede dönüyor.

DÖNEN ALANLAR ve neden değerli
------------------------------
    publishDate      tarih+saat (olay çalışması için gerekli)
    stockCodes       hangi hisse(ler)
    subject          bildirim TÜRÜ — kapalı sözlük, bkz. kap_sozluk.json
    summary          insan yazımı kısa özet, %91 dolu. Form başlığından çok
                     daha bilgilendirici: "Yatırım Teşvik Belgesi Alımı",
                     "Uzun Vadeli Kredi Anlaşması", "Genel Müdür Ataması"
    disclosureIndex  bildirim kimliği (tam metin için get_news_content)
    isLate           geç bildirim bayrağı — kendi başına bir sinyal adayı
    disclosureClass  ODA / DKB / DG / FR üst kategorisi

İKİ SINIR, ikisi de ölçüldü
---------------------------
1. Yanıt başına **2000 kayıt** tavanı var ve tavana çarpınca aralığın yalnız
   SON kısmı döner — sessizce. 1 aylık sorgu 2000 kayıt döndürüp sadece son 7
   günü kapsadı. Bu yüzden GÜN GÜN çekilir; günlük hacim 300-700 arası.
2. fromDate/toDate dışındaki filtreler (sayfalama vb.) yok sayılıyor.

Kullanım:
    python tools/kap_gecmis.py cek 2024-01-01 2024-12-31
    python tools/kap_gecmis.py ozet
"""

import json
import os
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta

import httpx

UC = "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria"
BASLIKLAR = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Origin": "https://www.kap.org.tr",
    "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}

VERI_DIZIN = "kap_veri"          # gün başına bir JSON, .gitignore'a girer
TAVAN = 2000                     # ucun sessiz kayıt tavanı
BEKLE = 0.35                     # istekler arası saniye — nazik davran


def _yuk(gun: str) -> dict:
    return {
        "fromDate": gun, "toDate": gun,
        "memberTypes": ["IGS"],      # IGS = borsa şirketleri (fonlar hariç)
        "mkkMemberOidList": [],      # boş = tüm piyasa
        "disclosureClass": "", "subjects": [], "mainSector": "", "sector": "",
        "subSector": "", "market": "", "index": "", "year": "", "term": "",
        "ruleType": "",
    }


def _gunler(bas: str, son: str):
    b = datetime.strptime(bas, "%Y-%m-%d").date()
    s = datetime.strptime(son, "%Y-%m-%d").date()
    while b <= s:
        if b.weekday() < 5:          # KAP hafta içi çalışır
            yield b.isoformat()
        b += timedelta(days=1)


def cek(bas: str, son: str, yenile: bool = False) -> None:
    """Gün gün indirir, her günü ayrı dosyaya yazar (kaldığı yerden devam eder)."""
    os.makedirs(VERI_DIZIN, exist_ok=True)
    gunler = list(_gunler(bas, son))
    yeni = atlanan = toplam = 0
    tavana_carpan = []
    t0 = time.time()

    with httpx.Client(timeout=60, headers=BASLIKLAR, follow_redirects=True) as c:
        for i, gun in enumerate(gunler, 1):
            yol = os.path.join(VERI_DIZIN, f"{gun}.json")
            if os.path.exists(yol) and not yenile:
                atlanan += 1
                with open(yol, encoding="utf-8") as f:
                    toplam += len(json.load(f))
                continue
            try:
                r = c.post(UC, json=_yuk(gun))
                r.raise_for_status()
                d = r.json()
                if not isinstance(d, list):
                    d = []
            except Exception as exc:
                print(f"  {gun}: HATA {type(exc).__name__} {str(exc)[:70]}")
                time.sleep(2.0)
                continue

            # Tavana çarpma SESSİZ bir veri kaybıdır — mutlaka bağır.
            if len(d) >= TAVAN:
                tavana_carpan.append(gun)

            with open(yol, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            yeni += 1
            toplam += len(d)
            if i % 20 == 0:
                print(f"  {i}/{len(gunler)} gün | {toplam} bildirim "
                      f"| {time.time()-t0:.0f}s")
            time.sleep(BEKLE)

    print(f"\nBitti: {len(gunler)} iş günü | {yeni} indirildi, {atlanan} zaten vardı "
          f"| {toplam} bildirim | {time.time()-t0:.0f}s")
    if tavana_carpan:
        print(f"DİKKAT: {len(tavana_carpan)} gün {TAVAN} tavanına çarptı, o günler "
              f"EKSİK olabilir: {tavana_carpan[:5]}")


def _tum_kayitlar():
    if not os.path.isdir(VERI_DIZIN):
        return
    for ad in sorted(os.listdir(VERI_DIZIN)):
        if not ad.endswith(".json"):
            continue
        with open(os.path.join(VERI_DIZIN, ad), encoding="utf-8") as f:
            for kayit in json.load(f):
                yield kayit


def ozet() -> None:
    """İndirilen verinin dökümü."""
    konu, hisse, sinif = Counter(), Counter(), Counter()
    n = ozetli = gec = 0
    ilk = son = None
    for k in _tum_kayitlar():
        n += 1
        konu[k.get("subject") or "?"] += 1
        sinif[k.get("disclosureClass") or "?"] += 1
        for h in (k.get("stockCodes") or "").split(","):
            if h.strip():
                hisse[h.strip()] += 1
        if (k.get("summary") or "").strip():
            ozetli += 1
        if k.get("isLate"):
            gec += 1
        t = (k.get("publishDate") or "")[:10]
        if t:
            ilk = t if ilk is None or t < ilk else ilk
            son = t if son is None or t > son else son
    if not n:
        print(f"{VERI_DIZIN}/ boş — önce: python tools/kap_gecmis.py cek <bas> <son>")
        return
    print(f"Bildirim   : {n}")
    print(f"Tarih      : {ilk} .. {son}")
    print(f"Hisse      : {len(hisse)}")
    print(f"Ayrı tür   : {len(konu)}")
    print(f"summary dolu: {ozetli} (%{ozetli*100/n:.0f})")
    print(f"isLate     : {gec}")
    print(f"sınıf      : {dict(sinif.most_common(6))}")
    print("\nEn sık 15 tür:")
    for k, c in konu.most_common(15):
        print(f"  {c:>6}  {k[:66]}")


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else ""
    if komut == "cek":
        if len(sys.argv) < 4:
            raise SystemExit("kullanım: cek <YYYY-MM-DD> <YYYY-MM-DD>")
        cek(sys.argv[2], sys.argv[3], yenile="--yenile" in sys.argv)
    elif komut == "ozet":
        ozet()
    else:
        print(__doc__)
