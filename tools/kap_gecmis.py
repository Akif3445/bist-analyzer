"""KAP tarihsel bildirim toplayıcı — doğrulamanın ön şartı.

NEDEN VAR
---------
Haber katmanı bugüne dek DOĞRULANAMIYORDU: Google News arşiv vermiyor (en çok
haber yapılan 10 BIST isminde 8'i sıfır makale döndü) ve borsapy yalnız son 20
bildirimi veriyor. Doğrulanamayan bir bileşen skorun parçası olamaz.

KAP'ın kendi sorgu ucu bunu çözüyor:

    POST https://www.kap.org.tr/tr/api/disclosure/members/byCriteria

Ölçüldü (2026-09-26): 2018'e kadar geçmiş var; ASELS 2018'de 145, 2023'te 168
bildirim döndürüyor.

NEDEN HİSSE BAŞINA, GÜN BAŞINA DEĞİL
------------------------------------
İlk sürüm tüm piyasayı GÜN GÜN çekiyordu: 1740 istek. Ölçülen sonuç kötüydü —
KAP sürekli istek altında bağlantıyı kesiyor ("Server disconnected without
sending a response"); bir koşuda 200 günde 33, diğerinde 547 günde 149 gün
kaybedildi. İlk teşhis "yalnız 2018-05-21 bozuk" idi ve YANLIŞTI: o tarihler
tek tek denenince çalışıyor, uzun koşuda çalışmıyor. Yani bu bir hız
sınırlaması, bozuk veri değil.

Hisse başına sorgu hem daha az istek hem daha doğal bir kullanım:
    ~500 hisse x 1 istek  =  ~500 istek   (1740 yerine)
ve bir hissenin 8 yılı tavanın altında kalıyor (ASELS ~910 kayıt < 2000).
Ayrıca zaten yalnız donmuş evrendeki hisseler lazım, tüm piyasa değil.

İKİ SINIR, ikisi de ölçüldü
---------------------------
1. Yanıt başına **2000 kayıt** tavanı var ve tavana çarpınca aralığın yalnız
   SON kısmı döner — SESSİZCE. Bu yüzden tavana değen hisse yıl yıl bölünür.
2. fromDate/toDate dışındaki filtreler (sayfalama vb.) yok sayılıyor.

Kopan bağlantı HAVUZDA kalır ve sonraki istekleri de düşürür (ölçüldü:
2018-05-22 döngüde atlandı, tek başına 169 kayıt döndürdü), bu yüzden hatada
istemci yeniden kurulur.

Kullanım:
    python tools/kap_gecmis.py cek 2018-01-01 2024-08-31
    python tools/kap_gecmis.py ozet
"""

import json
import os
import sys
import time
from collections import Counter

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

VERI_DIZIN = "kap_veri"          # hisse başına bir JSON, .gitignore'da
EVREN = "backtest_evren.json"    # donmuş evren — backtest ile aynı liste
TAVAN = 2000
BEKLE = 2.5                      # istekler arası saniye — NAZIK OL, aşağıyı oku
OID_ONBELLEK = "kap_oid.json"    # şirket->oid, bir kez çekilir

# HIZ HAKKINDA, pahalı öğrenildi (2026-09-26): 0.35 sn aralıkla ve üst üste
# koşan işlerle binlerce istek atıldı ve KAP ERİŞİMİ KESTİ — get_companies()
# bile "Server disconnected" vermeye başladı. Bu bir veri hatası değil, hak
# edilmiş bir engelleme. 500 hisse x 2.5 sn ~ 20 dakika; acelesi yok, veri
# geçmişe ait ve bir kez indirilecek. AYRICA: aynı anda BİRDEN FAZLA çekim
# işi çalıştırma.


def _yuk(bas, son, oid):
    return {"fromDate": bas, "toDate": son, "memberTypes": ["IGS"],
            "mkkMemberOidList": [oid], "disclosureClass": "", "subjects": [],
            "mainSector": "", "sector": "", "subSector": "", "market": "",
            "index": "", "year": "", "term": "", "ruleType": ""}


def _istek(c, bas, son, oid, deneme=3):
    """Tek sorgu; hatada istemci yenilenir (kopan bağlantı havuzda kalıyor)."""
    for i in range(deneme):
        try:
            r = c.post(UC, json=_yuk(bas, son, oid), timeout=30)
            r.raise_for_status()
            j = r.json()
            return (j if isinstance(j, list) else []), c
        except Exception as exc:
            print(f"      deneme {i+1}/{deneme}: {type(exc).__name__} "
                  f"{str(exc)[:50]}", flush=True)
            try:
                c.close()
            except Exception:
                pass
            c = httpx.Client(timeout=60, headers=BASLIKLAR, follow_redirects=True)
            time.sleep(4.0 * (i + 1))
    return None, c


def cek(bas: str, son: str, yenile: bool = False) -> None:
    from borsapy._providers.kap import get_kap_provider

    os.makedirs(VERI_DIZIN, exist_ok=True)
    with open(EVREN, encoding="utf-8") as f:
        hisseler = json.load(f)["tickers"]
    print(f"Evren: {len(hisseler)} hisse | {bas} .. {son}", flush=True)

    kap = get_kap_provider()
    oid_harita = {}
    if os.path.exists(OID_ONBELLEK):
        with open(OID_ONBELLEK, encoding="utf-8") as f:
            oid_harita = json.load(f)
        print(f"OID önbelleği: {len(oid_harita)} hisse", flush=True)
    c = httpx.Client(timeout=60, headers=BASLIKLAR, follow_redirects=True)
    t0 = time.time()
    yeni = atlanan = toplam = 0
    basarisiz, oidsiz = [], []

    try:
        for i, tk in enumerate(hisseler, 1):
            yol = os.path.join(VERI_DIZIN, f"{tk}.json")
            if os.path.exists(yol) and not yenile:
                with open(yol, encoding="utf-8") as f:
                    toplam += len(json.load(f))
                atlanan += 1
                continue
            oid = oid_harita.get(tk)
            if not oid:
                try:
                    oid = kap.get_member_oid(tk)
                except Exception as exc:
                    print(f"  {tk}: oid alınamadı ({type(exc).__name__}) — KAP "
                          f"engelliyor olabilir, bir süre bekle", flush=True)
                    oid = None
                if oid:
                    oid_harita[tk] = oid
                    with open(OID_ONBELLEK, "w", encoding="utf-8") as f:
                        json.dump(oid_harita, f, ensure_ascii=False, indent=1)
                    time.sleep(BEKLE)
            if not oid:
                oidsiz.append(tk)
                continue

            d, c = _istek(c, bas, son, oid)
            if d is None:
                basarisiz.append(tk)
                continue

            # Tavana değdiyse yıl yıl böl — yoksa aralığın BAŞI sessizce kaybolur
            if len(d) >= TAVAN:
                print(f"  {tk}: {TAVAN} tavanı — yıl yıl bölünüyor", flush=True)
                birlesik, gorulen = [], set()
                for yil in range(int(bas[:4]), int(son[:4]) + 1):
                    y1 = max(f"{yil}-01-01", bas)
                    y2 = min(f"{yil}-12-31", son)
                    dy, c = _istek(c, y1, y2, oid)
                    for k in (dy or []):
                        ix = k.get("disclosureIndex")
                        if ix not in gorulen:
                            gorulen.add(ix)
                            birlesik.append(k)
                    time.sleep(BEKLE)
                d = birlesik

            with open(yol, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            yeni += 1
            toplam += len(d)
            if i % 25 == 0:
                print(f"  {i}/{len(hisseler)} hisse | {toplam} bildirim "
                      f"| {time.time()-t0:.0f}s", flush=True)
            time.sleep(BEKLE)
    finally:
        c.close()

    print(f"\nBitti: {yeni} indirildi, {atlanan} zaten vardı | {toplam} bildirim "
          f"| {time.time()-t0:.0f}s")
    if oidsiz:
        print(f"KAP'ta bulunamadı ({len(oidsiz)}): {oidsiz[:10]}")
    if basarisiz:
        print(f"BAŞARISIZ ({len(basarisiz)}) — tekrar çalıştırınca denenir: "
              f"{basarisiz[:10]}")


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
        if len(t) == 10:
            g = t[6:10] + t[3:5] + t[0:2]      # GG.AA.YYYY -> YYYYAAGG
            ilk = g if ilk is None or g < ilk else ilk
            son = g if son is None or g > son else son
    if not n:
        print(f"{VERI_DIZIN}/ boş — önce: python tools/kap_gecmis.py cek <bas> <son>")
        return
    print(f"Bildirim    : {n}")
    print(f"Tarih       : {ilk} .. {son}")
    print(f"Hisse       : {len(hisse)}")
    print(f"Ayrı tür    : {len(konu)}")
    print(f"summary dolu: {ozetli} (%{ozetli*100/n:.0f})")
    print(f"isLate      : {gec}")
    print(f"sınıf       : {dict(sinif.most_common(6))}")
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
