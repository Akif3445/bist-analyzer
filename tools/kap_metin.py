"""KAP bildirim GÖVDE metnini çeker ve yön çıkarır.

NEDEN
-----
Bazı türlerin yönü ne türden ne özetten çıkıyor, yalnız gövde metninde var.
En net örnek kredi derecelendirmesi: 3.503 özette "yükseltildi" **2 kez**,
"düşürüldü" **hiç** geçiyor. Ama gövdede not değişimi yazılı.

NEREDE
------
KAP bildirimi serbest metin değil, ETİKETLİ FORM: HTML tablosunda XBRL
alanları var (oda_CreditRatingAbstract, oda_ExplanationTextBlock...) ve tablo
çift dilli geliyor (TR | EN). Anlatı metni **oda_ExplanationTextBlock**
alanında. Kaba etiket silme işe yaramaz — sitenin menüsünü getiriyor
(ölçüldü: 235 bin karakter HTML'den çıkan ilk 600 karakter KAP navigasyonu).

Kullanım:
    python tools/kap_metin.py cek "Kredi Derecelendirmesi" 800
    python tools/kap_metin.py yon "Kredi Derecelendirmesi"
"""

import glob
import io
import json
import os
import re
import sys
import time

METIN_DIZIN = "kap_metin"
BEKLE = 2.0          # KAP bir kez engelledi; nazik ol (bkz. kap_gecmis.py)


def _olaylar(tur):
    """Arşivden bu türün bildirimleri (yeni tarihli önce)."""
    kayit = []
    for y in glob.glob("kap_veri/*.json"):
        for k in json.load(io.open(y, encoding="utf-8")):
            if (k.get("subject") or "").strip() == tur and k.get("disclosureIndex"):
                kayit.append(k)

    def _t(k):
        t = (k.get("publishDate") or "")[:10]
        return t[6:10] + t[3:5] + t[0:2]
    kayit.sort(key=_t, reverse=True)
    return kayit


def _govde(html):
    """oda_ExplanationTextBlock satırındaki anlatı metni."""
    if not html:
        return ""
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        if "ExplanationTextBlock" not in tr and "AnnouncementContent" not in tr:
            continue
        x = re.sub(r"<[^>]+>", " ", tr)
        x = x.replace("&#x27;", "'").replace("&amp;", "&").replace("&nbsp;", " ")
        x = re.sub(r"oda_\w+\|?", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        if len(x) > 60:
            return x
    return ""


def cek(tur, limit):
    import borsapy
    os.makedirs(METIN_DIZIN, exist_ok=True)
    dosya = os.path.join(METIN_DIZIN,
                         re.sub(r"[^\w]+", "_", tur)[:40] + ".json")
    var = {}
    if os.path.exists(dosya):
        var = json.load(io.open(dosya, encoding="utf-8"))
        print(f"önbellek: {len(var)} metin")

    kayit = _olaylar(tur)
    print(f"{tur}: {len(kayit)} bildirim | hedef {limit} | ~"
          f"{max(0, min(limit, len(kayit)) - len(var)) * BEKLE / 60:.0f} dk",
          flush=True)

    t = borsapy.Ticker("ASELS")     # get_news_content ticker'dan bağımsız
    t0, yeni, bos, hata = time.time(), 0, 0, 0
    for i, k in enumerate(kayit[:limit], 1):
        did = str(k["disclosureIndex"])
        if did in var:
            continue
        try:
            g = _govde(t.get_news_content(did))
        except Exception:
            hata += 1
            time.sleep(3.0)
            continue
        if not g:
            bos += 1
        var[did] = {"metin": g, "tarih": k.get("publishDate"),
                    "hisse": k.get("stockCodes"), "ozet": k.get("summary")}
        yeni += 1
        if yeni % 25 == 0:
            with open(dosya, "w", encoding="utf-8") as f:
                json.dump(var, f, ensure_ascii=False)
            print(f"  {i}/{min(limit, len(kayit))} | yeni {yeni} | boş {bos} "
                  f"| hata {hata} | {time.time()-t0:.0f}s", flush=True)
        time.sleep(BEKLE)

    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(var, f, ensure_ascii=False)
    print(f"Bitti: {len(var)} metin ({yeni} yeni, {bos} boş, {hata} hata) "
          f"| {time.time()-t0:.0f}s -> {dosya}")


# Kredi notu yon kaliplari. Turkce KAP dilinde not degisimi bu kaliplarla
# yazilir; "teyit" (affirm) ucuncu ve notr bir durum.
YUKARI = [r"yükselt", r"artır(ıl|dı)", r"iyileş", r"upgrade", r"yukarı yönlü",
          r"pozitif görünüm", r"olumlu görünüm"]
ASAGI = [r"düşür", r"indir(il|di)", r"geriye çek", r"downgrade",
         r"aşağı yönlü", r"negatif görünüm", r"olumsuz görünüm"]
AYNI = [r"teyit", r"koru(n|du|muş)", r"sabit", r"değiştirme", r"affirm",
        r"aynı seviyede"]


def yon(tur):
    dosya = os.path.join(METIN_DIZIN, re.sub(r"[^\w]+", "_", tur)[:40] + ".json")
    if not os.path.exists(dosya):
        raise SystemExit(f"{dosya} yok — önce cek")
    var = json.load(io.open(dosya, encoding="utf-8"))
    from collections import Counter
    say = Counter()
    ornek = {}
    for did, v in var.items():
        m = (v.get("metin") or "").lower()
        if not m:
            say["metin yok"] += 1
            continue
        y = sum(1 for p in YUKARI if re.search(p, m))
        a = sum(1 for p in ASAGI if re.search(p, m))
        s = sum(1 for p in AYNI if re.search(p, m))
        if y > a and y > 0:
            g = "YUKARI"
        elif a > y and a > 0:
            g = "ASAGI"
        elif s > 0:
            g = "AYNI"
        else:
            g = "belirsiz"
        say[g] += 1
        ornek.setdefault(g, (v.get("hisse"), (v.get("metin") or "")[:150]))
    n = sum(say.values())
    print(f"{tur}: {n} metin")
    for g, c in say.most_common():
        print(f"  {g:<10} {c:>5}  (%{c*100/max(n,1):.0f})")
    print("\nörnekler:")
    for g, (h, m) in ornek.items():
        print(f"  [{g}] {h}: {m}")


if __name__ == "__main__":
    k = sys.argv[1] if len(sys.argv) > 1 else ""
    if k == "cek":
        cek(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 500)
    elif k == "yon":
        yon(sys.argv[2])
    else:
        print(__doc__)
