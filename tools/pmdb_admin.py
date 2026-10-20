#!/usr/bin/env python
"""PM veritabanı bakım aracı (Turso / lokal SQLite).

Neden ayrı bir script: bakım işleri için Claude'a "her türlü python kodunu
çalıştır" izni vermek yerine YALNIZ bu script'e izin verilir. Yapılabilecekler
buradaki alt komutlarla sınırlıdır — kod repoda, versiyon kontrolünde ve
gözden geçirilebilir. (En az yetki ilkesi; bkz. CLAUDE.md > Security.)

Kullanım:
    python tools/pmdb_admin.py enag-list
    python tools/pmdb_admin.py enag-delete 2026-07
    python tools/pmdb_admin.py enag-senkron          # yalnız gösterir
    python tools/pmdb_admin.py enag-senkron --yaz    # uygular
    python tools/pmdb_admin.py portfolios

Yazma yapan komutlar enag-delete ve enag-senkron --yaz'dır. enag-delete ay
biçimini doğrular ve silmeden önce satırı basar. enag-senkron kod
varsayılanlarını (ENAG_MOM_DEFAULTS) tabloya taşır: eksik ayı ekler, var olan
satırı YALNIZ kaynağı bir kod tahminiyse günceller; resmi, arşiv ya da elle
girilmiş değerlere dokunmaz.
"""
import argparse
import re
import sys

# Türkçe çıktı Windows konsolunda cp1254'e takılmasın
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__))))

from bist_analyzer import _PMDB  # noqa: E402

YM_RE = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")


def enag_list(_args):
    rows = _PMDB.execute(
        "SELECT ym, mom_pct, kaynak FROM enag_monthly ORDER BY ym DESC LIMIT 24")["rows"]
    if not rows:
        print("enag_monthly boş.")
        return 0
    print(f"{'Ay':<9} {'Aylık %':>8}  Kaynak")
    for r in rows:
        print(f"{r['ym']:<9} {r['mom_pct']:>8.2f}  {r.get('kaynak', '')}")
    return 0


def enag_delete(args):
    ym = args.ym.strip()
    if not YM_RE.match(ym):
        print(f"HATA: ay biçimi YYYY-AA olmalı (verilen: {ym!r})", file=sys.stderr)
        return 2
    rows = _PMDB.execute(
        "SELECT ym, mom_pct, kaynak FROM enag_monthly WHERE ym=?", (ym,))["rows"]
    if not rows:
        print(f"{ym} zaten yok — değişiklik yapılmadı.")
        return 0
    print(f"Silinecek: {rows[0]}")
    _PMDB.execute("DELETE FROM enag_monthly WHERE ym=?", (ym,))
    kalan = _PMDB.execute(
        "SELECT ym, mom_pct FROM enag_monthly ORDER BY ym DESC LIMIT 3")["rows"]
    print(f"Silindi. Kalan son 3 ay: {kalan}")
    return 0


TAHMIN_KAYNAKLARI = {"tahmini-yillik", "tahmini", "tahmini-olcekli"}


def enag_senkron(args):
    # InflationEngine._init_table() BURADA ÇAĞRILMAZ: o fonksiyon eksik ayları
    # INSERT OR IGNORE ile kendiliğinden ekler, yani "yalnız göster" modunu
    # sessizce yazar hâle getirirdi (ilk denemede tam olarak bu oldu).
    from bist_analyzer import ENAG_MOM_DEFAULTS
    mevcut = {r["ym"]: (r["mom_pct"], r.get("kaynak") or "")
              for r in _PMDB.execute("SELECT ym, mom_pct, kaynak FROM enag_monthly")["rows"]}
    ekle, guncelle, korunan = [], [], []
    for ym, (pct, kaynak) in sorted(ENAG_MOM_DEFAULTS.items()):
        if ym not in mevcut:
            ekle.append((ym, pct, kaynak))
        elif mevcut[ym][1] in TAHMIN_KAYNAKLARI:
            if abs(float(mevcut[ym][0]) - pct) > 1e-9 or mevcut[ym][1] != kaynak:
                guncelle.append((ym, mevcut[ym], pct, kaynak))
        elif abs(float(mevcut[ym][0]) - pct) > 1e-9:
            korunan.append((ym, mevcut[ym], pct))
    print(f"Eklenecek {len(ekle)} ay, güncellenecek {len(guncelle)} ay, "
          f"koddan farklı ama korunan (gerçek değer) {len(korunan)} ay.")
    for ym, pct, k in ekle:
        print(f"  + {ym}  {pct:6.2f}  {k}")
    for ym, (eski, ek), pct, k in guncelle:
        print(f"  ~ {ym}  {eski:6.2f} ({ek}) → {pct:6.3f} ({k})")
    for ym, (eski, ek), pct in korunan:
        print(f"  = {ym}  tabloda {eski:6.2f} ({ek}), kodda {pct:6.2f} — dokunulmadı")
    if not args.yaz:
        print("\nHiçbir şey yazılmadı. Uygulamak için: --yaz")
        return 0
    stmts = [("INSERT INTO enag_monthly (ym, mom_pct, kaynak) VALUES (?,?,?)", (ym, pct, k))
             for ym, pct, k in ekle]
    stmts += [("UPDATE enag_monthly SET mom_pct=?, kaynak=? WHERE ym=?", (pct, k, ym))
              for ym, _e, pct, k in guncelle]
    if stmts:
        _PMDB.execute_batch(stmts)
    print(f"\nYazıldı: {len(ekle)} eklendi, {len(guncelle)} güncellendi.")
    return 0


def portfolios(_args):
    rows = _PMDB.execute(
        "SELECT id, name, horizon, profile, status, kind FROM pm_portfolios "
        "ORDER BY id")["rows"]
    print(f"toplam {len(rows)} portföy")
    for r in rows:
        print(f"  {r['id']:>3} | {str(r['name'])[:40]:<40} | {r['horizon']:<8} | "
              f"{str(r['profile'])[:10]:<10} | {r['kind']:<9} | {r['status']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("enag-list", help="ENAG aylık oranlarını listele").set_defaults(fn=enag_list)

    p_del = sub.add_parser("enag-delete", help="Bir ENAG ayını sil (YYYY-AA)")
    p_del.add_argument("ym", help="Silinecek ay, YYYY-AA biçiminde")
    p_del.set_defaults(fn=enag_delete)

    p_sk = sub.add_parser("enag-senkron", help="Kod varsayılanlarını ENAG tablosuna taşı")
    p_sk.add_argument("--yaz", action="store_true", help="Değişiklikleri uygula (yoksa yalnız gösterir)")
    p_sk.set_defaults(fn=enag_senkron)

    sub.add_parser("portfolios", help="Kayıtlı portföyleri listele").set_defaults(fn=portfolios)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
