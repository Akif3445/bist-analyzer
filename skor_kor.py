"""Teknik-only skor (backtest) vs tam agirlikli skor (canli) — siralama korelasyonu."""
import warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
import sys, time; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from scipy import stats
import bist_analyzer as ba

_s = ba.PortfolioScanner.scan_all(market="BIST")
scan = _s if isinstance(_s, dict) else {r.ticker: r for r in _s}
gecerli = [r for r in scan.values() if not r.error and r.data_rows >= 200 and r.volume_ok]
gecerli.sort(key=lambda r: -r.score)
# Ust bant + orta bant karisik ornek (sadece tepeye bakmak korelasyonu sisirir)
ornek = gecerli[:20] + gecerli[len(gecerli)//2 : len(gecerli)//2 + 15]
print(f'Taramada {len(gecerli)} gecerli hisse | ornek {len(ornek)}')

t0 = time.time(); sat = []
for i, r in enumerate(ornek, 1):
    try:
        s = ba.compute_bist_score(r.ticker, market='BIST')
        sat.append({'ticker': r.ticker, 'teknik': s.teknik_score, 'tam': s.total_score,
                    'sentiment': s.sentiment_score, 'prim': s.prim_score, 'deger': s.deger_score})
        if i % 10 == 0: print(f'  {i}/{len(ornek)} ({time.time()-t0:.0f}s)')
    except Exception as e:
        print(f'  {r.ticker}: {str(e)[:50]}')
df = pd.DataFrame(sat).dropna()
print(f'\n{len(df)} hisse icin tam skor hesaplandi\n')
sp = stats.spearmanr(df.teknik, df.tam)
pe = stats.pearsonr(df.teknik, df.tam)
print(f'Spearman (siralama) : {sp.statistic:+.3f}  (p={sp.pvalue:.4f})')
print(f'Pearson  (seviye)   : {pe[0]:+.3f}')
print()
# Asil soru: ILK 6 secimi degisiyor mu?
t6 = set(df.nlargest(6, 'teknik').ticker); f6 = set(df.nlargest(6, 'tam').ticker)
print(f'Teknik-only ilk 6 : {sorted(t6)}')
print(f'Tam skor    ilk 6 : {sorted(f6)}')
print(f'ORTAK             : {len(t6 & f6)}/6')
print()
print('Bilesenlerin tam skorla korelasyonu:')
for k in ('teknik','sentiment','prim','deger'):
    print(f'  {k:<10} {stats.spearmanr(df[k], df.tam).statistic:+.3f}')
df.to_csv('skor_karsilastirma.csv', index=False)
print('\nskor_karsilastirma.csv yazildi')
