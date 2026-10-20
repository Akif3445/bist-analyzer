# BIST Smart Investment Assistant

A stock analysis and portfolio management platform for Borsa Istanbul (BIST) and US markets. It scores stocks 0–100 by combining technical analysis, news sentiment, analyst upside and valuation — then proposes portfolios, manages them daily, and measures whether any of it actually works.

Live demo: [bist-analyzer-wrtwzcsmdv4z9fxrpkkrcz.streamlit.app](https://bist-analyzer-wrtwzcsmdv4z9fxrpkkrcz.streamlit.app)

> **Not investment advice.** This is a research and educational project. See [Disclaimer](#disclaimer).

---

## What makes this different

Most stock screeners tell you what to buy. The harder question is *whether the thing telling you works at all* — and that is where most of the effort here went:

- **Weights are calibrated, not guessed.** Component weights come from measured cross-sectional Information Coefficients (Grinold–Kahn) on point-in-time historical panels, not intuition.
- **Scientific control groups.** Every result is compared against random 6-stock portfolios drawn from the same universe, an equal-weight passive basket, the index, and a textbook 12-1 momentum portfolio.
- **Management is measured separately from selection.** Each weekly live cohort is saved twice — once passive, once actively managed from identical picks and prices. The difference *is* the management contribution.
- **A locked data vault.** The last two years of history are held out of every backtest by default and can be opened exactly once; hypotheses for that run are pre-registered in the repository before it happens.
- **Multiple-testing correction.** The Deflated Sharpe Ratio is computed against the number of configurations actually tried (~695).
- **Negative results are reported.** Most hypotheses tested here failed, and the write-ups say so — including the ones that undercut the project's own headline claims.

---

## Architecture

| File | Lines | Role |
|---|---|---|
| `bist_analyzer.py` | ~12,700 | Everything user-facing: data fetching, scoring engines, portfolio manager, daily management, audit, storage, and every Streamlit page |
| `news_engine.py` | ~2,300 | Standalone news sentiment engine (no dependency back on the main file) |
| `pipeline_backtest.py` | ~2,350 | Full-pipeline historical simulation, controls, walk-forward, transfer tests, DSR, the data vault |
| `weight_calibration.py` | ~700 | Builds point-in-time panels and computes ICs to ground the scoring weights |
| `bilanco.py` | ~600 | Quarterly financial statements (TMS 29 aware, constant-TÜFE money) and course-note ratios |
| `skor_gunlugu.py` | ~330 | Daily durable log of every score component for the whole BIST universe |
| `daily_robot.py` | ~310 | Autonomous daily maintenance (runs on GitHub Actions) |
| `tools/` | — | KAP disclosure history, event study, phrase measurement, bounded DB maintenance |

### Core engines

`PortfolioScanner` scans the dynamic universe → `PortfolioManager` proposes portfolios by horizon × investor profile → `PortfolioDoctor` manages them daily → `PortfolioAudit` measures whether the rules are working.

The universe is dynamic and liquidity-filtered, refreshed daily: roughly 500 BIST names above a TL volume threshold, and roughly 1,400 US names above $50M daily dollar volume.

---

## Scoring model

| Component | Weight | Source |
|---|---|---|
| Technical | 60% | SMA, RSI, MACD, Bollinger, Stochastic, OBV, ADX, 52-week position |
| News sentiment | 10% | Tier-weighted RSS sources + KAP disclosures, with only price-validated directions allowed to move the score |
| Upside potential | 20% | Analyst consensus target vs. current price (TradingView, ≥3 analysts) |
| Valuation | 10% | P/E, P/B relative to sector, ROE |

When a component has no real input — no news, no analyst target — its weight moves to Technical instead of averaging in a neutral 50. Sentiment was cut from 25% to 10% after an audit showed most stocks received exactly 50, which compressed the composite's spread without adding ranking information.

**Signal bands:** 72+ Strong Buy · 57–71 Buy · 43–56 Neutral · 30–42 Sell · <30 Strong Sell

### The technical style is regime-dependent

The original contrarian scoring (rewarding oversold RSI, low Bollinger position, distance from 52-week highs) measured **zero predictive power** on BIST. The market is in a momentum regime: high RSI and high 52-week position predicted *higher* forward returns. Both the live and backtest scoring paths now use the momentum style.

### News that moves the score must have moved prices

411,656 historical KAP disclosures (2018–2024) were matched to prices in an event study. Of 16 directional hypotheses, two survived: **share buybacks** (+2.63% over 30 days vs. the market, t = 4.6) and **bonus share issues** (−3.86%, t = −5.0 — the opposite of what dilution theory predicts). Phrases whose measured effect was noise no longer vote; a positive label only raises the score when its direction is evidenced.

---

## Portfolio Manager

Proposes portfolios from a scan of the full universe, by **horizon** (short / medium / long) × **investor profile** (Cautious / Balanced / Aggressive), with sector caps, correlation filtering, ATR-inverse position sizing and a maximum single-position weight.

- **Market regime gate.** BIST: XU100 trend + BIST-30 breadth + currency pressure. US: S&P 500 trend + breadth + VIX as a *level*.
- **Minimum diversification guard.** If fewer than three stocks pass the filters, no portfolio is built.
- **Performance ledger.** Daily NAV per portfolio, reported nominal + ENAG-real + index-relative, with Sharpe, Sortino, max drawdown and Information Ratio.

### Daily active management

`PortfolioDoctor` runs every weekday after the close:

1. **Exits daily, and unbudgeted** — stop breach, trailing stop, thesis break (score collapse). Cutting risk is never rate-limited.
2. **One consolidated weight decision per position** — a one-time partial profit-take at target, then the concentration cap.
3. **Buying monthly** — on the first run of each month, toward that month's fresh picks, gated by a drawdown brake and the bear-regime gate. This matches the configuration the backtest validated; an earlier daily-buying version was measured at roughly 14 points of real return per year worse.

Cash is a real position, transaction costs are charged to NAV, and every action is written to an audit trail. `PortfolioAudit` measures stop accuracy, swap accuracy, cost burden and the managed-minus-passive spread, and only *proposes* parameter changes; applying one requires the editor key.

---

## Evidence and validation

All numbers below are on the **development window, 2018-08-31 → 2024-08-31**, with the following two years locked in the data vault. They are deliberately the unflattering ones: an earlier five-year window that included the reserved period looked far better, and that is exactly why the reserve was locked.

### Does selection beat luck?

Percentile of the system's final NAV among 50 random 6-stock portfolios from the same universe, same rebalancing, same costs; parameters chosen on BIST and applied unchanged elsewhere:

| Market | Selection only | Verdict |
|---|---|---|
| BIST | 10th | does not beat random |
| US | 94th | beats random and the S&P 500 |
| UK | 22nd | does not beat random (holds at every cost level, including zero) |

Against a **textbook 12-1 momentum portfolio** with no tuned parameters, the system's alpha is zero to negative in all three markets, and momentum alone does better in the US (final NAV 12.19 vs 4.55) and the UK (2.29 vs 0.68). The score function's value over plain momentum is not established.

**Deflated Sharpe Ratio: 0.11–0.31** across plausible trial counts, against a 0.95 threshold. The backtest alone cannot establish an edge.

### What does hold up: the management layer in BIST

| BIST | Selection only | Managed |
|---|---|---|
| Percentile vs random-6 | 10th | 62nd |
| Real return per year (ENAG) | −10.0% | **+4.8%** |
| Max drawdown | −69.9% | **−25.8%** |

For reference, XU100 returned −13.2% per year in real terms over the same window.

Measured over six consecutive sub-periods (2018–2026, before the vault existed), the managed track had the smaller drawdown in **6 of 6**. Through the 2018 currency crisis the unmanaged track drew down 69.9% while the managed one drew down 23.8%. The same rules did **not** transfer to the US, where they made both return and drawdown worse — so this is a market-specific risk overlay, not a general property of the method.

### Calibration findings

| Market | Finding |
|---|---|
| BIST | Momentum IC **+0.045 (t = 2.8)**. Strongest single signal: 52-week position. |
| US mega-cap | No stable edge in either style. |
| US mid-cap | An earlier "stable edge" did **not** replicate once the liquidity slice was corrected. |
| US, regime-split | In a bull regime 52-week position reaches IC **+0.074 (t = 8.6)**; in a bear regime the momentum score is significantly **inverted** (t = −4.7). All US proposals are blocked in a bear regime. |

### Other tests

- **Parameter search was stopped.** Walk-forward over 5 expanding folds rejected nearly every proposed change; when the training "optimum" ranges across the whole tested grid, the surface is noise.
- **Financial-health score** built from course-note ratios (20 criteria, sector-relative): pre-registered, tested once, **failed** on the managed track. A follow-up hypothesis for the unmanaged track is pre-registered for the vault opening.
- **Capacity:** with a square-root market-impact model the edge is largely intact below ~1M TL and gone by ~50M TL.

---

## Known Issues

Found in a code and calculation review in October 2026. Fixed items are listed under Roadmap history; these are still open.

**Affects reported numbers**
- **Inflation data is partly estimated.** ENAG did not exist before 2020, so 2017–2019 use official TÜFE. For 2020–2025, months without a published monthly figure are filled with one rate per year so that the year matches ENAG's published annual. Months after the last entered value use the trailing 12-month average until they are entered.
- **Price-return benchmarks against total-return portfolios.** The stock panels include dividends, while XU100, S&P 500 and FTSE 100 do not; "vs index" figures are flattered by ~2 points/yr in the US and more in the UK and BIST.
- **The backtest tests the technical core only.** News and analyst targets cannot be reconstructed historically; the live composite shares about 3 of its top 6 picks with the backtest. The live liquidity filter is also 20× looser than the backtest's.
- **Survivorship:** the universe is today's liquid list; bounded by simulation at ~2 points of real CAGR.
- **Max drawdown is measured on nominal NAV**, which understates real drawdown in high inflation.
- **Single-stock backtest drawdown** is computed from closed trades only, so intra-trade drawdowns are not counted.
- **RSI uses simple averages** rather than Wilder smoothing, so values differ from TradingView's. Kept as is because the calibration was measured with this definition.

**Behaviour and robustness**
- **US share classes with a dot** (e.g. `BRK.A`) are not converted to Yahoo's `BRK-A` format and are silently skipped.
- **No migration system for local SQLite.** A database created by a much older version can carry renamed columns (seen in the old Time Machine tables); fresh installs are unaffected.
- **Zaman Makinesi** (Time Machine) still simulates the old portfolio logic, not the validated pipeline, and a run took more than 10 minutes in testing with no way to cancel it. It is disabled for demo visitors for that reason.
- **`use_container_width`** is used ~85 times; it works on the pinned Streamlit 1.59 but is deprecated and will break on upgrade.
- **English UI translation is partial.**
- **Live evidence is short.** Live cohorts have run for weeks and carry four configuration breaks; managed portfolios also converge on the same names (35 portfolios held 49 unique stocks).
- **Transitive dependencies float.** Direct dependencies are pinned exactly; a hash-pinned lockfile is not in place yet.

---

## Roadmap

**Next**
- Open the data vault once, with the pre-registered tests: system vs random on the reserved two years, and the financial-health filter on the unmanaged track.
- Total-return benchmarks where the data source allows.

**After a few months of daily component logs**
- Forward IC of sentiment, analyst premium and valuation, which have never been validated historically. Their weights stay unchanged until then.

**Methodology**
- Combinatorial purged cross-validation and Probability of Backtest Overfitting, as a companion to the Deflated Sharpe Ratio.
- Keep the textbook momentum portfolio as a permanent control and test the score function against it, not just against random draws.

**Product**
- Rebuild Zaman Makinesi on the pipeline backtest logic.
- Map US share-class tickers, migrate off `use_container_width`, finish the English UI, add a hash-pinned lockfile.

**Fixed in the October 2026 review**
- A loss-making or negative-equity company received the *maximum* valuation score (negative P/E and P/B were read as huge discounts).
- Time Machine benchmark return was always 0 after a yfinance format change, so its "alpha" was the raw return.
- Portfolyum showed a single-stock portfolio at −100% (price read as 0); the US list could not price a single symbol.
- Information Ratio subtracted a nominal index return from an inflation-adjusted portfolio return, and US (dollar) portfolios were deflated with Turkish inflation.
- Analysis history was not saved on older local databases, and the reading side used a different column name from the writing side.
- My Portfolio crashed as soon as the first stock was added on a fresh database (the same kind of column-name mismatch), and its per-stock cards had been accidentally deleted long ago, so only "Remove" buttons were visible.
- Signal Tracker crashed whenever a period contained SELL signals but no BUY signals.
- A duplicated month in the ENAG defaults overwrote an official value with an estimate on fresh installs.
- **Inflation before 2020 was wrong.** ENAG has no data before 2020, and those months were filled with the latest year's average (~54%/yr instead of ~20%). Every real return on the 2018–2024 window was understated by about 5 points a year. Managed BIST moved from −0.8% to +4.8% per year; rankings against random portfolios did not change. 2020–2024 yearly totals now also match ENAG's published annual figures.

---

## Pages

Both markets have the same seven-page shape. The US pages are the same render functions parameterised on market, not duplicates.

**Kokpit** (default landing — market regime, macro strip, today's signals, system status) · **Portfolio Manager** · **Stock List** · **Stock Analysis** · **My Portfolio** · **Backtest** · **Signal Tracker**

Two themes (light "Gazete", dark "Obsidyen"). Market is deep-linkable via `?piyasa=US`, theme via `?tema=`.

---

## Automation

`daily_robot.py` runs on GitHub Actions twice on weekdays — after the BIST close and again after the US close. Each run updates NAV, refreshes weekly shadow and control cohorts, runs daily management and the rule audit, checks risk alerts and records the market regime. A second step writes every score component for the whole BIST universe to the database, with a run-quality row so broken days can be excluded later.

The robot probes the database before doing anything and exits non-zero on failure: an earlier incident ran green for a week while doing nothing, because success was never actually asserted.

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit + Plotly |
| Price data | yfinance |
| Universe & analyst targets | TradingView screener |
| News | feedparser (RSS sources), borsapy and the KAP API for disclosures |
| Financial statements | İş Yatırım via borsapy, converted with official TÜFE |
| Sentiment | Rule layers validated against prices; optional Turkish BERT |
| Storage | Turso (libSQL over HTTP) when configured, otherwise local SQLite — never a silent fallback between them |
| Automation | GitHub Actions |

---

## Installation

```bash
git clone https://github.com/Akif3445/bist-analyzer.git
cd bist-analyzer
pip install -r requirements.txt
streamlit run bist_analyzer.py
```

Dependencies are pinned to exact versions as a supply-chain measure — CI installs them with a database write token in the environment, so an unpinned hijacked release could exfiltrate it.

Optional `.env`:

```
ANTHROPIC_API_KEY=...      # AI-generated Turkish summaries
TURSO_DATABASE_URL=...     # cloud persistence; without it, local SQLite
TURSO_AUTH_TOKEN=...
APP_EDIT_KEY=...           # write protection (see below)
```

### Research tools

```bash
python pipeline_backtest.py run 8         # full pipeline vs XU100 / BIST-30 / random-6
python pipeline_backtest.py karsilastir 8 # managed vs unmanaged
python pipeline_backtest.py walkforward 8 5
python pipeline_backtest.py momentum 8    # system vs textbook momentum, BIST/US/UK
python pipeline_backtest.py dsr 8         # deflated Sharpe
python pipeline_backtest.py kasa          # data vault status
python tools/kap_olay.py --min-ornek 50   # KAP event study
python bilanco.py oran THYAO              # one stock's ratio history
```

`CLAUDE.md` is the full research log: every measurement, the reasoning behind each change, and every result that was later withdrawn.

---

## Security

The public demo runs with database write credentials and Streamlit has no built-in auth. If `APP_EDIT_KEY` is set, every UI write action requires unlocking via the sidebar; reads stay public. Visitors see all system portfolios and their live performance, but **My Portfolio** becomes a private sandbox for each visitor: what they add stays in their own browser session and is never stored or shown to anyone else. Long-running Time Machine jobs are disabled for visitors. External RSS content rendered as HTML is escaped and link schemes are checked. SQL is parameterised throughout. The daily robot runs with least privilege (`contents: read`).

**Do not enter real personal holdings into the public demo** — the write guard prevents pollution, not disclosure.

---

## Disclaimer

For informational and educational purposes only. This is not financial advice, and nothing here is a recommendation to buy or sell any security. The backtest does not establish a statistically defensible edge, and the live evidence spans weeks. Do your own research.

---

## License

Proprietary. All rights reserved.
