# BIST Smart Investment Assistant

A stock analysis and portfolio management platform for Borsa Istanbul (BIST) and US markets. It scores stocks 0–100 by combining technical analysis, news sentiment, analyst upside and valuation — then proposes portfolios, manages them daily, and measures whether any of it actually works.

Live demo: [bist-analyzer.streamlit.app](https://bist-analyzer-wrtwzcsmdv4z9fxrpkkrcz.streamlit.app)

> **Not investment advice.** This is a research and educational project. See [Disclaimer](#disclaimer).

---

## What makes this different

Most stock screeners tell you what to buy. The harder question is *whether the thing telling you works at all* — and that is where most of the effort here went:

- **Weights are calibrated, not guessed.** Component weights come from measured cross-sectional Information Coefficients (Grinold–Kahn) on point-in-time historical panels, not intuition.
- **Scientific control groups.** Every week the system saves its own picks as shadow portfolios *and* two controls: a random-draw portfolio and an equal-weight passive index basket. If the system cannot beat luck and passivity, it has no edge.
- **Management is measured separately from selection.** Each weekly cohort is saved twice — once passive, once actively managed from identical picks and prices. The difference *is* the management contribution.
- **Out-of-sample validation.** Rule parameters are chosen on one period and tested on another that never participated in the choice. Half the proposals produced by in-sample optimisation were exposed as noise-fitting and discarded.
- **Real returns, not nominal.** In a high-inflation economy nominal returns are close to meaningless, so performance is reported against ENAG inflation as well as the index.
- **Known biases are documented, not hidden.** Survivorship, bull-market sampling, in-sample selection and the gap between the backtest and the deployed system are all written down.

---

## Architecture

| File | Lines | Role |
|---|---|---|
| `bist_analyzer.py` | ~12,600 | Everything: data fetching, scoring engines, portfolio manager, daily management, audit, database, backtesting, and every Streamlit page |
| `news_engine.py` | ~1,950 | Standalone news sentiment engine (no dependency back on the main file) |
| `weight_calibration.py` | ~690 | Research tool: builds point-in-time panels and computes ICs to ground the scoring weights |
| `pipeline_backtest.py` | ~650 | Full-pipeline historical simulation, management-rule sweep, out-of-sample validation |
| `daily_robot.py` | ~240 | Autonomous daily maintenance (runs on GitHub Actions) |
| `tools/pmdb_admin.py` | ~90 | Bounded database maintenance commands |

### Core engines

`PortfolioScanner` scans the dynamic universe → `PortfolioManager` proposes portfolios by horizon × investor profile → `PortfolioDoctor` manages them daily → `PortfolioAudit` measures whether the rules are working.

The universe is dynamic and liquidity-filtered, refreshed daily: roughly 500 BIST names above a TL volume threshold, and roughly 1,400 US names above $50M daily dollar volume.

---

## Scoring model

| Component | Weight | Source |
|---|---|---|
| Technical | 45% | SMA, RSI, MACD, Bollinger, Stochastic, OBV, ADX, 52-week position |
| News sentiment | 25% | Tier-weighted RSS sources + KAP disclosures |
| Upside potential | 20% | Analyst consensus target vs. current price |
| Valuation | 10% | P/E, P/B relative to sector averages |

Weights were rebalanced from 35/35/20/10 in July 2026 after calibration showed news sentiment had weak predictive power. When target prices or fundamentals are missing, their weight is redistributed rather than left as a gap.

**Signal bands:** 72+ Strong Buy · 57–71 Buy · 43–56 Neutral · 30–42 Sell · <30 Strong Sell

### The technical style is regime-dependent

The original contrarian scoring (rewarding oversold RSI, low Bollinger position, distance from 52-week highs) measured **zero predictive power** on BIST. The market is in a momentum regime: high RSI and high 52-week position predicted *higher* forward returns (t > 3, stable across sub-periods). Both the live and backtest scoring paths now use the momentum style.

A segment-based hybrid was tried and rejected — mixing scales corrupts cross-sectional ranking, dropping IC from +0.045 to +0.012.

---

## Portfolio Manager

Proposes portfolios from a scan of the full universe, by **horizon** (short / medium / long) × **investor profile** (Cautious / Balanced / Aggressive), with sector caps, correlation filtering (>0.85 is "the same bet"), ATR-inverse position sizing and a maximum single-position weight.

- **Market regime gate.** BIST: XU100 trend + BIST-30 breadth + currency pressure. US: S&P 500 trend + breadth + VIX as a *level*. A dollar portfolio has no currency risk, it has a volatility regime.
- **Minimum diversification guard.** If fewer than three stocks pass the filters, no portfolio is built. A single stock at 100% weight is concentration, not caution.
- **Performance ledger.** Daily NAV per portfolio, reported nominal + ENAG-real + index-relative, with institutional metrics (Sharpe, Sortino, max drawdown, Information Ratio).

### Daily active management

`PortfolioDoctor` runs every weekday after the close and behaves like a real portfolio manager:

1. **Exits first, and unbudgeted** — stop breach, trailing stop, thesis break (score collapse). Cutting risk is never rate-limited; only opportunity-chasing is.
2. **One consolidated weight decision per position** — partial profit-take at target, then the concentration cap.
3. **Cash deployment and swaps** — gated by a drawdown brake and the bear-regime gate.

Cash is a real position: when nothing qualifies, weight waits in cash at 0% return. Discipline constraints (minimum holding period, cooldown after selling, daily trade cap) prevent churn, and **transaction costs are charged to NAV** — a frictionless NAV would overstate active management.

Every action is written to an audit trail with its reason. `PortfolioAudit` then measures whether the rules work — stop accuracy, swap accuracy, cost burden, cash drag, and the managed-minus-passive spread — and proposes parameter changes only when a finding clears a minimum sample size. Applying a change requires the editor key; the system never retunes itself, because a continuous self-optimiser with ten parameters and weeks of live data fits noise.

---

## Evidence and validation

### Calibration findings

| Market | Finding |
|---|---|
| BIST | Momentum IC **+0.045 (t = 2.8)**, stable across sub-periods. Strongest single signal: 52-week position. |
| US mega-cap | Momentum IC +0.0065 (t = 0.53) — **no edge** in the world's most efficient segment. |
| US mid-cap | Momentum IC **+0.0212 (t = 2.06)**, stable. The edge lives where analyst coverage thins out. |
| US, regime-split | In a bull regime US IC reaches **+0.053 (t = 6.0)**; in a bear regime the momentum score is significantly **inverted** (t = −3.24). Short-horizon proposals are blocked accordingly. |

### Full-pipeline backtest

Today's portfolio construction replayed point-in-time over five years, monthly rebalanced, with turnover costs charged:

| | Nominal | **Real (ENAG)** | Sharpe | Max drawdown |
|---|---|---|---|---|
| Actively managed | +4083% | **+10.9% CAGR** | 0.70 | −25.8% |
| Unmanaged | +3642% | +8.4% CAGR | 0.60 | −48.4% |
| XU100 index | +863% | **−17.3% CAGR** | — | — |

Inflation over the same window was 25×, which is exactly why nominal figures must never be quoted alone.

### Out-of-sample validation

Parameters are chosen on the first 60% of an 8-year window (which includes the 2018 currency crisis and the 2020 crash) and tested on the last 40%, which never participates in the choice. Of four proposals produced by in-sample optimisation, **two were rejected** because they performed materially worse out of sample.

### Known limitations

- The backtest exercises only the **technical core**; news and analyst inputs cannot be reconstructed historically, so it is not the deployed 45/25/20/10 system.
- The universe is today's liquid list, so delisted companies are absent (survivorship bias).
- The five-year window is a bull market, where insurance such as stops can only look like cost.
- No slippage model; costs are a flat percentage.
- Live tracking is measured in weeks, not years. Distinguishing skill from luck at a realistic Sharpe takes **years**, not weeks.

---

## Pages

Both markets have the same seven-page shape. The US pages are the same render functions parameterised on market, not duplicates.

**Kokpit** (default landing — market regime, macro strip, today's signals, system status) · **Portfolio Manager** · **Stock List** · **Stock Analysis** · **My Portfolio** · **Backtest** · **Signal Tracker**

Two themes (light "Gazete", dark "Obsidyen"). Market is deep-linkable via `?piyasa=US`, theme via `?tema=`.

The **Signal Tracker** logs every signal the system generates and checks its forward return at 1/3/7/14/30 days — live production tracking, not backtesting.

---

## Automation

`daily_robot.py` runs on GitHub Actions twice on weekdays — after the BIST close and again after the US close, since deciding on US positions mid-session would use half-day prices. Each run:

1. Updates NAV for every active portfolio (per market)
2. Refreshes weekly shadow and control cohorts
3. Runs daily portfolio management
4. Runs the rule audit
5. Checks risk alerts
6. Records the market regime
7. Handles monthly calibration and inflation-data reminders

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit + Plotly |
| Price data | yfinance |
| Universe & analyst targets | TradingView screener |
| News | feedparser (20+ RSS sources), borsapy for KAP disclosures |
| Sentiment | Turkish BERT (optional) with keyword fallback |
| Storage | Turso (libSQL over HTTP) with local SQLite fallback |
| Automation | GitHub Actions |

---

## Installation

```bash
git clone https://github.com/Akif3445/bist-analyzer.git
cd bist-analyzer
pip install -r requirements.txt
streamlit run bist_analyzer.py
```

Dependencies are pinned to exact versions as a supply-chain measure — CI installs them with a database write token in the environment, so an unpinned hijacked release could exfiltrate it. Bump versions deliberately, not automatically.

Optional `.env`:

```
ANTHROPIC_API_KEY=...      # AI-generated Turkish summaries
TURSO_DATABASE_URL=...     # cloud persistence; falls back to local SQLite
TURSO_AUTH_TOKEN=...
APP_EDIT_KEY=...           # write protection (see below)
```

Optional Turkish BERT sentiment (~2 GB, not in requirements because Streamlit Cloud's free tier is RAM-constrained):

```bash
pip install transformers torch
```

### Research tools

```bash
python weight_calibration.py build-tech      # build the point-in-time panel
python weight_calibration.py analyze         # IC analysis + suggested weights
python pipeline_backtest.py run 5            # full-pipeline simulation
python pipeline_backtest.py sweep 5          # management-rule sweep
python pipeline_backtest.py validate 8       # out-of-sample validation
python tools/pmdb_admin.py enag-list         # database maintenance
```

---

## Security

The public demo runs with database write credentials and Streamlit has no built-in auth. If `APP_EDIT_KEY` is set, every UI write action — saving portfolios, editing the inflation table, changing rule parameters — requires unlocking via the sidebar; reads stay public. With the key unset (local development) writes are unguarded and behaviour is unchanged.

External RSS content rendered as HTML is escaped and link schemes are checked. SQL is parameterised throughout. The daily robot runs with least privilege (`contents: read`).

**Do not enter real personal holdings into the public demo** — the write guard prevents pollution, not disclosure.

---

## Disclaimer

For informational and educational purposes only. This is not financial advice, and nothing here is a recommendation to buy or sell any security. All investment decisions carry risk, past performance does not guarantee future results, and the live evidence for this system currently spans weeks — far too short to establish skill. Do your own research.

---

## License

Proprietary. All rights reserved.
