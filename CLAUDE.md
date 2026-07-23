# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BIST Smart Investment Assistant — a Streamlit app that scores BIST (Borsa Istanbul) and US stocks 0-100 by combining technical analysis, news sentiment, analyst target upside, and valuation, then generates buy/sell signals. Single-user, runs locally or on Streamlit Cloud. Not financial advice (disclaimer is baked into the UI).

Live demo: https://bist-analyzer-wrtwzcsmdv4z9fxrpkkrcz.streamlit.app

## Commands

```bash
streamlit run bist_analyzer.py          # run the app (also: Streamlid.bat)
python test_backtest.py                 # ad-hoc manual backtest sanity check (not pytest, just run it)
python test_rss.py                      # check which RSS news sources are currently reachable
```

There is no test framework (no pytest) and no lint config — `test_*.py` files are standalone scripts you run and read the printed output of, not asserts.

`.env` needs `ANTHROPIC_API_KEY` for the AI Turkish summary feature (Claude Opus). Optional Telegram vars exist in some modules but core app runs fine without them (graceful degrade).

## Architecture

Two files carry the entire app:

- **bist_analyzer.py** (~8500 lines) — everything: data fetching, scoring engines, database, backtesting, time machine simulation, and every Streamlit page (`render_*_page` functions). Entry point is `run_app()` at the bottom, which picks BIST vs US market from a top bar, then renders one of ~15 pages selected from the sidebar per market.
- **news_engine.py** (~1600 lines) — standalone news sentiment engine, imported by bist_analyzer.py. Has no dependency back on bist_analyzer.py, so it can be read/edited in isolation.

`faz2_modules.py` and `test_*.py` are gitignored — local-only, not part of the deployed/tracked app.

### Data flow for one stock score

`DataFetcher` (yfinance, `.IS` suffix auto-added for BIST) → `TechnicalEngine` + `ValuationEngine` + `analyze_news()` (news_engine.py) → `compute_bist_score()` combines them into a `BISTScore` → `RiskEngine` derives stop-loss/target levels → rendered by a `render_*_page` function.

`BISTScore.stock` field uses `field(default_factory=lambda: StockData(ticker=""))` — `StockData` requires `ticker` as a mandatory arg, don't remove the lambda.

### Scoring weights (WEIGHTS dict ~line 134, applied in compute_bist_score)

Technical 45% + News Sentiment 25% + Upside Potential 20% + Valuation 10% (rebalanced 2026-07 from 35/35/20/10 based on weight_calibration.py IC evidence — see below). When target price or fundamentals are missing, their weight redistributes to Technical/Sentiment rather than leaving a gap. Analyst targets auto-fetch from TradingView screener (`fetch_tv_analyst_targets`, single bulk query cached 4h, min 3 analysts, ~45 BIST names covered) — manual sidebar input still overrides; the old faz2 TargetPriceFetcher path is effectively dead (module doesn't import). Signal bands: 72+ Strong Buy, 57-71 Buy, 43-56 Neutral, 30-42 Sell, <30 Strong Sell.

### News sentiment (news_engine.py)

Three-layer classification, in order: (1) regex strong-pattern override for unambiguous events, (2) BERT (`savasy/bert-base-turkish-sentiment-cased`, optional — not in requirements.txt because Streamlit Cloud free tier is RAM-constrained; app auto-falls back), (3) weighted keyword lists (120+/98+ terms, severity-weighted). Sources are tier-weighted (`_get_tier`/`_get_weight`, tier 1 = 3x ... tier 3 = 1x) and each source has its own `_src_*` fetcher function using feedparser. Recency weighting, duplicate-source confidence boosting, and a speculation filter (`_is_speculation`) are applied after classification.

### Persistence

Everything lives in one SQLite file (`bist_cache.db`, path resolves to `/tmp` on Streamlit Cloud, else app dir) via `AnalysisHistoryDB` (~line 348). It owns ~10 tables: analysis_history, alerts, accuracy_log, portfolio, backtest_trades/daily/summary, news_backtest_cache, accuracy_validation, plus `TimeMachineEngine`'s own run/pick/daily tables. There's no migration system — schema changes are additive `CREATE TABLE IF NOT EXISTS` blocks in `__init__`.

### Backtesting vs Time Machine — don't confuse these

- `BacktestEngine` (~line 3197): tests one strategy mode (swing/trend/universal/investor/buy&hold) against one ticker's history, point-in-time to avoid look-ahead bias.
- `TimeMachineEngine` (~line 2483): full portfolio simulation — picks stocks using the live scoring model at each historical point, then walks the portfolio forward. Six portfolio styles (Aggressive/Defensive/Momentum/Value/Stable/Custom).

### Portfolio Manager v2 (2026-07)

"Portfoy Yoneticisi" page (render_portfolio_manager_page): `InflationEngine` (ENAG E-TÜFE monthly table in `enag_monthly`, seeded from ENAG_MOM_DEFAULTS, user-extendable via UI; deflator() day-prorates months) converts nominal returns to real. PM tables go through `_PMDB` storage layer: if `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN` secrets are set, Turso HTTP API (v2/pipeline, pure requests — no libsql package, it won't build on Win/py3.14) gives persistence on Streamlit Cloud; otherwise falls back to local SQLite (and also falls back at runtime on any Turso error). `compute_market_regime()` (cached 30min) scores XU100 trend + BIST30 breadth + USDTRY into Boğa/Nötr/Ayı — Ayı regime blocks short-horizon proposals. `PortfolioManager` proposes portfolios from `PortfolioScanner.scan_all()` results by horizon (kisa/orta/uzun) × profile (Temkinli/Dengeli/Agresif) with sector caps and ATR-based stop/target; saved portfolios live in pm_portfolios/pm_positions/pm_nav, NAV updated once per session in run_app, performance reported as nominal + ENAG-real + XU100-relative. Rebalance suggestions flag stop breaches / score<43 (ÇIKAR), target hits (KÂR AL) and top candidates (EKLE). Min-position guard (2026-07-22): propose() refuses to build a portfolio when <3 stocks pass the filters (returns empty picks + explanation) — the W29 kisa/Temkinli shadow had saved a single stock at 100% weight (TUPRS), which is concentration, not caution; the shadow batch now skips such combos for the week. Comparison chart defaults to showing only the LATEST weekly shadow cohort (radio: Son kohort/Tümü/Yok) — older cohorts keep accruing NAV in the background and aggregate in the scoreboard; without this the chart gains 11 lines/week.

### Signal Tracker

Every signal the app generates gets logged and its forward return is checked at 1/3/7/14/30 days (`check_pending_signals`, called once per session in `run_app()`). This is what "Sinyal Takip" pages display — it's live production tracking, not backtesting.

### Weight calibration (weight_calibration.py)

Standalone research tool that builds a point-in-time panel (50 BIST stocks × 6y × weekly, `calibration_tech.csv`) plus a GDELT daily news-tone panel (`calibration_sentiment.csv`) and computes cross-sectional ICs (Grinold-Kahn) to ground the compute_bist_score weights in data. Key 2026-07 findings, now applied to the code: the original contrarian technical score (high points for oversold RSI/BB/52w) had ~zero IC because BIST is in a momentum regime (high RSI/52w-position predicted HIGHER returns, t>3, stable across sub-periods; strongest signal = week52_position). Both scoring paths (`TechnicalEngine._compute_score` for live, `BacktestEngine._vectorized_scores` for backtest/time-machine) now take a `style` param — "momentum" (all BIST, selected via `_tech_style_for`) or "dengeli" (original contrarian, still used for US). A segment-based hybrid (contrarian for BIST-30) was tried and REJECTED: mixing scales corrupts cross-sectional ranking (IC drops from +0.045 to +0.012). A/B backtest (8 stocks, 3y, universal mode): avg total return +1.9% → +13.4%, but not uniform (THYAO notably worse). GDELT news tone was weak everywhere (t<1), so sentiment's 35% weight is likely too high — weight rebalance pending. Full run: `build-tech` (~3 min), `build-sentiment` (~45 min, 5.5s/req rate limit, ~15/55 tickers succeed per pass), `analyze`. `analyze-stops` (2026-07, 2730 top-quintile entries) showed tight stops rank dead last (old kisa 1.5×ATR was 24/24) and hard profit-targets cut momentum winners badly; stops were widened to 2.5-3.5×ATR and targets became informational levels (alerts suggest, never auto-exit). Caveat: sample is a bull market — wide stops kept as disaster insurance alongside regime gate + drawdown brake.

## Roadmap — October professor presentation (agreed 2026-07-16)

Goal: system that credibly beats index and targets ENAG-real returns; academic-grade methodology matters as much as returns. Live shadow tracking needs 4-8 weeks — meanwhile, in priority order:

- **A. Full-pipeline historical simulation** (the "what if results disappoint" insurance): run TODAY's portfolio construction (dynamic universe → momentum score → sector cap → correlation filter → ATR weights → monthly rebalance) point-in-time over 5-6 years → monthly NAV series, reported nominal + ENAG-real + vs XU100. This becomes the presentation's backbone chart. NOT the old TimeMachine (that uses different logic).
- **B. Scientific control groups in the weekly shadow batch**: random-pick portfolio (6 stocks by lottery from universe) + equal-weight BIST-30 — "does the system beat luck and passive?"
- **C. Institutional metrics on NAV curves**: Sharpe, Sortino, max drawdown, Information Ratio, win/loss ratio; add transaction-cost/slippage assumption to NAV (currently frictionless).
- **D. Monthly recalibration loop**: rerun weight_calibration 2-3× before October, produce a "parameter stability" table.
- **E. Presentation packaging (September)**: methodology report + charts.

Expectation framing agreed with user: promise = "systematically better than the market; reaches real (ENAG) targets when the market allows, limits losses when it doesn't" — beating ENAG in all conditions is not promisable (even XU100 fails some years).

## UI state (2026-07-17)

BIST menu: Kokpit (default landing, "Piyasa Defteri" masthead + macro strip + signals + system status) · Portfoy Yoneticisi · BIST Listesi · Hisse Analizi · Portfolyum · Backtest · Sinyal Takip. "Piyasa Ozeti" and "Sistem Portfolyleri" retired from menu 2026-07-17 (render functions kept in code; macro strip moved to Kokpit via `_kokpit_macro`). Two themes via sidebar "Baskı" selector persisted in `?tema=` query param: Gazete (light, Playfair/Source Serif, default) and Obsidyen (dark graphite, Inter, mint accent) — palette in `THEMES`, all charts/inline colors read `_theme()`. Mobile rule learned the hard way: NEVER put primary controls in the sidebar (collapsed on phones) — backtest and analysis inputs were both moved to main area for this reason.

**UI backlog (user-approved deferrals):** EN translations half-done (option kept visible for now); US Markets tab unmaintained (kept visible); Zaman Makinesi still simulates the OLD portfolio logic (warning caption added; eventual rewrite on pipeline_backtest logic); Kokpit "Günün Sinyalleri" depends on scan cache (could use an inline scan button).

## Security

The public Streamlit Cloud demo runs with Turso WRITE credentials and Streamlit has no built-in auth, so an anonymous visitor could otherwise corrupt shared data (portfolios, NAV, and especially the ENAG inflation table that every real-return figure depends on). Write protection (2026-07-23): if the `APP_EDIT_KEY` secret is set, all UI write actions (portfolio save, ENAG rate edit, archive) require unlocking via the sidebar "Editör girişi" box (`hmac.compare_digest`); reads stay public. If the secret is unset (local dev), writes are unguarded — behaviour unchanged. Helpers: `_writes_locked()` / `_guard_write()` near `_get_secret`. The daily robot writes to Turso directly (not through the UI), so it's unaffected — **do not** route robot writes through `_guard_write`. To arm protection on Streamlit Cloud, set `APP_EDIT_KEY` in the app's secrets. ENAG `set_rate` UI input is also validated (YYYY-MM format + −50…100% bound). External RSS content rendered via `unsafe_allow_html` is `html.escape()`d and links are scheme-checked (http/https only) — see news rendering in bist_analyzer.py `_render_kokpit`/economy-news and news_engine.py `render_news_panel`. SQL is parameterized throughout; no eval/exec; no hardcoded secrets (all via `_get_secret`).

## Known constraints

- Turkish news RSS sources are unreliable — several return 404 or their feeds don't mention the ticker in the title, so `_is_relevant` filters them out. Google News TR is the most consistently working source. `test_rss.py` is the way to check current source health before debugging "why is sentiment score always 50" (low article count triggers a confidence pull toward neutral 50).
- KAP (Turkish public disclosure platform) is bot-protected — direct API/RSS access doesn't work. KAP disclosures are fetched via the third-party `borsapy` library instead (`_src_kap` in news_engine.py, graceful degrade if borsapy missing). borsapy's heavy deps (openai, pymupdf, onnxruntime) are lazy-loaded, so importing it costs no RAM.
- Comments in bist_analyzer.py are in Turkish; user-facing UI strings are Turkish (with an EN/TR/BOTH language toggle for AI-generated summaries).
