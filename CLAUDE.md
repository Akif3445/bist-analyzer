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
python tools/pmdb_admin.py enag-list    # PM DB maintenance: enag-list | enag-delete YYYY-MM | portfolios
```

`tools/pmdb_admin.py` exists so DB maintenance doesn't require handing Claude Code a blanket "run any Python" permission — the local `.claude/settings.local.json` allows only `python tools/pmdb_admin.py:*`, so the possible operations are bounded by this script's subcommands. Add new maintenance actions as subcommands here rather than widening the permission rule.

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

### Active daily management — PortfolioDoctor (2026-07-24)

`PortfolioManager` *builds* portfolios; `PortfolioDoctor` *manages* them over time. User decisions: management is **fully automatic** (no per-trade confirmation), profile is **"Dengeli — kurumsal standart"**, and it must behave like a real PM (may buy, sell, or sit in cash).

**NAV accounting was rewritten to allow rebalancing.** The old formula assumed a fixed `entry_price` and no cash. NAV is now *chained*: `pm_portfolios.nav_base` + `rebalanced_at`, positions may sum to <100 with the remainder held as **cash** (0% return), and `NAV = nav_base × factor_since_last_rebalance` (`PortfolioManager._nav_factor`). At each rebalance the current NAV is frozen into `nav_base` and all surviving positions get their `entry_price` reset to that day's price. Verified numerically identical to the old formula when weights sum to 100 with no cash, so pre-existing portfolios are unaffected. `pm_positions` gained `peak_price` (trailing stop) and `opened_at` (min hold); new `pm_trades` table is the audit trail.

Daily order of operations in `daily_manage(market)`: **exits first and unbudgeted** (stop breach → trailing stop 15% off peak → score<43, the last respecting min hold) — cutting risk is never rate-limited, only opportunity-chasing is. Then **one consolidated weight decision per position** (partial 1/3 profit-take at target, then the 30% concentration cap; these were two separate actions at first and the second silently undid the first *and* corrupted the cash figure). Then **cash deployment / swaps**, gated by the drawdown brake (15% off portfolio peak) and the bear-regime gate, with swaps requiring a 12-point score edge and respecting the sector cap. Discipline layer: 5-day min hold, 10-day cooldown before rebuying, max 2 discretionary trades/day. **Transaction costs are charged to `nav_base`** (0.20% BIST / 0.10% US of traded weight) — a frictionless NAV would overstate active management, which is exactly what a reviewer would attack.

**Scientific integrity — do not collapse these two tracks.** The existing passive shadow cohorts are the evidence that *selection* beats luck/passive, and their controls are buy-and-hold. Turning management on for them mid-flight would make W29+ uninterpretable. So `ensure_shadow_batch` now saves a **parallel "Yönetilen" cohort** from the identical picks with `managed=1`, while the passive "Gölge" cohort is untouched. Same day, same prices → the difference *is* the management contribution. User portfolios (`kind="kullanici"`) default to `managed=1`.

Robot runs the management pass for both markets. `.github/workflows/daily-robot.yml` has a **second cron at 21:30 UTC** (after the US close) with `ROBOT_ONLY_MARKET=US`, because the 15:40 UTC run happens mid-session for US and would decide on half-day prices.

### Rule audit + editor-approved parameters (2026-07-28)

`PortfolioAudit` runs in the robot every day, measures whether the management rules are actually working, writes to `pm_audit`, and **changes nothing on its own**. Metrics: **stop accuracy** (did a stopped-out stock keep falling? if it rebounded the stop was too tight), **swap accuracy** (did the replacement beat what was sold?), **cost burden** (commission paid ÷ gross NAV movement), **managed−passive spread** (the parallel cohorts — the actual question), and average cash. Exit/swap events only count once ≥5 days old.

**Why this is an audit and not an optimizer.** The user asked for a mechanism that continuously steers toward what works. That exact thing is dangerous: with 10 rule parameters, 6 years of history and only *weeks* of live data, a continuous self-tuner fits noise, chases whatever regime just ended, and destroys the ability to say "these rules produced this result" — which is the presentation's whole claim. So the design splits: **audit is continuous and automatic; parameter changes are proposals, gated, and rare.** Every finding is hard-gated on a minimum sample (`PortfolioAudit.MIN_ORNEK`, e.g. 8 stop events) — below that the metric is still recorded but produces no proposal.

**Parameter overrides.** `PortfolioDoctor.kural(key)` reads `pm_params` first and falls back to the `KURAL` dict in code, so code values remain the documented defaults and any change is reversible. Writing an override goes through `kural_ayarla`, and the UI button is behind `_writes_locked()` / `_guard_write()` — **only the unlocked editor can change a parameter** (user decision 2026-07-28). Each change stores date + reason, so "when did I change what, on what evidence" is answerable.

UI cost is deliberately near zero: one line in the Portfolio Manager (`🩺 Yönetim denetimi — sağlıklı`), which expands only when there is a finding; the parameter table appears only if overrides exist.

### Management-rule sweep (2026-07-30) — `pipeline_backtest.py sweep`

`_walk_managed` replays the daily management layer (trailing stop, score exit, position cap, drawdown brake, swap threshold) over the same point-in-time pipeline, and `sweep` varies one parameter at a time. Acceptance is deliberately strict: a change is only proposed if the return difference exceeds 5% **and** the better value wins in nearly every calendar year. Proposals are written to `pm_param_oneri` and **never applied automatically** — the UI surfaces them next to the live-audit findings behind the same editor gate.

**This sweep immediately paid for itself by finding a serious bug.** The drawdown brake measured drawdown from the **all-time** peak (`cummax`), which made it a permanent trap: once 15% below the peak, buying stopped, stops kept selling into cash, cash returns 0% so NAV could never recover, and the all-time peak never declined — so the portfolio stayed braked forever. Five-year backtest return with the brake on was **+102%** vs **+3311%** with it off. Fixed in both the live engine and the backtest by measuring drawdown from a **90-day rolling peak**, which releases naturally.

**After the fix, active management clearly adds value** (5y, BIST, monthly rebalance): return **+4083% vs +3642%** unmanaged, Sharpe **3.33 vs 2.14**, MaxDD **−25.8% vs −48.4%**. Roughly half the drawdown for a better return — this is the strongest evidence so far that the management layer is worth its cost (which the sim charges at ~12% cumulative turnover cost).

Two proposals passed the gate and await editor approval: `takas_farki` 12→6 (+6466%, 6/6 years) and `max_pozisyon` 30→40 (+90%, 5/6 years). `trailing_pct` scored best when **disabled** (+7344%) but was correctly rejected at 4/6 years — and that rejection is the point: the sample is a 5-year bull market, so any disaster insurance looks like pure cost. Same reasoning as the earlier `analyze-stops` finding. The discipline parameters (`min_tutma_gun`, `bekleme_gun`, `gunluk_islem`) are deliberately **not** swept — they are anti-churn constraints, not free variables.

### Backtest results in REAL terms, and the bias list (2026-07-30)

**Never quote the nominal number alone.** Over the 5-year window (2021-07-30 → 2026-07-30) ENAG cumulative inflation was **25×** (+2398%), so nominal figures are close to meaningless:

| | Nominal | Nominal CAGR | **Real** | **Real CAGR** |
|---|---|---|---|---|
| Managed | +4083% | 111% | **+67%** | **+10.9%** |
| Unmanaged | +3642% | 106% | +50% | +8.4% |
| XU100 | +863% | 57% | **−61%** | −17.3% |

The defensible sentence is "~11% real CAGR while the index lost 61% in real terms", not "40× in five years". Management contributes ~2.5 points of real CAGR and roughly halves the drawdown (−25.8% vs −48.4%, Sharpe 3.33 vs 2.14).

**Known biases, ranked.** Two are serious:

1. **The backtest tests a DIFFERENT system than the live one.** It uses the technical score only (`BacktestEngine._vectorized_scores`); the live score is 45% technical + 25% news + 20% analyst upside + 10% valuation. News and analyst data cannot be reconstructed historically, so these results describe the **technical core**, not the deployed system. State this explicitly.
2. **Parameter selection was in-sample** — `sweep` finds the best value on the same data it measures. Addressed by `validate` below.
3. **Survivorship / universe selection** — the universe is *today's* liquid list, so companies that delisted or shrank are absent (optimistic). Partly mitigated by applying each date's own volume filter.
4. **Bull-market sample** — the 5-year window is entirely a BIST upswing, where insurance (stops, brakes) can only look like cost. Addressed by using an 8-year window in `validate`, which contains the 2018 TL crisis (−28% drawdown) and the 2020 Covid crash (−32%).
5. **No slippage model** — cost is a flat percentage; in reality size moves price in small caps, which the universe contains.
6. **One-factor-at-a-time** — parameter interactions unexplored.

### Out-of-sample validation — `pipeline_backtest.py validate` (2026-07-30)

The answer to "did the system find something real or fit noise?". The window is split: the best parameter is chosen on the **first 60%** (train, 2018-07→2023-05) and then tested on the **last 40%** (test, 2023-05→2026-07), which never participates in the choice.

**Two of four sweep proposals were exposed as noise-fitting:**

| Parameter | Train picks | Test: default | Test: picked | Verdict |
|---|---|---|---|---|
| `trailing_pct` 15→10 | 10 | +1309% | **+951%** | rejected |
| `dusus_freni` 15→10 | 10 | +1309% | **+594%** | rejected |
| `max_pozisyon` 30→40 | 40 | +1309% | +1382% | validated |
| `takas_farki` 12→0 | 0 | +1309% | **+3176%** | validated |

`validate` overwrites `pm_param_oneri` with out-of-sample evidence and **deletes** proposals that failed, so the editor only ever sees survivors. Note the direction on `takas_farki` is robust (lower is better, confirmed on both windows) but the exact value is not — the 5-year sweep said 6, the 8-year validation says 0. Prefer the conservative end.

**Remaining honesty gap:** the test period (2023-05→2026-07) is itself mostly bullish, so parameters were validated on a bull holdout. The crisis periods sit in the *training* half. A stricter design would reverse the split or use walk-forward with multiple folds.

### Signal Tracker

Every signal the app generates gets logged and its forward return is checked at 1/3/7/14/30 days (`check_pending_signals`, called once per session in `run_app()`). This is what "Sinyal Takip" pages display — it's live production tracking, not backtesting.

### Weight calibration (weight_calibration.py)

Standalone research tool that builds a point-in-time panel (50 BIST stocks × 6y × weekly, `calibration_tech.csv`) plus a GDELT daily news-tone panel (`calibration_sentiment.csv`) and computes cross-sectional ICs (Grinold-Kahn) to ground the compute_bist_score weights in data. Key 2026-07 findings, now applied to the code: the original contrarian technical score (high points for oversold RSI/BB/52w) had ~zero IC because BIST is in a momentum regime (high RSI/52w-position predicted HIGHER returns, t>3, stable across sub-periods; strongest signal = week52_position). Both scoring paths (`TechnicalEngine._compute_score` for live, `BacktestEngine._vectorized_scores` for backtest/time-machine) now take a `style` param — "momentum" (all BIST, selected via `_tech_style_for`) or "dengeli" (original contrarian, still used for US). A segment-based hybrid (contrarian for BIST-30) was tried and REJECTED: mixing scales corrupts cross-sectional ranking (IC drops from +0.045 to +0.012). A/B backtest (8 stocks, 3y, universal mode): avg total return +1.9% → +13.4%, but not uniform (THYAO notably worse). GDELT news tone was weak everywhere (t<1), so sentiment's 35% weight is likely too high — weight rebalance pending. Full run: `build-tech` (~3 min), `build-sentiment` (~45 min, 5.5s/req rate limit, ~15/55 tickers succeed per pass), `analyze`. `analyze-stops` (2026-07, 2730 top-quintile entries) showed tight stops rank dead last (old kisa 1.5×ATR was 24/24) and hard profit-targets cut momentum winners badly; stops were widened to 2.5-3.5×ATR and targets became informational levels (alerts suggest, never auto-exit). Caveat: sample is a bull market — wide stops kept as disaster insurance alongside regime gate + drawdown brake.

## US / NASDAQ parity (started 2026-07-23)

Goal: bring the US side to the same shape as BIST. User decisions: **nominal USD only** (no inflation/real-return layer on the US side — ENAG is Turkish, and a US-CPI layer was declined), **dynamic S&P-500 + NASDAQ-100 universe**, **targeted at the October presentation**.

Shared infrastructure is market-parameterized rather than duplicated: `get_scan_universe(market)`, `PortfolioScanner.scan_all(market=)` (separate cache table `portfolio_scan_us`), `compute_market_regime(market)`, and `PortfolioManager.propose/ensure_shadow_batch/ensure_control_batch/update_navs(market=)`. `pm_portfolios` has a `market` column (idempotent migration; old rows read as BIST via `COALESCE(market,'BIST')`). Every call site defaults to `market="BIST"`, so BIST behaviour is unchanged.

US specifics: liquidity threshold `UNIVERSE_MIN_USD_VOLUME` = $200M/day → ~596 names (measured: 50M→1395, 100M→975, 150M→745, 200M→603, 300M→443; $200M matches the S&P-500 + NASDAQ-100 union of ~600). Regime uses S&P 500 trend + breadth + **VIX as a level** (<22 calm) instead of USDTRY — a dollar portfolio has no currency risk, it has a volatility regime. Benchmark is `_default_index(market)` = `^GSPC`; the `pm_nav.xu100_close` column keeps its name for compatibility but holds the portfolio's own index. `performance()` returns `reel=None` for US. Passive control group is the 100 most liquid US names (NASDAQ-100 proxy) instead of BIST-30.

UI: the US menu mirrors BIST exactly (US Kokpit · US Portfoy Yoneticisi · US Listesi · US Analiz · Portfolyum · US Backtest · US Sinyal Takip). `render_kokpit_page` and `render_portfolio_manager_page` take a `market` param instead of being duplicated — the US menu calls the same functions with `market="US"`. Helpers: `_bench_ad(market)` for the benchmark label, `_reel_var(market)` for whether a real-return column exists. The ENAG tab is only built for BIST, and real-return metrics render "—" when `perf['reel']` is None. **Market is deep-linkable via `?piyasa=US`** (same pattern as `?tema=`) and persists across reruns — this is also the only way to drive the US side from browser automation, since the market selector is an iframe component whose clicks don't propagate.

**Robot:** `daily_robot.py` loops both markets for the weekly shadow/control cohorts (each market wrapped separately so one failing doesn't stop the other; freshness checked per market). First US cohort landed 2026-W30: 9 shadow + 2 control (random-draw + NDX100 passive with 99 names). When the US shadow set comes back empty the robot prints why — in Ayı that is the calibration brake firing, not a failure. `check_risk_alerts()` and `update_navs()` also run per market; both previously built symbols as `t + ".IS"` for every portfolio, so **US positions would silently never price and never raise a stop/target alert** — fixed 2026-07-23.

### US calibration finding (2026-07-23) — the edge is in mid-caps, not mega-caps

Two point-in-time panels, same Grinold-Kahn method as BIST (120 stocks × 6y × weekly, ~28k observations each), via `weight_calibration.py build-tech-us` / `build-tech-us-mid` + `analyze-us` / `analyze-us-mid`:

- **Mega-cap (120 most liquid):** momentum IC +0.0065 (t=0.53) — insignificant, sign flips across sub-periods. Contrarian IC −0.0136 (t=−1.31), consistently on the *wrong* side. No technical edge in the world's most efficient segment.
- **Mid-cap (liquidity ranks 300–420):** momentum IC +0.0212 (**t=2.06**), **stable across sub-periods**. 52-week position t=2.14, RSI t=2.27 also signal.

So `_tech_style_for` now returns `"momentum"` for **both** markets. Effect measured on a live rescan: score↔1M-momentum correlation went **−0.354 → +0.323**, and the top-10 picks went from 14-of-20 down-on-the-month to **0-of-10**. Before this, the US top pick was a stock down 43% in a month.

**Caveat to state plainly in the presentation:** US IC (+0.021) is far below BIST's (+0.045, t=2.8). The US ranking is a *weak* signal and must not carry a strong claim on its own.

**Acted on (2026-07-23):** the universe threshold was lowered $200M → $50M (596 → 1406 names) to *add* the mid-cap band where the edge actually lives, without removing any mega-caps. Horizons were extended to 60/90 days because mid-cap IC was still rising at 30d.

**Longer-horizon result — read this before quoting any t-stat.** The best US signal found is **52-week position at 90 days: IC +0.0323, t=3.95** — the same indicator that dominates BIST. But the year-by-year breakdown shows it is **episodic, not persistent**: 2023 +0.101 (t=6.97) and 2026 +0.117 (t=3.67) carry almost all of it, while 2021 (−0.017), 2024 (−0.010) and 2025 (+0.016) are flat or negative. The half-sample "stable" label is technically true and substantively misleading. This is deliberately **not** wired into scoring. The honest reading: buying names near their 52-week highs paid off spectacularly in strong momentum years and did nothing otherwise — a regime-dependent effect, not a durable edge. A natural follow-up is to gate it on `compute_market_regime` — **this was then tested, see below.**

### Regime-conditional result (2026-07-23) — the US signal is conditional, not weak

`weight_calibration.py analyze-us-regime` rebuilds the historical regime point-in-time with the same 0-7 scheme as the live engine (SPX vs SMA200, golden cross, 1m momentum, breadth taken from the panel's own `above_sma200` column so there is no look-ahead, VIX), then splits IC by regime. On 26.5k observations at the 90-day horizon:

| Signal | Boğa | Nötr | Ayı |
|---|---|---|---|
| 52-week position | **+0.0528 (t=6.02)** | +0.002 | −0.029 |
| Momentum score | **+0.0448 (t=4.39)** | −0.015 | **−0.063 (t=−3.24)** |

Two conclusions. **In a bull regime US IC is on par with BIST's headline +0.045** — the earlier "US is half as strong" reading was an artifact of averaging across regimes. And **in a bear regime the momentum score is significantly inverted**: following it loses money.

Applied: `propose()` refuses *every* horizon for US in Ayı (previously only `kisa` was blocked) and warns on Nötr. BIST is unchanged — there is no equivalent evidence for it.

**Caveat for the presentation:** Boğa is 72% of the sample and the regime split largely coincides with the strong years (2023, 2026). What makes this a usable filter rather than hindsight is that *regime is observable at decision time, a year label is not*. Still one market, one 6-year window — present it as a conditional finding, not a law.

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

The public Streamlit Cloud demo runs with Turso WRITE credentials and Streamlit has no built-in auth, so an anonymous visitor could otherwise corrupt shared data (portfolios, NAV, and especially the ENAG inflation table that every real-return figure depends on). Write protection (2026-07-23): if the `APP_EDIT_KEY` secret is set, all UI write actions (portfolio save, ENAG rate edit, archive) require unlocking via the sidebar "Editör girişi" box (`hmac.compare_digest`); reads stay public. If the secret is unset (local dev), writes are unguarded — behaviour unchanged. Helpers: `_writes_locked()` / `_guard_write()` near `_get_secret`. The daily robot writes to Turso directly (not through the UI), so it's unaffected — **do not** route robot writes through `_guard_write`. To arm protection on Streamlit Cloud, set `APP_EDIT_KEY` in the app's secrets. ENAG `set_rate` UI input is also validated (YYYY-MM format + −50…100% bound). Local-SQLite user-data writes (Portfolyum add/remove, history delete) and the expensive ~500-stock rescan are guarded too — on Streamlit Cloud `DB_PATH` is `/tmp/bist_cache.db`, a SINGLE file shared by every anonymous visitor of the demo, so those tables are multi-tenant by nature. **Do not enter real personal holdings into Portfolyum on the public demo** — reads stay public, so any visitor can see them; the write guard stops pollution, not disclosure. `.github/workflows/daily-robot.yml` runs with least privilege (`permissions: contents: read`). **Dependencies are pinned exactly (`==`) in requirements.txt as of 2026-07-23** — supply-chain protection: CI runs `pip install` with `TURSO_AUTH_TOKEN` in env, so with `>=` a hijacked release of a small package (borsapy, tradingview-screener, streamlit-option-menu) would auto-install and could exfiltrate the token. Pins were verified resolvable for CI's Python 3.12 on Linux via `pip install --dry-run --only-binary=:all: --no-deps --python-version 3.12 --platform manylinux_2_28_x86_64` (note: pandas 3.x has no manylinux2014 wheel — it needs glibc ≥2.28, which ubuntu-latest and Streamlit Cloud both satisfy; re-run this check whenever pins change). Trade-off: **automatic security updates stop, so bump versions manually every few months** — `pip install -U <pkg>` → run the app and verify → update the number here → commit. Only direct dependencies are pinned; transitive ones still float (a hash-pinned lockfile via pip-tools/uv is the remaining open item). External RSS content rendered via `unsafe_allow_html` is `html.escape()`d and links are scheme-checked (http/https only) — see news rendering in bist_analyzer.py `_render_kokpit`/economy-news and news_engine.py `render_news_panel`. SQL is parameterized throughout; no eval/exec; no hardcoded secrets (all via `_get_secret`).

## Known constraints

- Turkish news RSS sources are unreliable — several return 404 or their feeds don't mention the ticker in the title, so `_is_relevant` filters them out. Google News TR is the most consistently working source. `test_rss.py` is the way to check current source health before debugging "why is sentiment score always 50" (low article count triggers a confidence pull toward neutral 50).
- KAP (Turkish public disclosure platform) is bot-protected — direct API/RSS access doesn't work. KAP disclosures are fetched via the third-party `borsapy` library instead (`_src_kap` in news_engine.py, graceful degrade if borsapy missing). borsapy's heavy deps (openai, pymupdf, onnxruntime) are lazy-loaded, so importing it costs no RAM.
- Comments in bist_analyzer.py are in Turkish; user-facing UI strings are Turkish (with an EN/TR/BOTH language toggle for AI-generated summaries).
