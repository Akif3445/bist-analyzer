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

python pipeline_backtest.py dondur 5    # (re)download the price panel for a window
python pipeline_backtest.py dondur-evren # re-fetch the UNIVERSE — new epoch, rare, invalidates old results
python pipeline_backtest.py run         # full pipeline sim vs XU100 / BIST-30 / random-6
python pipeline_backtest.py karsilastir # managed vs unmanaged — the presentation's core table
python pipeline_backtest.py sweep       # management-rule parameter sweep
python pipeline_backtest.py validate 8  # out-of-sample validation (8y, single 60/40 split)
python pipeline_backtest.py walkforward 8 5  # walk-forward: 5 expanding folds (stricter)
python pipeline_backtest.py pozisyon 8  # position-count sweep + luck-band measurement
```

`tools/pmdb_admin.py` exists so DB maintenance doesn't require handing Claude Code a blanket "run any Python" permission — the local `.claude/settings.local.json` allows only `python tools/pmdb_admin.py:*`, so the possible operations are bounded by this script's subcommands. Add new maintenance actions as subcommands here rather than widening the permission rule.

There is no test framework (no pytest) and no lint config — `test_*.py` files are standalone scripts you run and read the printed output of, not asserts.

`.env` needs `ANTHROPIC_API_KEY` for the AI Turkish summary feature (Claude Opus). Optional Telegram vars exist in some modules but core app runs fine without them (graceful degrade).

## Architecture

Two files carry the entire app:

- **bist_analyzer.py** (~8500 lines) — everything: data fetching, scoring engines, database, backtesting, time machine simulation, and every Streamlit page (`render_*_page` functions). Entry point is `run_app()` at the bottom, which picks BIST vs US market from a top bar, then renders one of ~15 pages selected from the sidebar per market.
- **news_engine.py** (~1600 lines) — standalone news sentiment engine, imported by bist_analyzer.py. Has no dependency back on bist_analyzer.py, so it can be read/edited in isolation.

`test_*.py` is gitignored — local-only, not part of the deployed/tracked app.

**`faz2_modules.py` was removed from the codebase entirely (2026-08-03).** It was gitignored, so `FAZ2_AVAILABLE` was `True` locally and `False` on Streamlit Cloud — the same commit behaved as two different applications. Locally it opened English-labelled tabs, sourced sentiment from `enhanced_news_analysis` instead of `news_engine`, and fetched analyst targets from `TargetPriceFetcher` instead of TradingView. A score seen locally was therefore not necessarily the score the cloud produced, which is indefensible for results presented as evidence. All four branch points, the import block, the ~68-line tab block and the `_target_consensus` plumbing (never rendered anywhere) are gone. Replacements already existed: news → `news_engine.analyze_news`, analyst target → `fetch_tv_analyst_targets`.

### Data flow for one stock score

`DataFetcher` (yfinance, `.IS` suffix auto-added for BIST) → `TechnicalEngine` + `ValuationEngine` + `analyze_news()` (news_engine.py) → `compute_bist_score()` combines them into a `BISTScore` → `RiskEngine` derives stop-loss/target levels → rendered by a `render_*_page` function.

`BISTScore.stock` field uses `field(default_factory=lambda: StockData(ticker=""))` — `StockData` requires `ticker` as a mandatory arg, don't remove the lambda.

### Scoring weights (WEIGHTS dict ~line 134, applied in compute_bist_score)

Technical 45% + News Sentiment 25% + Upside Potential 20% + Valuation 10% (rebalanced 2026-07 from 35/35/20/10 based on weight_calibration.py IC evidence — see below). When target price or fundamentals are missing, their weight redistributes to Technical/Sentiment rather than leaving a gap. Analyst targets auto-fetch from TradingView screener (`fetch_tv_analyst_targets`, single bulk query cached 4h, min 3 analysts, ~45 BIST names covered) — manual sidebar input still overrides. The faz2 `TargetPriceFetcher` fallback was **deleted** 2026-08-03 (see faz2 note under Architecture); when no target is found, `_auto_target` stays `None` and the Premium weight redistributes. Signal bands: 72+ Strong Buy, 57-71 Buy, 43-56 Neutral, 30-42 Sell, <30 Strong Sell.

### News sentiment (news_engine.py)

Three-layer classification, in order: (1) regex strong-pattern override for unambiguous events, (2) BERT (`savasy/bert-base-turkish-sentiment-cased`, optional — not in requirements.txt because Streamlit Cloud free tier is RAM-constrained; app auto-falls back), (3) weighted keyword lists (120+/98+ terms, severity-weighted).

**Language gate before layer 2 (2026-08-03).** The BERT model is Turkish-only. English US-market headlines were being tokenized into meaningless OOV pieces, yet the model still returned a label with >70% confidence often enough to clear `BERT_CONFIDENCE_THRESHOLD` — so it *pre-empted* layer 3 and the system asserted confident sentiment on text it could not read. `_is_english()` now routes English text to `_classify_english()` (layers 1 + 3 only). Turkish is detected **positively**: any Turkish-specific character settles it (real Turkish RSS headlines essentially always carry one), otherwise TR vs EN word evidence is counted, and with no evidence at all it defaults to Turkish.

The asymmetry runs the opposite way from what it first looks like, and getting it backwards cost a bug: a Turkish headline wrongly called English merely skips BERT — layers 1 and 3 contain Turkish patterns and keywords, so classification still mostly works — whereas English text wrongly called Turkish is exactly the failure being fixed. The first implementation used only function words and rated `"Apple raises guidance"` as Turkish, because terse financial headlines contain no function words at all. The word list therefore includes headline verbs (`raises`, `beats`, `cuts`, `misses`, `plunges`…) and finance nouns (`earnings`, `guidance`, `quarter`…), not just stopwords. Layers 1 and 3 already carried English patterns and keywords; the gap found while testing was that the keyword bank only had past-tense forms (`raised guidance`) while headlines are present-tense (`raises guidance`), so US news fell through to neutral — present-tense variants were added. FinBERT was deliberately **not** added: a second ~400 MB model contradicts the RAM constraint that keeps BERT itself optional. Sources are tier-weighted (`_get_tier`/`_get_weight`, tier 1 = 3x ... tier 3 = 1x) and each source has its own `_src_*` fetcher function using feedparser. Recency weighting, duplicate-source confidence boosting, and a speculation filter (`_is_speculation`) are applied after classification.

### Persistence

**`_PMDB` never silently falls back to local SQLite when Turso is configured (2026-08-03).** It used to: any network error dropped through to `sqlite3.connect(DB_PATH)`, which on Streamlit Cloud is `/tmp` and dies with the container. Some portfolio writes reached Turso and some went to a disk that was about to be erased — split-brain, with a NAV series that silently forks and no way to tell afterwards which row was right. Now: 3 attempts with exponential backoff (the session is discarded between attempts so a half-broken TLS connection isn't reused), then `TursoUnavailable` is raised. SQL errors still raise immediately — retrying an invalid statement is pointless. Note `is_cloud()` means "Turso is configured", not "running on Streamlit Cloud": with Turso set up locally, **offline local development now errors instead of falling back**. That is intended — a local fallback would fork the data just the same. `_safe_render` catches it and shows the error rather than a blank page.

`_PMDB._session()` is **thread-local**. It was a single shared `requests.Session`, which is not thread-safe; Streamlit runs every user session on its own ScriptRunner thread, so on the public demo two threads could share a urllib3 connection and one request's response could land in the other. The per-thread session keeps the TLS reuse win (~600ms → ~100ms per query).

Everything lives in one SQLite file (`bist_cache.db`, path resolves to `/tmp` on Streamlit Cloud, else app dir) via `AnalysisHistoryDB` (~line 348). It owns ~10 tables: analysis_history, alerts, accuracy_log, portfolio, backtest_trades/daily/summary, news_backtest_cache, accuracy_validation, plus `TimeMachineEngine`'s own run/pick/daily tables. There's no migration system — schema changes are additive `CREATE TABLE IF NOT EXISTS` blocks in `__init__`.

### Backtesting vs Time Machine — don't confuse these

- `BacktestEngine` (~line 3197): tests one strategy mode (swing/trend/universal/investor/buy&hold) against one ticker's history, point-in-time to avoid look-ahead bias.
- `TimeMachineEngine` (~line 2483): full portfolio simulation — picks stocks using the live scoring model at each historical point, then walks the portfolio forward. Six portfolio styles (Aggressive/Defensive/Momentum/Value/Stable/Custom).

### Portfolio Manager v2 (2026-07)

"Portfoy Yoneticisi" page (render_portfolio_manager_page): `InflationEngine` (ENAG E-TÜFE monthly table in `enag_monthly`, seeded from ENAG_MOM_DEFAULTS, user-extendable via UI; deflator() day-prorates months) converts nominal returns to real. PM tables go through `_PMDB` storage layer: if `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN` secrets are set, Turso HTTP API (v2/pipeline, pure requests — no libsql package, it won't build on Win/py3.14) gives persistence on Streamlit Cloud; otherwise falls back to local SQLite (and also falls back at runtime on any Turso error). `compute_market_regime()` (cached 30min) scores XU100 trend + BIST30 breadth + USDTRY into Boğa/Nötr/Ayı — Ayı regime blocks short-horizon proposals. `PortfolioManager` proposes portfolios from `PortfolioScanner.scan_all()` results by horizon (kisa/orta/uzun) × profile (Temkinli/Dengeli/Agresif) with sector caps and ATR-based stop/target; saved portfolios live in pm_portfolios/pm_positions/pm_nav, NAV updated once per session in run_app, performance reported as nominal + ENAG-real + XU100-relative. Rebalance suggestions flag stop breaches / score<43 (ÇIKAR), target hits (KÂR AL) and top candidates (EKLE). Min-position guard (2026-07-22): propose() refuses to build a portfolio when <3 stocks pass the filters (returns empty picks + explanation) — the W29 kisa/Temkinli shadow had saved a single stock at 100% weight (TUPRS), which is concentration, not caution; the shadow batch now skips such combos for the week. Comparison chart: **week multiselect + cohort-average mode (2026-08-11).** The old Son kohort/Tümü/Yok radio still left 11 lines for a single week, and by W33 the chart was unreadable. Now the user picks which weeks to plot (`pm_cmp_weeks`, defaults to the newest; empty = user portfolios only; Streamlit supplies a "Select all"), and a `Gösterim` radio collapses each week's system portfolios into one averaged line. **The average is taken AFTER rebasing each portfolio to 100** — otherwise a large-NAV portfolio would dominate — and **passive and managed cohorts are averaged SEPARATELY**, because the gap between those two lines is the live experiment itself (see Scientific integrity). Control groups (Rastgele / BIST30) stay as individual lines; they are the comparison. One week renders as 4 lines instead of 11.

### Active daily management — PortfolioDoctor (2026-07-24)

`PortfolioManager` *builds* portfolios; `PortfolioDoctor` *manages* them over time. User decisions: management is **fully automatic** (no per-trade confirmation), profile is **"Dengeli — kurumsal standart"**, and it must behave like a real PM (may buy, sell, or sit in cash).

**NAV accounting was rewritten to allow rebalancing.** The old formula assumed a fixed `entry_price` and no cash. NAV is now *chained*: `pm_portfolios.nav_base` + `rebalanced_at`, positions may sum to <100 with the remainder held as **cash** (0% return), and `NAV = nav_base × factor_since_last_rebalance` (`PortfolioManager._nav_factor`). At each rebalance the current NAV is frozen into `nav_base` and all surviving positions get their `entry_price` reset to that day's price. Verified numerically identical to the old formula when weights sum to 100 with no cash, so pre-existing portfolios are unaffected. `pm_positions` gained `peak_price` (trailing stop) and `opened_at` (min hold); new `pm_trades` table is the audit trail.

Daily order of operations in `daily_manage(market)`: **exits first and unbudgeted** (stop breach → trailing stop 15% off peak → score<43, the last respecting min hold) — cutting risk is never rate-limited, only opportunity-chasing is. Then **one consolidated weight decision per position** (partial 1/3 profit-take at target, then the 30% concentration cap; these were two separate actions at first and the second silently undid the first *and* corrupted the cash figure). Then **cash deployment / swaps**, gated by the drawdown brake (15% off portfolio peak) and the bear-regime gate, with swaps requiring a 12-point score edge and respecting the sector cap. Discipline layer: 5-day min hold, 10-day cooldown before rebuying, max 2 discretionary trades/day. **Transaction costs are charged to `nav_base`** (0.20% BIST / 0.10% US of traded weight) — a frictionless NAV would overstate active management, which is exactly what a reviewer would attack.

**Scientific integrity — do not collapse these two tracks.** The existing passive shadow cohorts are the evidence that *selection* beats luck/passive, and their controls are buy-and-hold. Turning management on for them mid-flight would make W29+ uninterpretable. So `ensure_shadow_batch` now saves a **parallel "Yönetilen" cohort** from the identical picks with `managed=1`, while the passive "Gölge" cohort is untouched. Same day, same prices → the difference *is* the management contribution. User portfolios (`kind="kullanici"`) default to `managed=1`.

This parallel-cohort design became considerably more important on 2026-08-03: the corrected backtest says the management layer **costs** real return and buys drawdown protection (see the managed-vs-unmanaged table below). The live Gölge/Yönetilen pair is the independent read on that question, on data no parameter was fitted to. Let it run.

Robot runs the management pass for both markets. `.github/workflows/daily-robot.yml` runs a cron per market — 15:40 UTC for BIST, 21:30 UTC for US (after its close), because the 15:40 run is mid-session for US and would decide on half-day prices. See "Robot market isolation" under US parity for the full wiring and the trap that came with it.

### Rule audit + editor-approved parameters (2026-07-28)

`PortfolioAudit` runs in the robot every day, measures whether the management rules are actually working, writes to `pm_audit`, and **changes nothing on its own**. Metrics: **stop accuracy** (did a stopped-out stock keep falling? if it rebounded the stop was too tight), **swap accuracy** (did the replacement beat what was sold?), **cost burden** (commission paid ÷ gross NAV movement), **managed−passive spread** (the parallel cohorts — the actual question), and average cash. Exit/swap events only count once ≥5 days old.

**Why this is an audit and not an optimizer.** The user asked for a mechanism that continuously steers toward what works. That exact thing is dangerous: with 10 rule parameters, 6 years of history and only *weeks* of live data, a continuous self-tuner fits noise, chases whatever regime just ended, and destroys the ability to say "these rules produced this result" — which is the presentation's whole claim. So the design splits: **audit is continuous and automatic; parameter changes are proposals, gated, and rare.** Every finding is hard-gated on a minimum sample (`PortfolioAudit.MIN_ORNEK`, e.g. 8 stop events) — below that the metric is still recorded but produces no proposal.

**Parameter overrides.** `PortfolioDoctor.kural(key)` reads `pm_params` first and falls back to the `KURAL` dict in code, so code values remain the documented defaults and any change is reversible. Writing an override goes through `kural_ayarla`, and the UI button is behind `_writes_locked()` / `_guard_write()` — **only the unlocked editor can change a parameter** (user decision 2026-07-28). Each change stores date + reason, so "when did I change what, on what evidence" is answerable.

UI cost is deliberately near zero: one line in the Portfolio Manager (`🩺 Yönetim denetimi — sağlıklı`), which expands only when there is a finding; the parameter table appears only if overrides exist.

**One audit rule pointed the wrong way and was reversed (2026-08-03).** When managed cohorts lagged the passive ones, the `yonetilen_fark` finding proposed *raising* `takas_farki` — the intuitive response ("management isn't paying, so trade less"). Measurement says the opposite: a high swap threshold is what stops the portfolio from adopting each month's fresh picks, so it holds stale names and falls behind. Raising it would have deepened exactly the problem the audit was detecting. It now proposes lowering, with the reasoning in the message. No harm was done because proposals are never auto-applied — which is precisely the argument for that gate. Worth remembering as a case where a plausible-sounding rule was backwards, and only the parameter sweep exposed it.

### Management-rule sweep (2026-07-30) — `pipeline_backtest.py sweep`

`_walk_managed` replays the daily management layer (trailing stop, score exit, position cap, drawdown brake, swap threshold) over the same point-in-time pipeline, and `sweep` varies one parameter at a time. Acceptance is deliberately strict: a change is only proposed if the return difference exceeds 5% **and** the better value wins in nearly every calendar year. Proposals are written to `pm_param_oneri` and **never applied automatically** — the UI surfaces them next to the live-audit findings behind the same editor gate.

**This sweep immediately paid for itself by finding a serious bug.** The drawdown brake measured drawdown from the **all-time** peak (`cummax`), which made it a permanent trap: once 15% below the peak, buying stopped, stops kept selling into cash, cash returns 0% so NAV could never recover, and the all-time peak never declined — so the portfolio stayed braked forever. Five-year backtest return with the brake on was **+102%** vs **+3311%** with it off. Fixed in both the live engine and the backtest by measuring drawdown from a **90-day rolling peak**, which releases naturally.

**Sharpe is computed on ENAG-deflated returns (corrected 2026-07-31).** Using nominal returns with rf=0 inflated it roughly **fivefold** — measured 3.46 nominal vs 0.69 real — because inflation lifts the mean without lifting daily volatility. A Sharpe above 3 would have made this one of the best funds in history and would not have survived one question. Real risk-free is assumed 0, which is *conservative*: Turkish policy rates ran below inflation, so the true real risk-free was negative.

### The managed-vs-unmanaged comparison was WRONG and its conclusion is now REVERSED (2026-08-03)

The old claim here was "active management clearly adds value: +4083% vs +3642% unmanaged, Sharpe 0.70 vs 0.60". **That comparison was invalid** and the invalidity flipped the sign of the answer.

Cause: `_walk_managed` let position weights drift with price between rebalances, but `_walk` held them **fixed**. A fixed-weight portfolio is one that returns to target weights every single day at zero cost — not a portfolio anyone can hold. In a trending market that means trimming the winner daily, which drags the return down hard. So the unmanaged *reference* was artificially crippled, and management only looked good by comparison. Fixed by giving `_walk` the same price-drift mechanics (`_walk` docstring carries the reasoning).

**Clean re-measurement** — frozen panel, 468 stocks, 2021-08-03 → 2026-08-03, fingerprint `d06dee9a3d6a`, run twice with byte-identical output (`python pipeline_backtest.py karsilastir`):

| | Nominal | **Sharpe** | MaxDD |
|---|---|---|---|
| Unmanaged (monthly rebal.) | +9760% | 0.92 | −49.1% |
| Managed, `takas_farki=12` (old default) | +4860% | 0.71 | −25.8% |
| **Managed, `takas_farki=6` (current)** | **+13552%** | **1.48** | **−19.7%** |
| XU100 | +858% | −0.54 | −22.9% |

**With the swap threshold at its corrected value the management layer wins on all three axes** — more return, roughly half the volatility-adjusted risk, and drawdown cut from −49.1% to −19.7%. The middle row is what the *previous* default produced and is kept only to show how much a single mis-set parameter cost: at 12 the layer could not adopt each month's fresh picks and carried stale names (see the `takas_farki` table below).

### Same comparison over 8 years, including the 2018 and 2020 crises

The five-year window is entirely a BIST bull market — exactly where downside insurance can only look like cost. Re-run on 2018-08-03 → 2026-08-03 (468 stocks, fingerprint `47a4ec949c92`, `karsilastir 8`), a window containing the 2018 TL crisis and the 2020 Covid crash:

| | Nominal | ENAG-real | **Real CAGR** | **MaxDD** | Sharpe |
|---|---|---|---|---|---|
| Unmanaged | +14974% | +77.1% | **+7.4%** | **−69.9%** | 0.39 |
| Managed | +10756% | +27.5% | +3.1% | **−25.8%** | 0.25 |
| XU100 | +1307% | −83.5% | −20.1% | −31.8% | −0.72 |

**With `takas_farki=6` the managed track wins here too**, and by more than it does over five years: real CAGR **+15.6% vs +7.4%** unmanaged, MaxDD **−25.2% vs −69.9%**, Sharpe **0.71 vs 0.39**. The `takas_farki=12` row above is the old default and is kept for the same reason as in the 5-year table.

So the "management trades return for risk" framing that stood for a few hours on 2026-08-03 is **withdrawn on both windows**. It was never a trade-off; it was one mis-set parameter.

### The parameter that did all of this

`MGMT_VARSAYILAN["takas_farki"]` was 12, and `validate` rejected that value on **both** windows in favour of 0. Measured on the 8-year panel, varying only `takas_farki`:

| `takas_farki` | Real CAGR | MaxDD | Sharpe | Cost | Out-of-sample test half |
|---|---|---|---|---|---|
| 0 | **+15.9%** | −25.2% | **0.71** | 23.5% | +4023% |
| 3 | +14.3% | −25.2% | 0.66 | 22.9% | +3508% |
| **6** | **+15.6%** | −25.2% | **0.71** | 22.3% | +4011% |
| 9 | +4.9% | −25.8% | 0.32 | 21.7% | +1969% |
| **12 (old default, superseded)** | **+3.1%** | −25.8% | **0.25** | 20.1% | +1477% |
| 20 | +3.3% | −25.5% | 0.26 | 18.8% | +1950% |

**There is a cliff between 6 and 9, and a flat plateau below it** — this is a robust result, not a tuned optimum: any value ≤6 gives ~+15% real CAGR and Sharpe ~0.7, any value ≥9 gives ~+3-5% and Sharpe ~0.25.

**Why the parameter matters this much.** `takas_farki` is the score edge a candidate must have over the weakest current holding before the layer will buy it. At 12 most of each month's fresh picks were refused, so the managed portfolio kept holding stale names while the unmanaged track adopted the new target list in full. "Managed" was therefore not the model plus risk control — it was **a stale version of the model with stops bolted on**, and that, not risk management, is what was costing the return.

**With validated parameters (`takas_farki` 0, `max_pozisyon` 40) the management layer dominates on every axis** over the 8-year window: real CAGR **+18.6% vs +7.4%** unmanaged, MaxDD **−26.8% vs −69.9%**, Sharpe **0.79 vs 0.39**. Nearly all of the gain comes from `takas_farki` alone (+15.9% by itself); `max_pozisyon` contributes little and remains the marginal proposal.

Caveat on precision: the full-window figures are partly in-sample, since the parameter was chosen on the first 60% of the same window. The clean number is the out-of-sample column above (test half 2023-05 → 2026-08, which took no part in the choice), and it tells the same story with the same cliff.

**Corrected reading, superseding everything above it:** the management layer is *not* a return-for-risk trade. Configured as its own validation says it should be, it improves return, drawdown and Sharpe simultaneously on both windows. The "management buys drawdown protection and pays for it out of real return" conclusion was an artifact of measuring it throttled. What survives from that discussion is the honest note that Sharpe under-prices drawdown — but it is no longer needed as a defence, because Sharpe now favours the managed track too.

**Methodological lesson worth telling in the presentation.** Three separate conclusions in this project turned out to be artifacts of a measurement error rather than facts about markets: management appearing to destroy value (`_walk` fixed weights), the drawdown brake appearing catastrophic (all-time-peak `cummax`), and the trailing stop appearing to be pure cost (measured while `takas_farki` blocked replacements). In each case the reported number was confidently wrong and internally consistent. The defence that actually worked was not intuition — every one of these was intuitive — but re-measurement under a changed assumption.

### Sweep re-run 2026-08-03 — every parameter now says DON'T CHANGE, and the old proposals were interaction artifacts

Frozen 5-year panel, `takas_farki=6` as the new default. Reference: unmanaged +9760%, Sharpe 0.92, MaxDD −49.1%.

| Parameter | Best value | Default | Verdict |
|---|---|---|---|
| `trailing_pct` | 20 (+2286% over default) | 15 | no change — only 3/6 years |
| `dusus_freni` | 15 | 15 | already optimal |
| `max_pozisyon` | 30 | 30 | already optimal |
| `takas_farki` | 6 | 6 | already optimal |

**The two proposals that were pending approval have evaporated, and that is the important finding.** They were:

- `max_pozisyon` 30→40 — now 40 scores *worse* than 30 (+12926% vs +13552%). Its earlier advantage existed only while `takas_farki=12` starved the portfolio of fresh names; a wider position cap partly compensated for a portfolio that could not rotate. Fix the rotation and the compensation becomes harmful.
- `trailing_pct` **disabled** used to score best (+7344%). Now disabling it (`=100`) gives +10942% against +13552% for the default 15 — **the trailing stop earns its keep once the portfolio actually rotates.** Previously it was cutting positions the layer then could not replace.

This is bias #6 on the list below (one-factor-at-a-time) biting hard: with `takas_farki` mis-set, single-parameter sweeps were measuring compensations for that error rather than the parameters themselves. Any future single-parameter conclusion should be re-checked whenever a different parameter moves.

Method note (unchanged): acceptance requires the return difference to exceed 5% **and** the better value to win in nearly every calendar year. Proposals go to `pm_param_oneri` and are **never applied automatically**. The discipline parameters (`min_tutma_gun`, `bekleme_gun`, `gunluk_islem`) are deliberately **not** swept — they are anti-churn constraints, not free variables.

### Backtest results in REAL terms, and the bias list (2026-07-30)

**Never quote the nominal number alone.** Over the 5-year window ENAG cumulative inflation was **~25×**, so nominal figures are close to meaningless. Current verified figures are in the managed-vs-unmanaged table above (unmanaged +31.7% real CAGR, managed +14.8%, XU100 −17.4%).

The defensible framing is "real CAGR while the index lost 61% in real terms", not "~100× in five years". **Do not repeat the old "management contributes ~2.5 points of real CAGR" line — it was an artifact of the broken baseline and the sign is now negative.**

### Control groups — this is the result that actually supports the thesis (`run`, 2026-08-03)

Same frozen panel and window, `python pipeline_backtest.py run`:

| | Nominal | ENAG-real | **Real CAGR** | MaxDD | Sharpe |
|---|---|---|---|---|---|
| **System (pipeline)** | +9760% | +295.5% | **+31.7%** | −49.1% | **0.92** |
| Equal-weight BIST-30 | +1042% | −54.2% | −14.5% | −28.2% | −0.37 |
| Random-6 (50 seeds, mean) | +884% | −60.5% | −17.0% | −41.5% | −0.29 |
| XU100 | +858% | −61.6% | −17.4% | −22.9% | −0.54 |

**The system's final NAV (98.60) sits above the 95th percentile of the random-6 distribution (5–95% band [4.73, 22.29]).** It is not a lucky draw: random 6-stock portfolios from the same universe, rebalanced monthly with the same costs, land nowhere near it. This — not the management layer — is the evidence that the *selection* model works, and it is the strongest thing in the project.

Honest caveat to state alongside it: the system also takes **more drawdown than the index** (−49.1% vs −22.9%). It wins on return, not on smoothness. That is precisely the gap the management layer was built to close, and per the table above it closes it (−25.8%) by giving back real return.

**Known biases, ranked.** Two are serious:

1. **The backtest tests a DIFFERENT system than the live one.** It uses the technical score only (`BacktestEngine._vectorized_scores`); the live score is 45% technical + 25% news + 20% analyst upside + 10% valuation. News and analyst data cannot be reconstructed historically, so these results describe the **technical core**, not the deployed system. State this explicitly.
2. **Parameter selection was in-sample** — `sweep` finds the best value on the same data it measures. Addressed by `validate` below.
3. **Survivorship / universe selection** — the universe is *today's* liquid list, so companies that delisted or shrank are absent (optimistic). Partly mitigated by applying each date's own volume filter.
4. **Bull-market sample** — the 5-year window is entirely a BIST upswing, where insurance (stops, brakes) can only look like cost. Addressed by using an 8-year window in `validate`, which contains the 2018 TL crisis (−28% drawdown) and the 2020 Covid crash (−32%).
5. **No slippage model** — cost is a flat percentage; in reality size moves price in small caps, which the universe contains.
6. **One-factor-at-a-time** — parameter interactions unexplored.

### Reproducibility anchor (2026-08-03) — `dondur`

Backtest results used to change between identical runs. Measured: two consecutive runs of the same command produced panels of 468 and 464 stocks (yfinance silently drops symbols) and 5-year unmanaged return moved **+9738% → +7810%**. With a 6-position concentrated portfolio, two or three missing names swing the compounded result ~20%. A number that moves when you re-run it cannot be defended in an academic setting.

So the pipeline is now pinned:

- **`backtest_evren.json`** — the universe, **tracked in git**. This is the anchor. Changing it makes past results incomparable, so treat it as a versioned artifact, not a cache.
- **`backtest_panel_<yil>y.pkl`** — the downloaded price panel, gitignored (~26 MB) but rebuildable from the frozen universe. Carries a SHA fingerprint printed on every run (currently `d06dee9a3d6a`).
- **Failed symbols are retried** in small batches (yfinance returns empty for random symbols in bulk queries but serves them individually). 32 of 500 remain missing — those are genuinely delisted (KOZAL, IPEKE, EMLAK…), which is the survivorship bias already on the list, not a data bug.
- **`_precompute` builds its dict sorted**, and the random control group draws from `sorted(feats)`. Dict insertion order was download order, so it silently changed both tie-breaks in `_pick_at` and the random-6 lottery.

**`dondur` and `dondur-evren` are deliberately separate commands.** `dondur [yil]` re-downloads only the price panel; `dondur-evren` re-fetches the universe from TradingView and starts a new measurement epoch, after which old results are not comparable and **every** panel cache must be rebuilt. The first version conflated them, and running `dondur 8` silently overwrote the universe the 5-year results were built on — leaving the 5y and 8y panels sitting on two different universes with only the newer one recorded. Liquidity ranking shifts with intraday volume every single day, so re-freezing the universe must be rare and intentional, never a side effect of adding a window.

Verified: `karsilastir` run three times back-to-back gives byte-identical output.

One residual: the window end is "today", so the 5-year span shifts daily. Always record the window with the number (current results are 2021-08-03 → 2026-08-03).

### Out-of-sample validation — `pipeline_backtest.py validate` (2026-07-30)

The answer to "did the system find something real or fit noise?". The window is split: the best parameter is chosen on the **first 60%** (train, 2018-07→2023-05) and then tested on the **last 40%** (test, 2023-05→2026-07), which never participates in the choice.

**Two of four sweep proposals were exposed as noise-fitting.** Re-run 2026-08-03 on the frozen panel (468 stocks, fingerprint `47a4ec949c92`) — **every verdict replicated**, which is itself a good sign for the method:

| Parameter | Train picks | Test: default | Test: picked | Verdict |
|---|---|---|---|---|
| `trailing_pct` 15→10 | 10 | +1476.6% | **+1006.2%** | rejected |
| `dusus_freni` 15→10 | 10 | +1476.6% | **+605.1%** | rejected |
| `max_pozisyon` 30→40 | 40 | +1476.6% | +1552.8% | validated (**marginal**) |
| `takas_farki` 12→0 | 0 | +1476.6% | **+4023.1%** | validated (decisive) |

**The two survivors are not equally strong and the gate does not distinguish them.** `validate`'s pass condition is bare `t_s > t_v` ([pipeline_backtest.py:649](pipeline_backtest.py:649)) with no minimum margin, while `sweep` — which only *proposes* — demands a >5% difference **and** consistency across nearly every calendar year. So the final arbiter is the more permissive of the two. On these numbers `takas_farki` improves the test window by 172% while `max_pozisyon` improves it by 5.2%, and both come back labelled "DOĞRULANDI". Treat `max_pozisyon` 30→40 as unproven until the gate carries a margin; a reviewer will find this in a minute.

`validate` overwrites `pm_param_oneri` with out-of-sample evidence and **deletes** proposals that failed, so the editor only ever sees survivors. Note the direction on `takas_farki` is robust (lower is better, confirmed on both windows) but the exact value is not — the 5-year sweep said 6, the 8-year validation says 0. Prefer the conservative end.

**Remaining honesty gap:** the test period (2023-05→2026-08) is itself mostly bullish, so parameters were validated on a bull holdout. The crisis periods sit in the *training* half. A stricter design would reverse the split or use walk-forward with multiple folds — still the top open methodology item.

### Signal Tracker

Every signal the app generates gets logged and its forward return is checked at 1/3/7/14/30 days (`check_pending_signals`, called once per session in `run_app()`). This is what "Sinyal Takip" pages display — it's live production tracking, not backtesting.

### Weight calibration (weight_calibration.py)

Standalone research tool that builds a point-in-time panel (50 BIST stocks × 6y × weekly, `calibration_tech.csv`) plus a GDELT daily news-tone panel (`calibration_sentiment.csv`) and computes cross-sectional ICs (Grinold-Kahn) to ground the compute_bist_score weights in data. Key 2026-07 findings, now applied to the code: the original contrarian technical score (high points for oversold RSI/BB/52w) had ~zero IC because BIST is in a momentum regime (high RSI/52w-position predicted HIGHER returns, t>3, stable across sub-periods; strongest signal = week52_position). Both scoring paths (`TechnicalEngine._compute_score` for live, `BacktestEngine._vectorized_scores` for backtest/time-machine) now take a `style` param — "momentum" (all BIST, selected via `_tech_style_for`) or "dengeli" (original contrarian, still used for US). A segment-based hybrid (contrarian for BIST-30) was tried and REJECTED: mixing scales corrupts cross-sectional ranking (IC drops from +0.045 to +0.012). A/B backtest (8 stocks, 3y, universal mode): avg total return +1.9% → +13.4%, but not uniform (THYAO notably worse). GDELT news tone was weak everywhere (t<1), so sentiment's 35% weight is likely too high — weight rebalance pending.

**Calibration universes are frozen too** (2026-08-03): `_us_tickers` writes its resolved slice to `calib_evren_us.json` (tracked in git, keyed `offset_n`) and reads it back on later runs. The US universe is a live TradingView query that changes daily, so without this "re-run the same analysis" is not a well-defined request and IC/t-statistics drift between runs. Same principle as `backtest_evren.json`. Full run: `build-tech` (~3 min), `build-sentiment` (~45 min, 5.5s/req rate limit, ~15/55 tickers succeed per pass), `analyze`. `analyze-stops` (2026-07, 2730 top-quintile entries) showed tight stops rank dead last (old kisa 1.5×ATR was 24/24) and hard profit-targets cut momentum winners badly; stops were widened to 2.5-3.5×ATR and targets became informational levels (alerts suggest, never auto-exit). Caveat: sample is a bull market — wide stops kept as disaster insurance alongside regime gate + drawdown brake.

## US / NASDAQ parity (started 2026-07-23)

Goal: bring the US side to the same shape as BIST. User decisions: **nominal USD only** (no inflation/real-return layer on the US side — ENAG is Turkish, and a US-CPI layer was declined), **dynamic S&P-500 + NASDAQ-100 universe**, **targeted at the October presentation**.

Shared infrastructure is market-parameterized rather than duplicated: `get_scan_universe(market)`, `PortfolioScanner.scan_all(market=)` (separate cache table `portfolio_scan_us`), `compute_market_regime(market)`, and `PortfolioManager.propose/ensure_shadow_batch/ensure_control_batch/update_navs(market=)`.

**`get_scan_universe` now sorts by dollar/TL volume descending (2026-08-03).** It previously returned TradingView's raw row order. The order is not cosmetic: `weight_calibration._us_tickers` slices it **positionally** (`[300:420]` for the mid-cap panel), so before this fix "liquidity ranks 300–420" was not what was actually being measured — see the US calibration section for the consequences. `pm_portfolios` has a `market` column (idempotent migration; old rows read as BIST via `COALESCE(market,'BIST')`). Every call site defaults to `market="BIST"`, so BIST behaviour is unchanged.

US specifics: liquidity threshold `UNIVERSE_MIN_USD_VOLUME` = $200M/day → ~596 names (measured: 50M→1395, 100M→975, 150M→745, 200M→603, 300M→443; $200M matches the S&P-500 + NASDAQ-100 union of ~600). Regime uses S&P 500 trend + breadth + **VIX as a level** (<22 calm) instead of USDTRY — a dollar portfolio has no currency risk, it has a volatility regime. Benchmark is `_default_index(market)` = `^GSPC`; the `pm_nav.xu100_close` column keeps its name for compatibility but holds the portfolio's own index. `performance()` returns `reel=None` for US. Passive control group is the 100 most liquid US names (NASDAQ-100 proxy) instead of BIST-30.

UI: the US menu mirrors BIST exactly (US Kokpit · US Portfoy Yoneticisi · US Listesi · US Analiz · Portfolyum · US Backtest · US Sinyal Takip). `render_kokpit_page` and `render_portfolio_manager_page` take a `market` param instead of being duplicated — the US menu calls the same functions with `market="US"`. Helpers: `_bench_ad(market)` for the benchmark label, `_reel_var(market)` for whether a real-return column exists. The ENAG tab is only built for BIST, and real-return metrics render "—" when `perf['reel']` is None. **Market is deep-linkable via `?piyasa=US`** (same pattern as `?tema=`) and persists across reruns — this is also the only way to drive the US side from browser automation, since the market selector is an iframe component whose clicks don't propagate.

**Robot market isolation (2026-08-03).** Each cron run now handles only the market that has actually closed: `40 15 * * 1-5` → `ROBOT_ONLY_MARKET=BIST`, `30 21 * * 1-5` → `US`. Previously the early run had no value set, so it processed **both** — meaning US portfolios got decisions twice a day, and the early one was priced mid-session. `workflow_dispatch` leaves it empty on purpose (manual run = full pass).

> **Trap that came with this change:** the weekly shadow/control loop was written `for _mkt in (_piyasalar if not _only else ())` — under any restriction it iterated over **nothing**. That was harmless only while the early run was unrestricted. Restricting both runs without fixing it would have stopped every shadow cohort from being created, silently, with no error. Now `for _mkt in _piyasalar`; idempotency was already guaranteed by the per-week per-market `eksik_g`/`eksik_k` check. `update_navs` is likewise called per market now, so US NAV is no longer written from mid-session prices.

Regime logging is per market too, and `pm_regime_history` was migrated from `PRIMARY KEY (date)` to `(date, market)` — with one row per day the second market silently overwrote the first. Migration is idempotent, runs from `_rejim_tablosu()` on the robot's next pass, copies existing rows as `market='BIST'` (they were all written with the default), and adds a `vix` column (US stress is VIX, BIST's is FX — folding them into one column would corrupt any later analysis).

### Incident 2026-07-31 → 08-05: the robot ran green every day and did nothing

Turso started returning **400 Bad Request** to the Actions runner. The then-current `_PMDB` silently fell back to `sqlite3.connect(DB_PATH)` — on a GitHub runner that is a scratch disk destroyed with the container. Reads fell back too, so the robot saw an empty database, reported `NAV güncellendi (0 aktif portföy)`, made no decisions, and wrote its work into a directory that no longer exists. Every run showed a green checkmark, because each step catches its own exception and `main()` returned 0 unconditionally.

Cost: five weekdays with no stop-loss execution, no score exits, no swaps, and — worse for the October evidence — the "Yönetilen" shadow cohorts ran *effectively passive* for a week, so that stretch of the managed-vs-passive comparison is contaminated and cannot be recovered.

The URL was identical on both sides and the same code worked locally, so the cause is the **`TURSO_AUTH_TOKEN` repo secret**, not the code.

Three defences added, because any one of them would have caught it on day one:

1. **`_PMDB` no longer falls back** when Turso is configured (see Persistence) — it retries, then raises `TursoUnavailable`.
2. **The robot probes Turso with a real `SELECT 1` before doing any work** and exits 1 if it fails. "Configured" is not "reachable", and that distinction is the whole incident.
3. **The exit code now tells the truth.** Step failures are collected in `hatalar` and produce a non-zero exit. ENAG scraping and the monthly calibration are deliberately excluded — both are known-flaky external sources (the ENAG site's TLS is broken) and would produce permanent red, which blunts the signal.

Lesson for the presentation's reliability story: a green build proved nothing here because success was never actually asserted. The failure was silent for a week in a system that had monitoring, an audit layer, and a daily cron — all of which were themselves reporting success.

**Robot:** `daily_robot.py` loops both markets for the weekly shadow/control cohorts (each market wrapped separately so one failing doesn't stop the other; freshness checked per market). First US cohort landed 2026-W30: 9 shadow + 2 control (random-draw + NDX100 passive with 99 names). When the US shadow set comes back empty the robot prints why — in Ayı that is the calibration brake firing, not a failure. `check_risk_alerts()` and `update_navs()` also run per market; both previously built symbols as `t + ".IS"` for every portfolio, so **US positions would silently never price and never raise a stop/target alert** — fixed 2026-07-23.

### US calibration finding (2026-07-23) — the edge is in mid-caps, not mega-caps

Two point-in-time panels, same Grinold-Kahn method as BIST (120 stocks × 6y × weekly, ~28k observations each), via `weight_calibration.py build-tech-us` / `build-tech-us-mid` + `analyze-us` / `analyze-us-mid`:

- **Mega-cap (120 most liquid):** momentum IC +0.0065 (t=0.53) — insignificant, sign flips across sub-periods. Contrarian IC −0.0136 (t=−1.31), consistently on the *wrong* side. No technical edge in the world's most efficient segment.

> **Mega-cap re-measured 2026-08-03** after the universe sort (117 stocks, 25,794 obs). The momentum half replicated exactly — IC +0.0062 (t=0.47) at 30d, sign still flipping across sub-periods (−0.0223 / +0.0367). **The contrarian claim did not.** It is now +0.0085 (t=0.69) at 30d and **+0.0622 (t=5.33) at 90d** — the largest single t-statistic anywhere in the mega-cap table, where the old note said it was "consistently on the wrong side."
>
> **Do not act on that t-statistic.** Contrarian's 30-day sub-period signs also flip (+0.0321 / −0.0167), which is the same episodic pattern already documented for 52-week position: a large full-sample number carried by one sub-period, not a durable effect. Switching US to contrarian on the strength of one t=5.33 would repeat precisely the mistake this file keeps recording. What the re-measurement actually establishes is weaker and more useful: **neither style has a stable edge in US mega-caps**, so the earlier justification for `_tech_style_for` returning `"momentum"` ("contrarian is reliably wrong") is withdrawn, and the real justification is the regime-conditional result below plus the mid-cap contrarian sign. Both styles' unconditional verdict on the corrected panel is the tool's own: *do not use the US score alone as a ranking.*
- **Mid-cap (liquidity ranks 300–420):** momentum IC +0.0212 (**t=2.06**), **stable across sub-periods**. 52-week position t=2.14, RSI t=2.27 also signal.

> ### ⚠️ The mid-cap result above did NOT replicate (re-measured 2026-08-03)
>
> It was computed before `get_scan_universe` sorted by volume, so `[300:420]` sliced TradingView's raw row order — the panel was **not** the liquidity-rank band it claimed to be. Re-run on the correctly ranked slice (`calib_evren_us.json` key `300_120`, 118 stocks, 26,774 obs, 2021-08 → 2026-03):
>
> | Signal | Horizon | IC | t | vs. old claim |
> |---|---|---|---|---|
> | Momentum | 30d | +0.0149 | +1.33 | was +0.0212, t=2.06 → **no longer significant** |
> | Momentum | 90d | +0.0246 | +2.55 | significant only at the long horizon |
> | 52-week position | 60d | +0.0323 | +2.94 | holds |
> | 52-week position | 90d | +0.0476 | **+5.26** | **stronger than before** |
> | RSI | 30d | +0.0107 | +0.99 | was t=2.27 → **gone** |
>
> And the claim that mattered most is falsified: momentum's 30-day IC **flips sign across sub-periods** (first half −0.0080, second half +0.0393). "Stable across sub-periods" was an artifact of the wrong panel. The tool's own verdict on the corrected data is *"both styles insignificant; do not use the US score alone as a ranking."*
>
> What survives is **52-week position at long horizons** — the same indicator that dominates BIST, and consistent with the "episodic, not persistent" reading recorded further down.
>
> **`_tech_style_for` still returns `"momentum"` for US, but for a weaker reason than originally written.** Not "momentum has a measurable mid-cap edge" — it doesn't, unconditionally. Rather: contrarian scores −0.0097 (t=−1.05) and is *consistently* on the wrong side across both sub-periods, so momentum remains the better of two weak options, and the regime gate below is what actually makes it usable.

So `_tech_style_for` now returns `"momentum"` for **both** markets. Effect measured on a live rescan: score↔1M-momentum correlation went **−0.354 → +0.323**, and the top-10 picks went from 14-of-20 down-on-the-month to **0-of-10**. Before this, the US top pick was a stock down 43% in a month.

**Caveat to state plainly in the presentation:** US IC (+0.021) is far below BIST's (+0.045, t=2.8). The US ranking is a *weak* signal and must not carry a strong claim on its own.

**Acted on (2026-07-23):** the universe threshold was lowered $200M → $50M (596 → 1406 names) to *add* the mid-cap band where the edge actually lives, without removing any mega-caps. Horizons were extended to 60/90 days because mid-cap IC was still rising at 30d. (The threshold change still stands after the 2026-08-03 re-measurement — a wider universe is harmless — but its stated *justification* was the mid-cap edge, which did not replicate. The surviving justification is the long-horizon 52-week signal, which also favours a wider universe.)

**Longer-horizon result — read this before quoting any t-stat.** The best US signal found is **52-week position at 90 days: IC +0.0323, t=3.95** — the same indicator that dominates BIST. But the year-by-year breakdown shows it is **episodic, not persistent**: 2023 +0.101 (t=6.97) and 2026 +0.117 (t=3.67) carry almost all of it, while 2021 (−0.017), 2024 (−0.010) and 2025 (+0.016) are flat or negative. The half-sample "stable" label is technically true and substantively misleading. This is deliberately **not** wired into scoring. The honest reading: buying names near their 52-week highs paid off spectacularly in strong momentum years and did nothing otherwise — a regime-dependent effect, not a durable edge. A natural follow-up is to gate it on `compute_market_regime` — **this was then tested, see below.**

### Regime-conditional result (2026-07-23) — the US signal is conditional, not weak

`weight_calibration.py analyze-us-regime` rebuilds the historical regime point-in-time with the same 0-7 scheme as the live engine (SPX vs SMA200, golden cross, 1m momentum, breadth taken from the panel's own `above_sma200` column so there is no look-ahead, VIX), then splits IC by regime. On 26.5k observations at the 90-day horizon:

**Re-measured 2026-08-03 on the corrected liquidity slice — this one REPLICATED and got stronger** (26,774 obs; regime split Boğa 72% / Ayı 20% / Nötr 8%):

| Signal | Boğa | Nötr | Ayı |
|---|---|---|---|
| 52-week position | **+0.0743 (t=8.60)** | +0.032 (t=0.60) | −0.044 (t=−2.09) |
| Momentum score | **+0.0558 (t=5.41)** | +0.022 (t=0.52) | **−0.0881 (t=−4.72)** |

(Previous figures on the mis-sliced panel: 52w Boğa +0.0528 t=6.02, momentum Boğa +0.0448 t=4.39, momentum Ayı −0.063 t=−3.24. Same signs, same story, larger magnitudes.)

Two conclusions. **In a bull regime US IC exceeds BIST's headline +0.045** — the earlier "US is half as strong" reading was an artifact of averaging across regimes. And **in a bear regime the momentum score is significantly inverted**: following it loses money, now at t=−4.72.

This is the load-bearing US result. Note the contrast with the unconditional mid-cap finding above, which did *not* survive re-measurement: the US edge is real but exists **only conditionally**, which is exactly why the regime gate is not optional decoration.

Applied: `propose()` refuses *every* horizon for US in Ayı (previously only `kisa` was blocked) and warns on Nötr. BIST is unchanged — there is no equivalent evidence for it.

**Caveat for the presentation:** Boğa is 72% of the sample and the regime split largely coincides with the strong years (2023, 2026). What makes this a usable filter rather than hindsight is that *regime is observable at decision time, a year label is not*. Still one market, one 6-year window — present it as a conditional finding, not a law.

## Roadmap — October professor presentation (agreed 2026-07-16)

Goal: system that credibly beats index and targets ENAG-real returns; academic-grade methodology matters as much as returns. Live shadow tracking needs 4-8 weeks — meanwhile, in priority order:

- **A. Full-pipeline historical simulation** (the "what if results disappoint" insurance): run TODAY's portfolio construction (dynamic universe → momentum score → sector cap → correlation filter → ATR weights → monthly rebalance) point-in-time over 5-6 years → monthly NAV series, reported nominal + ENAG-real + vs XU100. This becomes the presentation's backbone chart. NOT the old TimeMachine (that uses different logic).
- **B. Scientific control groups in the weekly shadow batch**: random-pick portfolio (6 stocks by lottery from universe) + equal-weight BIST-30 — "does the system beat luck and passive?"
- **C. Institutional metrics on NAV curves**: Sharpe, Sortino, max drawdown, Information Ratio, win/loss ratio; add transaction-cost/slippage assumption to NAV (currently frictionless).
- **D. Monthly recalibration loop**: rerun weight_calibration 2-3× before October, produce a "parameter stability" table.
- **E. Presentation packaging (September)**: methodology report + charts.

**Status after the 2026-08-03 correction pass.** A is done and now reproducible (`dondur`). B is done and is the **strongest result in the project**: the system sits above the 95th percentile of random 6-stock portfolios. C is done and, once the management layer is configured as its validation says, it improves return, drawdown and Sharpe together (+18.6% vs +7.4% real CAGR, −26.8% vs −69.9%, 0.79 vs 0.39 over 8 years). D and E are open.

**`takas_farki` changed 12 → 6 in `PortfolioDoctor.KURAL` (2026-08-03, editor-approved).** 6 is the conservative edge of the plateau — statistically tied with 0 but trading less — and `sweep` independently picked 6 on the 5-year window. The code default was changed rather than a `pm_params` override because this is a settled decision, not an experiment: it belongs in git with its reasoning. `max_pozisyon` was left at 30; its evidence is marginal.

Consequence for the live experiment: managed shadow cohorts from W29 onward carry a before/after break at this date. With ~3 weeks of history the statistical cost was near zero, and running production on the wrong side of a measured cliff until October was the larger risk.

**All re-runs are done** (2026-08-03): `run`, `karsilastir` 5y and 8y, `validate 8`, `sweep 5`, `build-tech-us` + `analyze-us`, `build-tech-us-mid` + `analyze-us-mid`, `analyze-us-regime`. Every number quoted in this file now comes from the frozen panels.

### Walk-forward validation (2026-08-05) — `pipeline_backtest.py walkforward [yil] [katman]`

Replaces `validate`'s single 60/40 split with an expanding window: the period is cut into `katman+1` slices and each fold trains on everything before its test slice. Every test slice is chronologically *after* its training data and takes no part in the choice. Acceptance is stricter than `validate` on purpose — a change must win in **more than half the folds** and by **more than 5%**, where `validate`'s bare `t_s > t_v` accepted a 0.1% edge.

**Section 1 — is the management layer worth having, period by period?** NAVs come from one continuous run (the portfolio is not reset each period, so entry costs are not repeated). 8-year panel, `takas_farki=6`:

| Period | Unmanaged | Managed |
|---|---|---|
| 2018-08→2019-12 | −11.8% (dd **−69.9%**) | **+7.8%** (dd −23.8%) |
| 2019-12→2021-04 | **+167.1%** (dd −59.9%) | +113.5% (dd −25.2%) |
| 2021-04→2022-08 | **+128.3%** (dd −39.8%) | +114.8% (dd −20.6%) |
| 2022-08→2023-12 | +121.3% (dd −49.1%) | **+152.7%** (dd −19.7%) |
| 2023-12→2025-04 | +72.7% (dd −29.0%) | **+174.1%** (dd −13.9%) |
| 2025-04→2026-08 | +688.8% (dd −39.9%) | **+710.4%** (dd −13.8%) |

Return: managed wins **4 of 6**. Drawdown: managed wins **6 of 6**, almost always by roughly half. The drawdown result has no exceptions in any sub-period, which makes it a property of the layer rather than an artifact of the window — this is the single most defensible claim about the management layer.

The first period is the one to show: through the 2018 TL crisis the unmanaged track **lost 11.8% while drawing down 69.9%**, and the managed track **made 7.8% with a 23.8% drawdown**. The two periods management loses are both strong bull runs — the cost of the insurance, paid where insurance is not needed.

**Section 2 — do parameter choices survive?** No, and this replicates at both fold counts:

| Parameter | 5 folds | 3 folds | Training's picks (5-fold) |
|---|---|---|---|
| `trailing_pct` | 0/5 | 0/3 | [10, 10, 15, 20, 15] |
| `dusus_freni` | 1/5 | 1/3 | [20, 100, 100, 15, 15] |
| `max_pozisyon` | 0/5 | 0/3 | [40, 40, 40, 30, 30] |
| `takas_farki` | 1/5 | 1/3 | [20, 12, 3, 0, 6] |

**The "training's picks" column is the real finding.** For `takas_farki` the training half chose 20, 12, 3, 0 and 6 across five folds — the entire tested range. When the "optimum" moves that much with the training period, the parameter surface is mostly noise at that time scale, and any single-window optimum is a coincidence waiting to be quoted. This is also the cleanest available refutation of the two proposals `validate` had marked DOĞRULANDI: `max_pozisyon` 30→40 wins **zero** folds, `takas_farki`→0 wins one.

Note this run does **not** re-litigate the 12→6 change: the default under test *is* 6, so the question asked was "is anything better than 6", and the answer is no. Supporting detail: when training picked high values (20, 12 in folds 1-2) the test half lost (−8.4%, −22.9%); when it picked low (3 in fold 3) the test half won (+38.7%). The direction still favours the low end.

Fold-count sensitivity was checked deliberately (5 vs 3): identical verdicts on all four parameters, so the conclusion is not an artifact of how the window was cut.

### Position count (2026-08-11) — `pipeline_backtest.py pozisyon [yil]`

`MAX_POS = 6` had never been tested; it was an assumption carried since the first version. It is now tested and it stays. The command also parameterises `_pick_at(max_pos=)`, which nothing else did.

**Section 1 — performance collapses as positions are added** (8-year panel, real CAGR):

| N | Unmanaged | Managed | Managed Sharpe |
|---|---|---|---|
| 4 | +4.5% | **+23.6%** | 0.88 |
| **6 (default)** | **+7.4%** | +15.6% | 0.71 |
| 8 | +4.9% | +9.0% | 0.49 |
| 10 | −4.6% | +0.7% | 0.14 |
| 15 | −8.9% | −7.6% | −0.34 |

Going 6 → 10 costs about **15 points of real CAGR**. The edge lives at the very top of the ranking and dilutes away by rank 7–15 — which is itself weak evidence that the ranking carries real information, since that is where information *should* concentrate.

**Section 2 — but the luck band really does narrow.** 50 random N-stock portfolios from the same universe, 5–95% spread of final NAV: N=6 → **9.6×**, N=10 → 4.6×, N=15 → 3.3×. This is the honest explanation for why two live user portfolios from the same system sit 38 points apart. **It is not fixable by adding positions without destroying the edge** — the mitigation is to judge the system on cohort averages, never on one portfolio.

**Section 3 — N=4 looked better and failed walk-forward.** Training picked 4 in four of five folds, and the full-window number favoured it clearly (+23.6% vs +15.6%). Out of sample it beat the default in only **2/5** folds, and fold 3 was a collapse: 56.0% against 127.1% for N=6. Note also that N=4's advantage appears *only* with the management layer (unmanaged, 4 is worse than 6) — the same interaction signature that made `max_pozisyon` 30→40 look validated.

So: the single-window optimum was wrong for the fourth time in this project, and the walk-forward gate caught it again.

Remaining methodology work, in priority order:
1. **Per-fold system vs random-6** — the "above the 95th percentile" claim still rests on one window; showing it holds in each fold would close the "lucky period" objection.
2. **8-year `sweep`** — the 5-year sweep says every parameter is at its optimum, but that window is bull-only. Lower priority now that walk-forward covers the same ground more strictly.

Note on what each command answers, because these got conflated once already: `validate` tests whether *parameter choices* survive out of sample; it compares `_walk_managed` to `_walk_managed` and says nothing about whether the management layer is worth having. That question is `karsilastir`.

Expectation framing agreed with user: promise = "systematically better than the market; reaches real (ENAG) targets when the market allows, limits losses when it doesn't" — beating ENAG in all conditions is not promisable (even XU100 fails some years).

## UI state (2026-07-17)

BIST menu: Kokpit (default landing, "Piyasa Defteri" masthead + macro strip + signals + system status) · Portfoy Yoneticisi · BIST Listesi · Hisse Analizi · Portfolyum · Backtest · Sinyal Takip. "Piyasa Ozeti" and "Sistem Portfolyleri" retired from menu 2026-07-17 (render functions kept in code; macro strip moved to Kokpit via `_kokpit_macro`). Two themes via sidebar "Baskı" selector persisted in `?tema=` query param: Gazete (light, Playfair/Source Serif, default) and Obsidyen (dark graphite, Inter, mint accent) — palette in `THEMES`, all charts/inline colors read `_theme()`. Mobile rule learned the hard way: NEVER put primary controls in the sidebar (collapsed on phones) — backtest and analysis inputs were both moved to main area for this reason.

**TCMB policy rate in the macro strip (2026-07-31).** `_tcmb_politika_faizi()` parses the MPC decision sentence on tcmb.gov.tr ("politika faizi olan bir hafta vadeli repo ihale faiz oranının yüzde XX'de ..."), two regex patterns cover the sabit/indirilme/yükseltilme phrasings. EVDS API was skipped because it requires registering a key. Same defences as the ENAG scraper: a code-side fallback constant (`TCMB_POLITIKA_FAIZI_YEDEK`), a 0–100 sanity bound, a 6-hour cache (MPC meets ~8×/year), and a source label so a fallback value reads "son bilinen" instead of "politika faizi". BIST strip only — it would be noise on the US strip, which already carries VIX.

**52-week position labels were saying the opposite of the engine (fixed 2026-08-03).** The Risk/Indicators tab called ≥80% "Yıllık Zirveye Yakın — Yüksek Risk" (red) and ≤20% "İdeal Alım Bölgesi" (green) — leftovers from the pre-calibration contrarian scoring. Meanwhile `TechnicalEngine._compute_score`'s momentum branch awards **+10** for ≥80% and **−6** for ≤20%, because 52-week position is the strongest signal in the whole calibration. The UI bands are now copied one-for-one from that branch (0.80/0.60/0.40/0.20) so the screen explains what actually produces the score, plus a caption saying so. `_tech_style_for` returns `"momentum"` for both markets and `render_analysis_page` is BIST-only, so the label is unconditional.

Deliberately **not** changed: [bist_analyzer.py:2799](bist_analyzer.py:2799) still gives `+3` for `week52_pos <= 0.70`. That one is inside a single defensive portfolio style (paired with a low-ATR bonus and an RSI 30–62 band), i.e. an intentional tilt within a named strategy rather than a stale contrarian remnant. Changing it would alter portfolio construction while shadow cohorts are mid-flight.

**Open item — do NOT wire the rate into the regime score untested.** Rate direction is *obviously* meaningful for equities, which is exactly the trap: contrarian scoring, the drawdown brake and tight stops were all "obviously right" and all measured badly. Test first — forward BIST returns after MPC cuts vs hikes, using the `weight_calibration` method. Historical policy rates need EVDS (one-time key registration).

**UI backlog (user-approved deferrals):** EN translations half-done (option kept visible for now); US Markets tab unmaintained (kept visible); Zaman Makinesi still simulates the OLD portfolio logic (warning caption added; eventual rewrite on pipeline_backtest logic); Kokpit "Günün Sinyalleri" depends on scan cache (could use an inline scan button).

## Security

The public Streamlit Cloud demo runs with Turso WRITE credentials and Streamlit has no built-in auth, so an anonymous visitor could otherwise corrupt shared data (portfolios, NAV, and especially the ENAG inflation table that every real-return figure depends on). Write protection (2026-07-23): if the `APP_EDIT_KEY` secret is set, all UI write actions (portfolio save, ENAG rate edit, archive) require unlocking via the sidebar "Editör girişi" box (`hmac.compare_digest`); reads stay public. If the secret is unset (local dev), writes are unguarded — behaviour unchanged. Helpers: `_writes_locked()` / `_guard_write()` near `_get_secret`. The daily robot writes to Turso directly (not through the UI), so it's unaffected — **do not** route robot writes through `_guard_write`. To arm protection on Streamlit Cloud, set `APP_EDIT_KEY` in the app's secrets. ENAG `set_rate` UI input is also validated (YYYY-MM format + −50…100% bound). Local-SQLite user-data writes (Portfolyum add/remove, history delete) and the expensive ~500-stock rescan are guarded too — on Streamlit Cloud `DB_PATH` is `/tmp/bist_cache.db`, a SINGLE file shared by every anonymous visitor of the demo, so those tables are multi-tenant by nature. **Do not enter real personal holdings into Portfolyum on the public demo** — reads stay public, so any visitor can see them; the write guard stops pollution, not disclosure. `.github/workflows/daily-robot.yml` runs with least privilege (`permissions: contents: read`). **Dependencies are pinned exactly (`==`) in requirements.txt as of 2026-07-23** — supply-chain protection: CI runs `pip install` with `TURSO_AUTH_TOKEN` in env, so with `>=` a hijacked release of a small package (borsapy, tradingview-screener, streamlit-option-menu) would auto-install and could exfiltrate the token. Pins were verified resolvable for CI's Python 3.12 on Linux via `pip install --dry-run --only-binary=:all: --no-deps --python-version 3.12 --platform manylinux_2_28_x86_64` (note: pandas 3.x has no manylinux2014 wheel — it needs glibc ≥2.28, which ubuntu-latest and Streamlit Cloud both satisfy; re-run this check whenever pins change). Trade-off: **automatic security updates stop, so bump versions manually every few months** — `pip install -U <pkg>` → run the app and verify → update the number here → commit. Only direct dependencies are pinned; transitive ones still float (a hash-pinned lockfile via pip-tools/uv is the remaining open item). External RSS content rendered via `unsafe_allow_html` is `html.escape()`d and links are scheme-checked (http/https only) — see news rendering in bist_analyzer.py `_render_kokpit`/economy-news and news_engine.py `render_news_panel`. SQL is parameterized throughout; no eval/exec; no hardcoded secrets (all via `_get_secret`).

**Repository visibility (2026-07-31).** The user decided to make the repo **private** (done via GitHub web UI; `gh` CLI is not installed on this machine). Rationale: CLAUDE.md, the code comments and `weight_calibration.py` together expose every calibration finding and parameter, so trimming the README alone would have been theatre — the coherent lever is repo visibility. Verified before the switch that **no secret ever leaked**: full-history scan found no `.env`, no `secrets.toml`, and no real `sk-ant-`/JWT/Turso-URL value in any commit; the only config file ever tracked is `.streamlit/config.toml` (theme/server settings only). If the repo goes public again, review CLAUDE.md and both READMEs together — they now carry the full evidence trail.

Two Actions notes for a private repo: minutes are metered (free tier 2000/month; this robot uses ~220), and GitHub disables scheduled workflows after 60 days with no repository activity — check the Actions tab occasionally during quiet stretches.

## Known constraints

- Turkish news RSS sources are unreliable — several return 404 or their feeds don't mention the ticker in the title, so `_is_relevant` filters them out. Google News TR is the most consistently working source. `test_rss.py` is the way to check current source health before debugging "why is sentiment score always 50" (low article count triggers a confidence pull toward neutral 50).
- KAP (Turkish public disclosure platform) is bot-protected — direct API/RSS access doesn't work. KAP disclosures are fetched via the third-party `borsapy` library instead (`_src_kap` in news_engine.py, graceful degrade if borsapy missing). borsapy's heavy deps (openai, pymupdf, onnxruntime) are lazy-loaded, so importing it costs no RAM.
- Comments in bist_analyzer.py are in Turkish; user-facing UI strings are Turkish (with an EN/TR/BOTH language toggle for AI-generated summaries).
