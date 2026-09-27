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
python skor_gunlugu.py yaz              # daily component log (robot runs it; ~13 min, writes Turso)
python skor_gunlugu.py ozet             # what the component log holds, per day

python pipeline_backtest.py dondur 5    # (re)download the price panel for a window
python pipeline_backtest.py dondur-evren # re-fetch the UNIVERSE — new epoch, rare, invalidates old results
python pipeline_backtest.py run         # full pipeline sim vs XU100 / BIST-30 / random-6
python pipeline_backtest.py karsilastir # managed vs unmanaged — the presentation's core table
python pipeline_backtest.py sweep       # management-rule parameter sweep
python pipeline_backtest.py validate 8  # out-of-sample validation (8y, single 60/40 split)
python pipeline_backtest.py walkforward 8 5  # walk-forward: 5 expanding folds (stricter)
python pipeline_backtest.py pozisyon 8  # position-count sweep + luck-band measurement
python pipeline_backtest.py slippage 8  # market impact — how much capacity does the edge have
python pipeline_backtest.py kurulum 8   # construction params (score_min, sector_cap, corr, volume)
python pipeline_backtest.py sanstest 8  # system vs random-6, period by period
python pipeline_backtest.py dsr 8       # deflated Sharpe — multiple-testing correction
python pipeline_backtest.py us 8        # US transfer test — out-of-sample in space
python pipeline_backtest.py uk 8        # UK transfer test — the method did NOT transfer
python pipeline_backtest.py transfer UK 8   # generic form; markets in TRANSFER_PIYASA
python pipeline_backtest.py kasa        # data vault status — what is locked
python pipeline_backtest.py kasa-ac 8 EVET  # OPEN the vault — once, in October

python tools/kap_gecmis.py cek 2018-01-01 2024-08-31  # pull KAP history (weekly chunks, ~35 min)
python tools/kap_gecmis.py ozet         # what is in kap_veri/
python tools/kap_olay.py --min-ornek 50 # event study: which disclosure type moves the price
python tools/kap_ayrim.py sermaye       # split a type by summary text and measure each group
python tools/kap_metin.py cek "Kredi Derecelendirmesi" 800   # fetch disclosure bodies
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

#### News engine audit (2026-09-26) — the 25% sentiment weight was measuring artifacts

The sentiment component had never been checked end to end. It was audited on
demand and the verdict is that it did not work — not because news is
unpredictive, but because of four separate defects, each of which produced
confident output from input that carried no information.

**First, the finding that constrains everything else: the component cannot be
validated historically.** `analyze_news_for_date` queries Google News with
`after:`/`before:` operators, and Google serves almost no archive through that
interface. Probed on ten of the most heavily covered BIST names across
2019–2024: **eight returned zero articles**, and all ten returned "notr".
GARAN having no news in a week of January 2022 is not reality, it is a
retrieval failure. So `news_backtest_cache` (1019 rows, 42.6% with zero
articles) is largely an artifact of the same failure, and **no backtest can
ever measure whether the news layer helps.** Only forward logging can.

**What the live feed actually contains.** 12 major stocks, 215 items:

| | Share |
|---|---|
| KAP bureaucratic form titles ("Özel Durum Açıklaması", "Şirket Genel Bilgi Formu") | 48% |
| Technical-analysis content farms, live-chart pages, bare company names | ~18% |
| Other companies matched by name similarity | ~5% |
| Genuinely informative, company-specific news | a handful |

And the classification of those 215 items was **20 positive, 1 negative, 194
neutral**. One negative item across twelve large caps in a week is not a
market observation; it is a broken classifier.

**The four defects, each measured before fixing:**

1. **Keyword matching was substring, not word, matching.** Turkish is
   agglutinative and `"kar" in text` matches *kar*ar, *kar*şılık, *kar*go.
   Measured: `"Yönetim Kurulu KARar Özeti"` → **positive**; `"Rekabet Kurulu
   KARarı açıklandı"` → **positive**; `"KARşılık artışı nedeniyle zarar
   açıklandı"` → **neutral**, because the spurious positives cancelled two real
   negatives. Fixed with a root-plus-closed-suffix-set regex (`_kelime_kalibi`)
   — `\bkar(ı|a|da|…)?\b` matches "kârı" but not "karar".
2. **Corporate boilerplate sat in the positive word list.** `"yatırım"` appears
   in the name of every Turkish brokerage, so `"Yatırım Kuruluşu Varant -
   Sertifika"` — a routine warrant filing — scored positive. This is what
   produced GARAN's 65.3, the highest score in the sample. `"ortaklık"` is worse:
   in KAP legal language it simply means "the listed company", and it fired
   **twice** (both spellings) on `"Ortaklık Aleyhine DAVA Açılması"`, outvoting
   the lawsuit and marking it **positive**. Both removed and replaced with
   directional phrases.
3. **Relevance matching was substring too.** KCHOL's feed contained *"İskoçya,
   Galler ve Kuzey İrlanda liderleri bağımsızlık için buluşuyor"* — because
   İs-**koç**-ya contains the alias "koç". The same defect matches BIMSA on
   "bim" and Oxford on "ford". Aliases are now word-bounded. Separately, KAP
   titles arrive as `KAP *** COMPANY *** TICKERS *** form`, so GARANTİ PORTFÖY
   (GPY) and GARANTİ FAKTORİNG (GARFA) filings were landing in GARAN's feed;
   `_kap_baskasinin` now requires our ticker in the code field.
4. **Content-free items inflated confidence.** `confidence_factor =
   total_news / 10` scales how far the score may move from 50, and form titles
   counted toward `total_news`. So 36 bureaucratic headers made the system
   *more* certain of a score built on nothing. Such items are now excluded from
   scoring entirely (`_bilgisiz_baslik`), like the speculation filter.

**The eyeball check between rounds is what kept this honest.** After the first
fix the negative count jumped 1 → 13 and it looked like a success — until the
headlines were read. Nearly every new negative was *"Temerrüt İşlemi"*, a
routine Takasbank settlement-default notice about a market participant, not the
company. FROTO's score had fallen to 21.8 entirely on those. The first round had
swapped false positives for false negatives; only reading the output caught it.

**After both rounds**, same 12 stocks: 188 of 236 items excluded as
content-free, and the five items that move a score are

| Stock | Label | Headline | Verdict |
|---|---|---|---|
| ASELS | positive | ROKETSAN contract, €1.23bn | correct, material |
| YKBNK | negative | lawsuit filed against the company | correct (was positive before) |
| GARAN | positive | analyst raises target to 169.30 TL | reasonable |
| TUPRS | positive | 2027 dividend forecast article | weak |
| BIMAS | positive | "dividend payment will cause price adjustment" | **wrong** — that is the ex-dividend drop |

So the layer is now honest but thin: it says something for roughly a third of
stocks and exactly 50.0 for the rest, and it still makes context errors of the
same keyword-without-meaning kind.

**Consequence for the weight, and it is the real output of this audit.** For 8
of 12 stocks the component returns exactly 50.0. At a 25% weight that is not
merely uninformative — every stock receives an identical 12.5-point
contribution, which **compresses the composite's cross-sectional spread by a
quarter** and dilutes the one component with measured predictive power. This is
the mechanism behind the already-recorded −0.010 rank correlation between
sentiment and the composite. The component cannot be validated historically,
measurably adds no ranking information, and actively shrinks the signal it is
averaged with, so **25% is indefensible**. The principled fix is the pattern the
codebase already uses for missing analyst targets: when there is no real news,
**redistribute the weight to Technical rather than applying a neutral 50**, and
cut the base weight when there is.

**APPLIED 2026-09-26 (user-approved).** `WEIGHTS` is now Technical 60 / Sentiment
10 / Premium 20 / Valuation 10, and `compute_bist_score` gained a `has_news`
branch: when the engine found no real item (`total_news == 0`) the sentiment
weight goes to Technical instead of averaging in a neutral 50. The
missing-target branch was redirected too — it used to hand half the Premium
weight to Sentiment (25 → 35), filling a data gap with the component measured at
~0 contribution; it now all goes to Technical. Effective weights:

| Case | Tech / Sent / Prem / Val |
|---|---|
| everything present | 60 / 10 / 20 / 10 |
| no real news | **70 / 0** / 20 / 10 |
| no analyst target | 80 / 10 / 0 / 10 |
| neither | 90 / 0 / 0 / 10 |

Verified end to end: ASELS 34.2×60 + 52.2×10 + 76.5×20 + 20.0×10 = 43.0. Not 0%
but 10%, because after the audit the layer does catch real events (ASELSAN's
€1.23bn ROKETSAN contract, the YKBNK lawsuit) — thin, but not nothing.
`SCORING_VERSION` bumped to **v3** so Signal Tracker can separate v2 and v3 hit
rates. This is the third live-cohort break (after `takas_farki` 2026-08-03 and
`skor_cikis` 2026-08-29); per the note below it resets the selection-vs-random
track but not managed-vs-passive.

#### No usable Turkish financial sentiment model exists (researched 2026-09-26)

Checked before considering a model swap, because hand-extending keyword lists is
not sustainable. The obvious upgrade path turns out not to be available:

| Model | Verdict |
|---|---|
| [`onerayhan/bert_finance_BIST100_turkish`](https://huggingface.co/onerayhan/bert_finance_BIST100_turkish) | The only *financial* Turkish model, and **unusable**. No `config.json`, no tokenizer, no safetensors — a bare TF SavedModel. Worse, its labels are **the sign of the XU100 daily return on the publication date**, so it learned "what text appears on up days", i.e. market beta rather than per-stock news direction. Binary, no neutral class. University coursework, ~3 years old. |
| [`savasy/bert-base-turkish-sentiment-cased`](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased) | The one `news_engine` already references. Movie/product reviews (48,290 rows), 95.4% accuracy *on reviews*, binary with **no neutral class**, and **no license stated**. |
| [`saribasmetehan/bert-base-turkish-sentiment-analysis`](https://huggingface.co/saribasmetehan/bert-base-turkish-sentiment-analysis) | 3 classes, MIT, 444 MB, 0.962 accuracy — but fine-tuned on only 10k rows, and its **neutral class is Turkish Wikipedia prose**. A flat factual BIST headline reads like Wikipedia, so it would likely dump everything into neutral. |

All three are ~110M-param BERT-base: roughly 1.2–1.8 GB peak RSS and 2–5 minutes
for ~2500 headlines on CPU — comfortable on a **GitHub Actions runner (7 GB)**,
which matters because the daily robot runs there, not on RAM-constrained
Streamlit Cloud. Capacity is not the blocker; **suitability is**. None of them
has seen "kâr", "zarar" or "temerrüt" in a financial sense.

**The real blocker is measurement, not the classifier.** Every option — keywords,
a BERT swap, a paid API — has to be scored against something, and there is no
ground truth at all: no historical news (Google serves no archive), no labelled
headlines, and component scores were never durably logged (`analysis_history`
holds 32 rows and lives in `/tmp` on Streamlit Cloud). A model swap today would
be one more untested change. The cheapest thing that unblocks every other option
is **a few hundred hand-labelled Turkish headlines**: after that the current
keyword system, any candidate model and any future method can all be scored on
the same set, and a small fine-tune becomes possible.

Ruled out on cost: `borsapy`'s `tweets()` needs Twitter auth plus a paid extra.
Still untapped and free: `get_news_content(disclosure_id)` returns the **full KAP
disclosure HTML** in ~0.2 s (verified — note the naive tag-strip returns the
site's navigation chrome, so it needs a parser aimed at the disclosure body),
and KAP form titles are a *closed* vocabulary of routine types rather than open
language, so mapping them is finite work. `income_stmt` / `balance_sheet` /
`cashflow` / `quarterly_*` expose actual financial statements, so earnings
growth is computable numerically with no NLP at all.

Also removed: the dead `NewsScraper` class in bist_analyzer.py (~50 lines), a
second Yahoo-scraping news engine carrying its own cruder word list ("hedef" and
"uzerinde" positive, "risk" negative) with the same substring defect. It was
never called. Removing it is not about dead weight but about the duplicate-copy
hazard this file keeps recording — the one place a fix does not reach.

**Note on the live-cohort break, for whenever the weight is changed.** A scoring
change resets the "does selection beat random" track, but **not** the
managed-vs-passive comparison: those two cohorts are built from the *same* picks
on the same day, so both get the new scores and the difference between them
stays clean. The most valuable live experiment survives a scoring change; only
the weaker one restarts.

#### KAP history is reachable after all — the news layer can finally be validated (2026-09-26)

The audit above closed with "this component can never be validated
historically". That turned out to be wrong, and the correction matters more
than the audit: **KAP's own query API serves the full disclosure history of the
entire market, free, back to at least 2018.**

Found by reading `borsapy`'s KAP provider, noticing it fetches a Next.js page
and truncates client-side, then extracting the real API routes from the site's
own JS bundles rather than guessing:

    POST https://www.kap.org.tr/tr/api/disclosure/members/byCriteria
    {"fromDate": "2024-03-01", "toDate": "2024-03-01", "memberTypes": ["IGS"],
     "mkkMemberOidList": [], ...}        # empty list = the whole market

Measured: ASELS returns 145 disclosures for 2018 and 168 for 2023, so the
archive spans the entire backtest window. One market-wide day comes back in
~0.2 s. A full month of every BIST company took **11 seconds: 7,302
disclosures, 723 stocks, 125 distinct types**. Eight years is roughly 17
minutes of polite pulling.

**The returned record is far better than the disclosure title we were parsing:**

| Field | Why it matters |
|---|---|
| `subject` | the disclosure **type** — the closed vocabulary already labelled in `kap_sozluk.json` |
| `summary` | a short human-written description, **91% populated** — "Yatırım Teşvik Belgesi Alımı Hakkında", "Uzun Vadeli Kredi Anlaşması", "Genel Müdür Ataması". Vastly more informative than the form title, and it is what any classifier should read |
| `publishDate` | date **and time**, so an event study can align to the next open |
| `isLate` | late-filing flag — a candidate signal on its own |
| `disclosureIndex` | id for `get_news_content` when the full text is needed |
| `disclosureClass` | ODA / DKB / DG / FR top-level taxonomy |

Note the `summary` examples answer a question raised separately: **government
incentives are in this data as a searchable field** ("Yatırım Teşvik Belgesi
Alımı"), with eight years of history — so the subsidy idea is testable the same
way everything else is, without hand-building an event list from memory.

**Two limits, both measured, and the first one bites silently.** The endpoint
caps a response at **2000 records and returns only the tail of the range
without saying so** — a one-month query came back with 2000 rows covering only
the last seven days. Daily chunking avoids it (300–700/day) and
`tools/kap_gecmis.py` shouts if any day reaches the cap. Second, no filter
other than `fromDate`/`toDate` is honoured; paging parameters are ignored.

`tools/kap_gecmis.py` does the collection: day-by-day, resumable, one JSON per
day under `kap_veri/` (gitignored, regenerable), 0.35 s between requests.

**What this unblocks, and the discipline that applies.** With dates, tickers and
the frozen price panel we can now run a proper event study: for each disclosure
type, the average abnormal return over the following 1/7/30/90 days. That turns
every direction in `kap_sozluk.json` from a hypothesis into a measurement — and
several of them deserve to fail. **The data vault applies here too**: disclosures
after 2024-08-31 are inside the reserve, so the event study must stop at the
cutoff minus the horizon, exactly as `_panel_oku` already does for the
calibration panels. Building the news layer on reserve data would burn the vault
for the sake of a component measured at ~0 contribution.

#### The KAP event study ran: one signal out of twelve hypotheses survived (2026-09-26)

411,656 disclosures collected (2018-01-01 → 2024-08-30, 1,031 stocks, 192
types, 90% carrying a summary), then measured against the frozen price panel:
entry at the next close after publication, 30-day return minus the panel's
cross-sectional median, t-statistics clustered by day. 163,919 events matched a
panel stock. The vault applies — the panel is trimmed at the cutoff, so forward
windows reaching into the reserve drop out.

**The headline is a negative result about my own labels.** `kap_sozluk.json`
carried 12 directional hypotheses, each derived by reading what the disclosure
type means and reasoning about whether it should be good or bad news. **One
survived.**

| Type | n | 30d median | pos% | t | Hypothesis | Verdict |
|---|---|---|---|---|---|---|
| **Payların Geri Alınması** (buyback) | 5,158 | **+2.63%** | 58 | **+4.6** | + | **held** |
| Kar Payı Dağıtımı (dividend) | 5,100 | +0.37% | 51 | +0.4 | + | noise |
| Yeni İş İlişkisi (new business) | 2,098 | −1.23% | 47 | −1.2 | + | noise |
| Finansal Duran Varlık Edinimi | 1,087 | +0.29% | 50 | −0.3 | + | noise |
| Ortaklık Aleyhine Dava (lawsuit) | 494 | −0.41% | 48 | −0.5 | − | noise |
| **Faaliyetlerin Durdurulması** (operations halted) | 360 | **+0.01%** | 50 | **+0.0** | − | noise |
| Maddi Duran Varlık Alımı | 359 | −0.90% | 49 | +0.3 | + | noise |
| Pay Alım Teklifi (tender offer) | 182 | +0.91% | 55 | +0.6 | + | noise |
| **TTK 376** (capital erosion / technical insolvency) | 176 | **−0.51%** | 49 | **+0.0** | − | noise |
| İzahname ×3 (prospectus) | 53–70 | +1.5 … +3.8% | 54–62 | +0.8…+1.3 | − | noise, wrong sign |

The two I marked **high confidence** — a company halting operations and a
lawsuit filed against it — came back at **+0.01%** and **−0.41%**, i.e. exactly
nothing. So did TTK 376, which is the legal threshold for having lost a
significant part of share capital. Reading a disclosure type and reasoning about
its direction predicted returns **about as well as chance**. The dictionary's
`yon` values are now set to 0 wherever the measurement said noise, and each type
carries an `olcum` block with n, median, pos% and t so the claim is inspectable.

**A small-sample lesson, and a sharp one.** On partial data (136 events) "Yeni
İş İlişkisi" showed −4.73% at t=−4.3 and I reported it as a striking finding
with the caveat that it was provisional. At 2,098 events it is −1.23% at t=−1.2.
The whole effect was sampling noise, and the day-clustered t did not protect
against it. This is the fifth or sixth time in this file that a confident number
dissolved when the sample grew.

**What actually survived, and why it is believable.** Of 92 types with n≥50, 9
cleared |t|>2.5, and most of those are artifacts the tool already warns about:

- **Buyback +2.63%** (n=5,158, t=+4.6) — the one real signal, and note it is the
  one with a *mechanical* cause rather than an interpretive one: the company is
  literally buying its own shares, which is demand, not news about prospects.
- **Sermaye Artırımı − Azaltımı −2.01%** (n=3,480, t=−3.1) — capital increase,
  negative as dilution predicts. Labelled "read the content" because bedelli
  (rights, dilutive) and bedelsiz (bonus, neutral) differ; the negative average
  says the dilutive cases dominate. Splitting it by summary text is the obvious
  next step.
- **Özel Durum Açıklaması (Genel) −0.21%** (n=25,988, t=−2.7) — the generic
  material-event wrapper, slightly negative across 26k events. Tiny, but a real
  "sell the news" tilt.
- Discarded as artifacts: **Yatırım Kuruluşu Varant** (+3.39%, t=+8.4) and
  **Piyasa Yapıcılığı** (+2.51%, t=+2.9) are *selection* — brokerages write
  warrants and make markets in whatever is already hot; **Olağan Dışı Fiyat ve
  Miktar Hareketleri** (−3.03%) is reverse causality, it fires because the stock
  moved; **Entegre Rapor** (+5.17%, n=66) is a routine annual filing on a sample
  too small to mean anything.

**What this says about the original goal.** The ask was to generate signals from
KAP disclosures. The measured answer is that KAP yields **one** robust signal,
not many, and it is not the kind that comes from understanding the news — it
comes from a company buying its own stock. That is worth having and it is worth
wiring in. It is not a news-comprehension engine, and the 12-for-1 record says
plainly that building one on top of type labels would be building on sand.

Collection cost, for the record: `tools/kap_gecmis.py` pulled all 348 weeks in
~35 minutes including retries. Three schemes were tried first and two were
eliminated by measurement — day-by-day market-wide got rate-limited, and
per-stock queries return **HTTP 500 for any range beyond about a year** (ASELS:
1 year → 200 with 168 records, 2 years → 500). Weekly market-wide chunks work,
stay under the 2000-record cap, and the 21 weeks that did hit the cap were split
to daily automatically.

#### Reading the content: the bonus-issue signal, and dilution theory backwards (2026-09-26)

The event study left 27% of the flow in a "read the body" bucket. Before
downloading 400k disclosure bodies it was worth checking the cheap route: the
`summary` field is already on disk for all 411,656 records and is 90%
populated. Measured which types carry their direction in the summary:

| Type | Direction in summary? |
|---|---|
| Sermaye Artırımı | **yes** — "bedelli" 985, "bedelsiz" 1396 |
| İhale Süreci / Sonucu | partly — "Sonucu" 22%, "Süreci" 19%, "kazan" only 4% |
| Kredi Derecelendirmesi | **no** — "yükseltildi" appears twice, "düşürüldü" never |
| Geleceğe Dönük Değerlendirmeler | no |

So the capital-increase split is free, and it is the one with a clean
pre-specified hypothesis: **bedelli** (rights issue) dilutes existing holders
and should underperform **bedelsiz** (bonus issue), which is an accounting
operation and should be neutral. `tools/kap_ayrim.py` tests it.

**The hypothesis is falsified, and the sign is reversed:**

| Group | n | 7d | 30d | pos% | t |
|---|---|---|---|---|---|
| BEDELLİ (theoretically dilutive) | 664 | −0.89% | −1.04% | 48 | −0.2 |
| **BEDELSİZ (theoretically neutral)** | **1160** | **−1.62%** | **−3.86%** | **41** | **−5.0** |
| AZALTIM | 50 | +0.43% | +4.23% | 51 | −0.1 |

Bedelli is noise. **Bedelsiz — the one that should do nothing — is the strongest
single content-derived effect found so far.** It also explains the type-level
−2.01%: that number was bedelsiz, not dilution.

**Checked for the obvious artifact first, because a bonus issue mechanically
cuts the share price** (share count rises, price divides). If the panel's split
adjustment were imperfect the −3.86% would be arithmetic, not information.
Daily abnormal returns indexed to the announcement:

| Day | −3 | −2 | −1 | **0** | +1 | +2 | +3 | +5 | +15 → +30 |
|---|---|---|---|---|---|---|---|---|---|
| median | +0.26 | +0.28 | +0.43 | 0.00 | −0.52 | −0.41 | −0.35 | −0.22 | ≈0 |

No single-day cliff, so it is not a split artifact. The shape is a **run-up into
the announcement and a bleed out of it**, concentrated in the first week and
done by day 15 — textbook buy-the-rumour-sell-the-news, and in BIST the
"bedelsiz" chase is a well-known retail habit. The pre-announcement run-up (+0.26
/ +0.28 / +0.43 on days −3/−2/−1) is itself notable: something leaks.

One more detail worth keeping. On almost every day after the event the **mean is
positive while the median is negative** (day +1: mean +0.00 vs median −0.52; day
+7: mean +0.42 vs median −0.06). Bonus issues are a lottery — the typical stock
drifts down, a minority rip. For a long-only system built on medians that
matters: the expected pick is a loser even though the headline cases are winners.

**Usable form**, since the system cannot short: treat a bedelsiz announcement as
a reason **not to buy for the following week or two**, not as a buy trigger. It
is the second measured signal after buybacks, and unlike buybacks it is
*negative*, which is the kind the scoring layer currently has none of.

İhale Süreci/Sonucu was split the same way and came back empty — result −0.99%,
process −0.33%, neither significant (|t|≈0.1). So the tender type needs the body
text or nothing at all.

**A tool bug found on the way, and it was the dangerous kind.** The
hypothesis-verdict line hardcoded `fark < 0` as "matches", which was right for
the capital-increase case it was written against. Applied to the tender split —
whose hypothesis runs the other way — it printed **UYUYOR for the exact
opposite** of the stated hypothesis. The expected sign is now declared per split
in the config, and the verdict refuses to rule at all when neither group clears
|t|>2.5 rather than reading meaning into a difference of two insignificant
numbers.

Scoreboard on content so far: **two splits attempted, one real signal.** Better
than the 1-in-12 record of labelling types by reasoning, and the reason is the
same as with buybacks — the effects that survive are the ones with a mechanical
cause (a company buying its own shares; retail crowding into a bonus issue),
not the ones where I reasoned about whether news sounded good or bad.

#### Where the KAP signal search stands, and two more free tests that failed (2026-09-26)

**Standing: two measured signals out of sixteen attempts.**

| Signal | n | 30d median | t | Usable as |
|---|---|---|---|---|
| **Payların Geri Alınması** (buyback) | 5,158 | **+2.63%** | +4.6 | positive |
| **Bedelsiz sermaye artırımı** (bonus issue) | 1,160 | **−3.86%** | −5.0 | avoid for ~2 weeks |

Failed: 11 directional type hypotheses, the tender process-vs-result split, the
late-filing flag, and seven subgroups of the biggest category. Both survivors
have a **mechanical** cause — a company buying its own stock, retail crowding
into a bonus issue. Nothing I reasoned my way to survived.

**Two more tests, both free** — the data was already on disk, no requests:

*Late filing (`isLate`).* Present on all 411k records and a documented distress
marker in the literature. 1,505 late filings, 616 matched to panel stocks:
**−0.75% at t=−0.0.** Nothing. This also kills the dictionary's
"Finansal Rapor Ek Süre Talebi" label, which was hypothesised negative.

*The biggest category, split by summary.* "Özel Durum Açıklaması (Genel)" is
47,408 stock-events. Seven subgroups defined **before** looking at results:

| Subgroup | n | 30d | t |
|---|---|---|---|
| yatırım/teşvik | 538 | −2.70% | −2.2 |
| kredi/borç | 255 | −2.51% | −2.1 |
| üretim/kapatma | 1,224 | −1.24% | −1.3 |
| satın alma | 254 | −1.02% | −0.5 |
| sözleşme/ihale | 2,328 | −0.55% | −0.8 |
| hukuk/ceza | 615 | −0.53% | −0.2 |
| yönetim değişikliği | 3,134 | +0.00% | +0.3 |

None clears |t|>2.5. Note the investment-incentive group: **announcing a
government incentive or a new investment is followed by −2.70%**, the most
negative of the seven. It does not clear the bar, but it is the opposite of the
intuition that drove the original interest in subsidies — and it is the fourth
time an "obviously good news" category has come back flat-to-negative.

**A pattern across all of it worth stating: every subgroup is negative or zero,
none positive.** Combined with the type-level −0.21% on 26k events, the honest
reading is that KAP disclosures are, if anything, **mildly contrarian** — the
announcement marks the point where the news is already in the price. That is a
finding about the market, not a defect in the measurement, and it is consistent
with the bedelsiz run-up-then-bleed shape.

Also learned from the word counts: about 5,700 of those 47,408 summaries share
one boilerplate template about the weighted-average price exceeding a multiple —
i.e. exchange-mandated *responses to price moves*. Reverse causality is sitting
inside the biggest category, which partly explains why it measures at zero.

**Credit ratings are the remaining candidate and they need real work.** The body
text is reachable — `oda_ExplanationTextBlock` in the disclosure's XBRL table,
verified on 20 fetches with zero failures via `tools/kap_metin.py`. But keyword
direction fails: 75% come back undetermined because the announcement **states**
the rating ("Uzun Vadeli Ulusal Notunu AAA (Tr)") without saying whether it moved.
Getting direction means parsing the rating symbol and comparing it to that
company's previous rating — structured, no language understanding needed, and we
have the history to do it. Cost: ~3,500 fetches at 2s, about two hours, plus
normalising symbols across JCR / Kobirate / Fitch scales. Worth doing, but it is
the expensive option and it was correctly deferred until the free tests were
exhausted.

#### The measured KAP signals are now wired into the live score (2026-09-27)

Until now the dictionary and the event study lived in `tools/` and
`kap_sozluk.json` and touched nothing. Checking that turned up something worse
than "not connected": the live keyword lists had one of the measured signals
**backwards**.

| Live engine before | Measured | |
|---|---|---|
| `("pay geri alımı", 2)` strong positive | +2.63%, t=+4.6 | ✅ correct by luck |
| **`("bedelsiz", 2)` strong positive** | **−3.86%, t=−5.0** | ❌ **backwards** |
| `("sermaye artırımı", 2)` strong positive | −2.01%, t=−3.1 | ❌ backwards |

So the scoring layer was awarding points for the single most negative event in
the whole study. (A correction to an earlier note in this file: `OFFICIAL_TERMS`
is a *confidence* bonus map, not a direction map — only `FIN_POSITIVE_WEIGHTED`
and `POSITIVE_WORDS` set sentiment.)

**Three changes, all in `news_engine.py`:**

1. **Signs.** `bedelsiz` and `sermaye artırımı` moved from
   `FIN_POSITIVE_WEIGHTED` to `FIN_NEGATIVE_WEIGHTED`, and `bedelsiz` added to
   the plain `NEGATIVE_WORDS` so both classifier paths agree. `pay geri alımı`
   stays positive — it is the one the measurement confirms. Every entry carries
   its n / median / t in a comment.
2. **The dictionary is now authoritative for KAP items.** `_kap_sozluk()`,
   `_kap_turu()` and `_kap_yon()` load `kap_sozluk.json` (114 types, 98% of the
   archive) and `_bilgisiz_baslik` consults it before the hand-written regex
   list. A measured direction is returned from the very top of `_classify`,
   ahead of the language gate and every keyword layer — measurement beats word
   counting, and where the dictionary has no measured direction it returns 0 and
   the normal layers run.
3. **Repeats of the same KAP type collapse to one event.** Found in the
   end-to-end test, and it mattered: MAVI scored **82.0** off *five* daily
   notices of one buyback programme. A company files a notice every day it buys;
   the measured effect belongs to the programme, not to the notice count, and
   the event study's overlapping windows were already flagged as an upper bound.
   Collapsing to the newest per type took MAVI to 64.2 and TTKOM 71.7 → 56.7.

**Why the dictionary rather than editing word lists.** This week's live flow
showed the alternative failing: "Esas Sözleşme Tadili" — amending one clause of
the articles of association, pure paperwork — scored **positive in six separate
stocks** because "sözleşme" is in `POSITIVE_WORDS`. Same for "Sürdürülebilirlik
Raporu" via "sürdürülebilir". Deleting those words is not the fix, because a
real contract *is* good news; the fix is recognising the KAP **type**. Both now
filter out as routine before any keyword sees them.

**Verified on this week's live market flow** (2026-09-22 → 26, 1,372
disclosures): 799 routine and filtered (58%), 143 producing a measured signal,
396 still in the "read the body" bucket, 9 unknown. 28 items where the old path
said positive and the dictionary disagrees — including KLRHO and MHRGY bonus
issues, now correctly negative. End to end on live stocks: EKGYO 67.8, TTKOM
56.7, MAVI 64.2, each driven by a buyback notice plus genuine free-text news,
with the free-text path (ASELSAN's €1.23bn contract) still working.

**This is the fourth live-cohort break** (after `takas_farki` 2026-08-03,
`skor_cikis` 2026-08-29 and the weight change 2026-09-26). Taken deliberately:
the previous three were parameter choices, this one is a **measured sign error**
in production, and leaving the system awarding points for bonus issues until
October is worse than restarting the selection-vs-random clock again. The
managed-vs-passive comparison is unaffected, as always.

**Limits, stated plainly.** Only two types carry a measured direction, so the
layer speaks for a small slice of the flow and returns neutral for the rest —
including the biggest category, which the dictionary marks "read the content"
rather than guessing. That is the honest state: a narrow, verified signal
instead of a broad, invented one.

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

### Partial profit-take fired every day, not once (fixed 2026-08-18)

`PortfolioDoctor`'s partial profit-take had no memory of having fired. The condition was just `price >= target` against a **static** `target_price`, so while the price stayed above target it trimmed another 1/3 **every single day**. Live example (pid 4, TUPRS): weight went 33% → 22% → 15% → 10% → 7% → 4% across six trims while the stock rose 295 → 340. **The system sold its best position down to nothing precisely because it kept winning.** Systemic, not isolated: 47 repeated trims across 12+ portfolios (TUPRS, CRFSA, TKFEN).

Fix: `pm_positions.kar_alindi` flag (idempotent migration), checked in the condition and written by `_uygula` when the trim was a profit-take. `SAT` deletes the row and `AL` uses `INSERT OR REPLACE`, so a reopened position starts clean.

Note this behaviour was **never modelled in the backtest** — `_walk_managed` has no partial profit-taking. So the live system was doing something the simulation did not, and no amount of re-running the backtest would have found it. Live audit metrics and reading the trade log are what caught it.

### The drawdown brake looked like a trap and is NOT — measured, then reverted

pid 4 sat at 96% cash for 9 consecutive days, brake firing every day (NAV 18.9% below its 17 July peak, threshold 15%). The obvious reading is the documented `cummax` trap: cash earns 0%, so NAV freezes, so the drawdown never heals, so the brake never releases. The live `tail(90)` is 90 NAV **rows** (~4.5 months of weekdays) and a five-week-old portfolio has ~25 rows, so the window is effectively since-inception and the fix documented under the sweep section does nothing for young portfolios.

So a release condition was added — brake off once cash exceeds a threshold, on the argument that a mostly-cash portfolio has already de-risked and the brake is now only blocking recovery. **Measured on the 8-year panel, it is harmful:**

| | Real CAGR | MaxDD | Sharpe |
|---|---|---|---|
| Brake always on (current) | **+15.6%** | **−25.2%** | **0.71** |
| Released on cash threshold | +11.7% | −39.5% | 0.56 |

Identical at every threshold tested (30/40/50/60%), because once braked the cash level is already above all of them. Reverted in both engines.

The reasoning error: in the backtest, "braked and mostly cash" means *positions were stopped out during a decline*, and buying back into that is what costs the 14 points of drawdown. pid 4 reached 96% cash mainly through the profit-take bug above, which is a different cause with a different fix. **Fix the cause, not the symptom** — and this is the fifth time in this project that an intuitive fix was measured and rejected.

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

### US transfer test (2026-08-31) — `pipeline_backtest.py us` — out-of-sample in SPACE

Out-of-sample testing in *time* takes years. Testing in *space* does not. If the same construction logic works in a completely different market with parameters chosen on BIST, that is evidence the method generalises — available immediately, and it is the one kind of evidence that more BIST backtesting cannot produce (see the DSR section: additional trials make the BIST result weaker, not stronger).

**Rule observed strictly: no behavioural parameter was touched.** `skor_cikis`, `takas_farki`, `trailing_pct`, `dusus_freni`, `max_pozisyon`, `MAX_POS`, `sector_cap`, `corr_limit`, `score_min` all kept their BIST-chosen values. Tuning any of them would have destroyed the entire point. Only structural conversions were made — symbol suffix, benchmark `^GSPC`, cost 0.4% → 0.2% (US commission is half), volume floor 20M TL → $50M (each equal to its own market's universe threshold).

Panel: 593 US stocks, 2018-08-31 → 2026-08-31, fingerprint `fa8182eb025d`, 96/96 rebalances.

| | Total | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|
| **Unmanaged (selection only)** | **+1294.7%** | **+39.0%** | −35.9% | **1.06** |
| Managed (BIST rules applied) | +73.6% | +7.1% | −40.7% | 0.41 |
| S&P 500 | +164.3% | +12.9% | −33.9% | 0.72 |
| Random-6, 50 seeds | median NAV 3.91 | | 5–95%: [1.88, 8.35] | |

**The selection model transfers, and strongly.** Final NAV 13.95 against a random-6 5–95% band of [1.88, 8.35] — the **100th percentile**. It beats the S&P on both return (39.0% vs 12.9% CAGR) and risk-adjusted return (Sharpe 1.06 vs 0.72), using parameters selected in a different country, currency and market structure. Because the random-6 control experiences the same US bull market, the outperformance against *it* cannot be explained by the market being strong.

**The management layer does not transfer, and actively destroys value.** It cuts CAGR from 39.0% to 7.1%, and — unlike in BIST — it does not even deliver its supposed compensation: drawdown gets *worse* (−40.7% vs −35.9%). Its entire justification in BIST was halving drawdown in 6 of 6 sub-periods; in the US it fails on both axes at once.

The likely mechanism is that the management thresholds are volatility-scaled quantities I wrongly classed as "behavioural". A 15% trailing stop is a much tighter leash on a US large-cap than on a BIST small-cap, so the same number means something different in each market. That reading is consistent with the DSR concern applying *specifically* to the management parameters: those are the ones fitted to BIST, and they are the ones that fail to travel.

**What this changes:**

1. The strongest claim in the project is now the **selection** model, and it rests on two independent legs — above the 95th percentile of random draws in BIST *and* in the US, with the same parameters. That is real out-of-sample evidence, and it is what partially answers the deflated-Sharpe problem.

> **Withdrawn 2026-09-13 by the UK transfer test.** A third market, same parameters, same six-year window: the system finishes at the **22nd percentile** of random draws and loses money in absolute terms. The BIST leg had already collapsed to the 10th percentile once the data vault removed the reserved period. One market in three works, so "two independent legs" is no longer the right description of the evidence — see "Third market: the UK".
2. The **management layer's claim shrinks to BIST**. It must be presented as a market-specific risk overlay, not a general property of the method. Do not describe it as validated.
3. The US live cohorts currently run with `managed=1` and BIST thresholds. On this evidence that is the wrong configuration, but it is **not changed here** — a parameter decision needs its own walk-forward and editor approval, and this measurement is one window in one market.

Caveat: 2018–2026 was a strong US bull run, so absolute figures flatter any long-only momentum strategy. The random-6 comparison is the control that survives that objection; the S&P comparison does not, on its own.

### Backtest methodology audit against the published standard (2026-08-31)

Two references define the standard here: Bailey & López de Prado's **Deflated Sharpe Ratio** and the Arnott–Harvey–Markowitz **Backtesting Protocol**. Both make the same central point: a researcher tries hundreds of configurations and reports the best, so the reported Sharpe is inflated and must be discounted by the number of trials.

Where this project stands, item by item:

| Requirement | Status |
|---|---|
| Point-in-time data, no look-ahead | ✅ |
| Realistic costs incl. market impact | ✅ (2026-08-18) |
| Out-of-sample beyond a single split | ✅ walk-forward, 5 and 3 folds |
| Reproducibility | ✅ frozen panel + SHA fingerprint |
| Control groups | ✅ random-6, equal-weight BIST-30, index |
| Inflation adjustment | ✅ ENAG-real throughout |
| Failed variants reported, not just winners | ✅ this file |
| **Multiple-testing correction** | ⚠️ **implemented and FAILED — see below** |
| Survivorship-free universe | ⚠️ bounded, not eliminated |
| **Backtest tests the deployed system** | ❌ quantified below, and it is substantial |
| More than one market | ✅ US transfer test — selection transfers, management does not |

#### Deflated Sharpe Ratio — `pipeline_backtest.py dsr`

Roughly **695 parameter configurations** were evaluated across this project. The DSR asks: if the strategy had *no* edge, what is the highest Sharpe we would expect to see after that many trials?

| Trials N | Expected max Sharpe under the null | DSR |
|---|---|---|
| 25 (this run's grid) | 0.46 | 0.885 |
| 100 | 0.58 | 0.805 |
| 300 | 0.66 | 0.735 |
| **695 (whole project)** | **0.72** | **0.679** |

Uncorrected PSR against SR\*=0 is **0.9934** — the Sharpe looks decisively significant until the correction is applied. After it, **DSR = 0.68 against a 0.95 threshold. We do not clear the bar at any plausible trial count**, including the most generous one.

Daily inputs: SR 0.0557 (0.88 annualised), T=1994, skew ≈ 0, kurtosis 6.46 — the fat tails matter and the DSR penalises them.

**This does not say the strategy does not work. It says the backtest alone cannot establish that it does**, given how much searching was done. Three consequences:

1. **More sweeping makes the result weaker, not stronger.** Every new configuration raises N and lowers DSR. The search phase is over; further parameter hunting is now actively counterproductive.
2. **Independent evidence is the only remedy** — a different market (US transfer test) or live time. Both are out-of-sample in a way more backtesting cannot be.
3. **The number must be presented.** A reviewer asking "how many parameters did you try?" will apply this discount mentally; doing it first is the difference between rigour and salesmanship.

Mitigating argument, offered with caution because it is self-serving: DSR assumes the maximum was selected. Walk-forward *rejected* nearly every proposal and the defaults were kept in almost every case — only `takas_farki` and `skor_cikis` were ever changed. The effective N is therefore below 695. But even at N=25 the DSR is 0.885, still short of 0.95.

#### How much does "the backtest tests a different system" actually cost? (measured)

The simulation runs the technical core only; the live score is 45% technical + 25% news + 20% analyst upside + 10% valuation. Measured on 35 live stocks (top and mid band of the current scan), computing both scores:

| Measure | Value |
|---|---|
| Spearman rank correlation | **+0.761** |
| Pearson level correlation | +0.989 |
| **Overlap in the top 6 picks** | **3 of 6** |

Levels track almost perfectly because technical carries 45% of the weighted sum, but **ranking is what the strategy acts on**, and there the two disagree enough to change half the portfolio. So bias #1 is **not a footnote** — the backtest builds a portfolio roughly half of which the live system would not have chosen.

Component rank correlations with the composite: technical +0.761, analyst premium +0.115, sentiment **−0.010**, valuation −0.167. Sentiment being ~zero is consistent with the earlier GDELT finding (t<1) and with most stocks receiving a near-neutral 50 when article counts are low — it shifts levels without moving ranks.

#### Survivorship, bounded rather than eliminated

A factual correction first: of the 32 universe names missing from the panel, **most are recent IPOs, not delistings** — EKDMR has 70 bars, MCARD 118, SVGYO 121. They fail the 260-bar minimum because they are new. Only a handful (KOZAL, IPEKE, EMLAK, FINBN…) are genuine delistings. Earlier notes in this file called all 32 "delisted"; that was wrong.

The real bias is invisible by construction: companies liquid eight years ago that no longer exist never enter `get_scan_universe()` at all, and no historical constituent list is available from our sources. So it is **bounded by simulation** instead — positions are silently zeroed at various annual rates:

| Annual blow-up rate | Real CAGR | Cost |
|---|---|---|
| 0% (current) | +21.0% | — |
| 1% | +20.0% | −1.0 pt |
| 2% | +18.9% | −2.1 pts |
| 5% | +15.7% | −5.3 pts |

BIST delisting runs at roughly 1–2% of listed companies per year, so the defensible upper bound on this bias is about **2 points of real CAGR**. It does not reverse any conclusion, and it is now a number rather than a disclaimer.

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

> **Qualified hard by the data vault (2026-08-31).** This is the *five-year* window. On the six-year development window that remains after reserving 2024-08→2026-08, the same measurement puts the selection model at the **10th percentile** of random draws (final NAV 13.20, random median 27.46), and the managed track at the 62nd. The claim above is not withdrawn — it is measured — but it is window-dependent, and the window that carries it is largely the one now locked away. See "The data vault" section.

Honest caveat to state alongside it: the system also takes **more drawdown than the index** (−49.1% vs −22.9%). It wins on return, not on smoothness. That is precisely the gap the management layer was built to close, and per the table above it closes it (−25.8%) by giving back real return.

**Known biases, ranked.** Two are serious:

1. **The backtest tests a DIFFERENT system than the live one.** It uses the technical score only (`BacktestEngine._vectorized_scores`); the live score is 45% technical + 25% news + 20% analyst upside + 10% valuation. News and analyst data cannot be reconstructed historically, so these results describe the **technical core**, not the deployed system. State this explicitly.
2. **Parameter selection was in-sample** — `sweep` finds the best value on the same data it measures. Addressed by `validate` below.
3. **Survivorship / universe selection** — the universe is *today's* liquid list, so companies that delisted or shrank are absent (optimistic). Partly mitigated by applying each date's own volume filter.
4. **Bull-market sample** — the 5-year window is entirely a BIST upswing, where insurance (stops, brakes) can only look like cost. Addressed by using an 8-year window in `validate`, which contains the 2018 TL crisis (−28% drawdown) and the 2020 Covid crash (−32%).
5. ~~**No slippage model**~~ — CLOSED 2026-08-18, see "Market impact / scale capacity". Capacity is a few million TL; all other figures in this file assume zero impact.
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

### Portfolios converge and turn over too fast — `skor_cikis` 43 → 39 (2026-08-29)

Symptom reported from the live system: "portfolios have started to move identically". Measured, and it is real — but the accompanying guess ("they stopped buying") was the opposite of the truth. The system trades heavily: 26–98 trades/day across the managed set, roughly one trade per portfolio every two days.

What is actually wrong is **turnover and convergence**:

| Measure | Value |
|---|---|
| Managed BIST portfolios | 35 |
| Unique stocks across all of them | **49** (of 210 slots) |
| INVES / KUVVA / DEVA / POLHO | in 43% / 40% / 37% / 37% of portfolios |
| Average holding period | **10.3 days** |
| Sell reasons, 14 days | score exit 96, swap 77, stop 49 |
| Cost burden | **4.2%** of gross movement (was 1.4%) |

Two things follow. The convergence is a direct consequence of `takas_farki=6`: a low swap threshold pushes every portfolio toward the same top-ranked names, so a set of portfolios that looks diversified across horizons and profiles is closer to one bet. **The backtest cannot see this** — it simulates a single portfolio, so cross-portfolio convergence is invisible to it. Worth stating in the presentation as a limitation of the simulation, not just of the system.

The turnover is worse. The calibration puts the edge at **30–90 days**, and the live system was holding for **10**. Score exits were the largest driver, so `skor_cikis` — never swept until now — was the suspect.

Measured on the 8-year panel, the curve is single-peaked and smooth, which is what a real effect looks like:

| `skor_cikis` | Real CAGR | MaxDD | Sharpe | With 5M TL portfolio |
|---|---|---|---|---|
| 30 | +18.8% | −31.0% | 0.81 | +17.0% |
| 35 | +19.9% | −26.3% | 0.84 | +13.7% |
| **39 (new)** | **+21.0%** | −25.8% | **0.88** | +14.8% |
| **43 (old)** | +15.6% | −25.2% | 0.71 | +10.3% |
| 47 | +5.7% | −27.1% | 0.35 | +1.6% |
| 52 | +2.6% | −24.0% | 0.23 | −2.1% |

MaxDD is flat across 35–43, so the extra return is not bought with risk, and the advantage survives market impact. Walk-forward: training picked 39 in **5 of 5** folds and it held in 3, with large wins (+37.9%, +43.7%, +36.2%) and small losses (−0.8%, −7.0%).

**Contrast with `min_tl_vol`, which failed the same review a week earlier.** Both showed 5/5 training picks and 3/5 folds held. The difference is the shape of the wider curve: `min_tl_vol` zig-zagged (18.3 → 14.0 → 17.8 → 15.6) and was correctly read as noise; `skor_cikis` rises to a single peak and falls away cleanly. **When the walk-forward margin is thin, the shape of the full curve is the tie-breaker** — that is the transferable lesson.

Cost for the live experiment: the managed shadow cohorts now carry a **second** before/after break (first `takas_farki` on 2026-08-03, now this). The live track is therefore three configurations deep and correspondingly weaker as evidence. Accepted deliberately: the live leg is already short and was contaminated by the robot outage, while leaving a measurably bad exit threshold in production for six more weeks has a certain cost.

### System vs random, period by period (2026-08-18) — `sanstest`

**This qualifies the strongest claim in the project and must be presented alongside it.** The headline was "the system's final NAV sits above the 95th percentile of random 6-stock portfolios". That is true over the full window. Broken into six sub-periods it is far less flattering:

| Period | System | Random median | Random 95th | System percentile |
|---|---|---|---|---|
| 2018-08→2019-12 | **−14.6%** | +78.3% | +156.6% | **0%** |
| 2019-12→2021-04 | +143.1% | +179.4% | +504.6% | 40% |
| 2021-04→2022-08 | +166.0% | +59.7% | +131.7% | **98%** |
| 2022-08→2023-12 | +92.4% | +189.8% | +318.6% | **8%** |
| 2023-12→2025-04 | +106.0% | +26.6% | +103.4% | **94%** |
| 2025-04→2026-08 | +636.0% | +52.2% | +142.6% | **100%** |

**Above the 95th percentile in only 3 of 6 periods, and in the bottom decile in two.** In 2018-08→2019-12 — the TL crisis — the system *lost 14.6% while the median random portfolio made 78.3%*, the worst possible ranking.

The full-window result is not wrong: compounding is what an investor actually experiences, and a strategy that wins big in three periods and loses in two can still finish far ahead of every random draw. But **"consistently beats random" is not supported.** The edge is episodic and regime-dependent, and it fails badly in exactly the reversal environments where momentum is known to crash.

Present it as: *over the full window the system finishes above the 95th percentile of random draws; period by period it is above the 95th percentile in half the sub-periods and in the bottom decile in two, including the 2018 crisis.* That is the honest version, and it is far more defensible than the headline alone.

Method caveat printed by the command itself: the periods are slices of one continuous run (not independent) and the random portfolios reuse the same 50 seeds across periods, so this is a consistency indicator, not a significance test.

### Construction parameters (2026-08-18) — `pipeline_backtest.py kurulum [yil]`

`score_min`, `sector_cap`, `corr_limit` and `min_tl_vol` had never been tested. `_pick_at` now accepts them via `kur=`. Walk-forward gate, same as elsewhere (majority of folds, >5%):

| Parameter | Default | Training's picks | Folds held | Verdict |
|---|---|---|---|---|
| `score_min` | 50 | [40, 45, 40, 45, 45] | 0/5 | no change |
| `sector_cap` | 2 | [1, 1, 1, 3, 3] | 2/5 | no change |
| `corr_limit` | 0.85 | [0.70 ×5] | 0/5 | no change |
| **`min_tl_vol`** | **20M** | **[10M ×5]** | **3/5** | **change proposed** |

Three defaults confirmed. `sector_cap=2` is notable: it beats both 1 (+11.5%) and 3 (+14.7%) on the full window, so the diversification limit is genuinely at a sweet spot rather than being arbitrary. `corr_limit` barely binds — 0.95 and 1.01 (filter off) give identical results, so the correlation filter is close to inert at its current setting.

**`min_tl_vol` 20M → 10M passed the strict gate — and was then WITHDRAWN when measured across a wider range. See the addendum below.** Full window: +17.8% vs +15.6% real CAGR, MaxDD −22.1% vs −25.2%, Sharpe 0.78 vs 0.71 — better on all three axes, and training chose it in 5 of 5 folds, which is far more stable than the wild swings seen for `takas_farki` or `max_pozisyon`.

The obvious objection is that a lower liquidity floor means thinner names and therefore worse market impact. **Tested directly** against the slippage model, and the objection does not hold — 10M wins at every portfolio size:

| Portfolio | 10M floor | 20M floor | 50M floor |
|---|---|---|---|
| no impact | **+17.8%** | +15.6% | +6.6% |
| 1M TL | **+14.4%** | +13.1% | +5.1% |
| 5M TL | **+11.0%** | +10.3% | +3.4% |
| 25M TL | **+5.6%** | +2.6% | 0.0% |

Note also `min_tl_vol=100M` collapses to −6.4% real CAGR. The edge really does live in less-liquid names — consistent with the US finding that it appears where analyst coverage thins, and it explains the capacity limit measured above. Awaiting editor approval; `MIN_TL_VOL` is the backtest's own constant, the live equivalent is the scanner's volume gate.

#### Addendum — the `min_tl_vol` proposal dissolved under a wider sweep, and NOT applied

The sweep above tested 10/20/50/100M. Extending it downward to include the value the **live** system actually uses exposes the problem:

| Threshold | Real CAGR | MaxDD | Sharpe | With 5M TL portfolio |
|---|---|---|---|---|
| **1M (live scanner)** | +18.3% | −26.8% | 0.81 | +10.1% |
| 5M | +14.0% | −25.7% | 0.65 | +7.5% |
| **10M** (proposed) | +17.8% | −22.1% | 0.78 | +11.0% |
| **20M (backtest default)** | +15.6% | −25.2% | 0.71 | +10.3% |
| 50M | +6.6% | −28.9% | 0.38 | +3.4% |

**The curve is not monotonic** — 18.3 → 14.0 → 17.8 → 15.6 across adjacent thresholds. A real effect would be smooth; this pattern says the differences across 1M–20M are noise, and the 3/5 walk-forward majority was a thin one. What survives is only the high end: **50M+ is clearly bad**, which is the same "the edge lives in less-liquid names" result seen elsewhere.

**Not applied.** There is no measurable basis for preferring 10M over 20M.

**But the sweep found something more important on the way.** The backtest and the live system use *different* liquidity filters:

| | Threshold | Window |
|---|---|---|
| Backtest `MIN_TL_VOL` | **20M TL** | 30-day rolling |
| Live `volume_ok` ([bist_analyzer.py:2525](bist_analyzer.py:2525)) | **1M TL** | 20-day rolling |

The live system is **twenty times more permissive** than the backtest. This is a fresh instance of known bias #1 ("the backtest tests a different system than the live one") and it had not been noticed before. It is disclosed rather than fixed: aligning the constants would invalidate every frozen result in this file for a difference the table above shows to be within noise. State it in the presentation; do not quietly change it.

### Market impact / scale capacity (2026-08-18) — `pipeline_backtest.py slippage [yil]`

Bias #5 on the list ("no slippage model") is now closed. `COST_RT` is a flat percentage that ignores trade *size*; in reality buying pushes the price up and selling pushes it down, and the effect grows with the trade's share of daily volume. The portfolio holds genuinely thin names (OZATD, OZSUB, ODINE), so this was never a theoretical concern.

Model: square-root law, `impact = SLIP_K × sqrt(trade / ADV)` with `SLIP_K = 0.02` — 1% of ADV costs 0.20%, 10% costs 0.63%, 50% costs 1.41%. ADV comes from each stock's own `tlvol` series at that date, so it is point-in-time. `portfoy_tl=0` disables it and reproduces the old numbers exactly, so nothing already published moved.

**How much capacity does the edge have?** (8-year panel, managed track)

| Portfolio size | Unmanaged real CAGR | Managed real CAGR | Sharpe | Cost vs frictionless |
|---|---|---|---|---|
| (no impact) | +7.4% | **+15.6%** | 0.71 | — |
| 100k TL | +6.7% | +14.8% | 0.68 | −0.8 pts |
| 500k TL | +5.9% | +13.8% | 0.64 | −1.8 pts |
| 1M TL | +5.3% | +13.1% | 0.62 | −2.6 pts |
| 5M TL | +2.9% | +10.3% | 0.52 | −5.3 pts |
| 10M TL | +1.3% | +8.5% | 0.45 | −7.1 pts |
| 50M TL | −3.6% | **+0.8%** | 0.16 | −14.9 pts |

**The strategy has a capacity of a few million TL.** Below ~1M TL the edge is essentially intact; by 10M TL roughly half of it has gone to market impact; by 50M TL it is gone. This is a property of a concentrated small/mid-cap momentum strategy and is not a defect — but it must be stated, because **every figure previously reported in this file assumes zero market impact, i.e. an infinitely small portfolio.**

Caveat: `SLIP_K` is an order-of-magnitude estimate from the conservative end of the published range, not a calibrated constant for BIST. The shape of the decay is more trustworthy than any single row.

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

### The data vault (2026-08-31) — the last two years are locked, and locking them changed the picture

Step 1 of the agreed plan, and it had to come first: every result in this file
was computed on a window whose last two years had already been used by every
decision, so there was nothing left to check anything against. The reserve is
now **2024-08-31 → today**, and no command in `pipeline_backtest.py` or
`weight_calibration.py` can see past that date.

**Mechanism, and why it is built this way.** The cutoff lives in
`veri_kasasi.json` (tracked in git — one date, one source of truth). The lock
is the **default**, not a flag: every command takes its window from the single
gate `_endeks_penceresi()`, and the price panel is trimmed a second time on
load by `_kasa_panel()`. Two layers because one missed code path is all it
takes — the same trap as the robot's market isolation, where a restriction was
added in one place and silently skipped the weekly cohort loop. A trimmed panel
prints its **own** fingerprint (`45f4f3dc809a` against the full panel's
`47a4ec949c92`), so a vault-era number can never be mistaken for a pre-vault one.

The calibration side is locked too, and there the subtlety matters: a panel row
dated 2024-08-01 carries `fwd_90d`, computed from prices running into November
— its *date* is outside the vault but its *content* is not. `_panel_oku()`
therefore cuts rows at `cutoff − horizon`, not at the cutoff. Locking only the
backtest would have been theatre, because weights and the scoring style come
from these ICs.

**Window semantics:** `8` means **2018-08-31 → 2024-08-31 (6.0 years)** — the
same start as the published 8-year results, minus the reserve. Every number this
repo produces from now on therefore differs from the ones above; compare against
the new baseline below, never against a pre-vault figure.

> **Both ends are anchored to the cutoff (fixed 2026-09-13).** As first written,
> the vault pinned only the *end*; the start was still counted back from *today*,
> so the window slid forward one day per day. Caught two weeks later by re-running
> `us 8`: on 2026-08-31 it covered 2018-08-31→2024-08-29 (72 rebalances, final NAV
> 4.55), on 2026-09-13 it covered 2018-09-11→2024-08-30 (71 rebalances, NAV 4.58).
> A number written into this file was already unreproducible a fortnight later,
> which is exactly what the frozen panel exists to prevent — the anchor had a hole
> in it. The rule is now `start = cutoff − (years − rezerv_yil)`, so `8` gives
> 2018-08-31 → 2024-08-31 on any calendar date. Verified after the change: `run 8`
> returns NAV 13.20 at the 10th percentile and `us 8` returns NAV 4.55 with 47/50
> seeds — both identical to the figures recorded here before it.

#### The baseline on the development window — this is uncomfortable and it is the point

`karsilastir 8` and `run 8` under the lock, 448 stocks, 72/72 rebalances,
2018-08-31 → 2024-08-29:

| | Nominal | ENAG-real | **Real CAGR** | MaxDD | Sharpe |
|---|---|---|---|---|---|
| System, unmanaged (selection only) | +1220% | −61.5% | **−14.7%** | −69.9% | −0.09 |
| System, managed | +3175% | −4.5% | **−0.8%** | −25.8% | 0.10 |
| Equal-weight BIST-30 | +1694% | −47.7% | −10.2% | −32.8% | −0.23 |
| Random-6 (50 seeds, mean) | +3468% | +4.1% | +0.7% | −39.3% | 0.16 |
| XU100 | +960% | −69.1% | −17.8% | −31.8% | −0.59 |

And the control comparison that the whole thesis rests on, measured exactly
rather than as "inside the band" (final NAV against the 50 random seeds):

| | Final NAV | Percentile | Seeds beaten |
|---|---|---|---|
| **Selection only (unmanaged)** | 13.20 | **10th** | 5 / 50 |
| **Managed** | 32.75 | 62nd | 31 / 50 |
| Random-6 median | 27.46 | — | — |

**Strip the reserved two years and the selection model is in the bottom decile
of random 6-stock portfolios.** The published headline — "the system's final
NAV sits above the 95th percentile of random draws" — is a *five-year*
(2021-08 → 2026-08) result, and the vault window is a different six years.

The decomposition is not new; it is already in the `sanstest` table above,
which reported the system's per-period percentile as 0%, 40%, 98%, 8%, 94%,
100%. The five-year window happens to **exclude both weak periods** (2018-08 →
2021-04) and **include both strongest ones** (2023-12 → 2026-08, now mostly
reserved). What is new is seeing it compounded: on the BIST data we are still
allowed to develop on, selection does not beat luck. (In the US over the
same six years it does — see below, and that difference is the finding.)

Note also which leg survives. On this window the **management layer** is the
one carrying the result (62nd percentile vs 10th, real CAGR −0.8% vs −14.7%,
MaxDD −25.8% vs −69.9%). That is the reverse of the reading taken after the US
transfer test, where selection was the strong general leg and management the
market-specific overlay. Both readings are window-dependent, which is itself
the finding: **neither leg is robust enough for an unconditional claim.**

`dsr 8` under the lock says the same thing without ambiguity: annualised
Sharpe falls to **0.10** (it was 0.88 on the full window) and DSR is **0.03–0.14**
across every trial count. On the development window there is no statistically
defensible edge at all.

#### The US leg survives the lock, and that is what saves the selection claim

`us 8` under the same lock (585 US stocks, 72/72 rebalances, 2018-08-31 →
2024-08-30, vault fingerprint `d24dd47f7e48`):

| | Total | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|
| **Unmanaged (selection only)** | **+355.1%** | **+28.7%** | −32.0% | **0.95** |
| Managed (BIST rules) | −0.5% | −0.1% | −40.7% | 0.10 |
| S&P 500 | +94.7% | +11.7% | −33.9% | 0.65 |
| Random-6, 50 seeds | median NAV 2.38 | | 5–95%: [1.40, 4.43] | |

Final NAV 4.55 — above the 95th-percentile value (4.43), beating 47 of 50
random draws. **The US result does not depend on the reserved period at all.**
Same parameters, chosen on BIST, on six years that end where the vault begins:
still beats the index on return and on Sharpe, still beats random.

So the two markets now say different things about the same model on the same
six years — **10th percentile in BIST, 94th in the US** — and the management
layer inverts too: it rescues BIST (62nd vs 10th) and destroys the US result
(+28.7% → −0.1% CAGR), which is the market-specific pattern the earlier US
transfer test already found.

The defensible summary going into October: *the selection method has
out-of-sample support in the US on data that took no part in tuning; its BIST
evidence is concentrated in periods that are either early-window strong (2021-04
→ 2022-08) or inside the reserve, and over the six-year development window BIST
selection does not beat random.* Both halves have to be said.

> **Qualified again 2026-09-13:** a third market (UK) fails as well — 22nd
> percentile, −32.3% absolute. With one market working out of three, the US is
> better described as the outlier than as confirmation. See "Third market: the UK". Quoting only the
US half is salesmanship; quoting only the BIST half throws away the one leg that
survived every test applied to it so far.

#### What the vault does and does not prove

It must not be oversold, because the reserve is **not unseen data**. Every
8-year table in this file already covered it, `sanstest` reported it
period-by-period, and two parameter decisions were taken on windows containing
it — `takas_farki` 12→6 (2026-08-03) and `skor_cikis` 43→39 (2026-08-29). So
the October run cannot be described as "a test on data never looked at". The
defensible claim is narrower and still worth having: **no decision taken after
2026-08-31 can have been fitted to it.**

A blunt consequence: the vault as specified **cannot produce a surprise**. We
already know from published runs that the system did well in 2024-2026, so
opening it in October will confirm what we know, and a confirmation you can
predict is weak evidence. The genuinely uncontaminated evidence remains what it
was — the live shadow cohorts running forward in time, and transfer to markets
the parameters never saw.

The temptation now is obvious and must be named: the development window looks
bad, so re-tune until it looks good. That is precisely the overfitting the
vault exists to prevent, and it would also burn the reserve, since parameters
fitted post-cutoff make the October number meaningless. **The standing rule
applies with more force, not less** — a change needs a stated hypothesis
tested once, not a sweep for whatever rescues 2018-2024.

### Third market: the UK (2026-09-13) — `pipeline_backtest.py uk` — the method does NOT transfer

Step 3 of the agreed plan, and it is the most consequential result in this file
since the vault. The US transfer test was read as evidence that the selection
method generalises. A second transfer test says it does not.

**Why the UK and not Germany.** Germany was measured first and rejected on
breadth: TradingView lists 429 German names in total, of which only **20** clear
€50M/day, 59 clear €10M and 133 clear €1M. A universe that thin cannot support a
6-position portfolio with a sector cap plus a 50-seed random control — the
control would be drawing from nearly the same names as the system. The UK lists
1081, of which **298** clear £1M/day, which is comparable breadth to BIST (448)
and the US (585). Breadth is what makes the random control meaningful, so the UK
is the fairer test. (Recorded so nobody re-derives it: Germany is not a bad
market, it is a *narrow* one, and the narrowness is the disqualifier.)

**No behavioural parameter was touched**, same rule as the US test. What changed
is structural only, and all of it lives visibly in one table (`TRANSFER_PIYASA`):
symbol suffix `.L`, benchmark `^FTSE`, liquidity floor £1M/day, cost 0.6%.

Two traps were found while building it, both of which would have produced a
confident wrong number:

1. **Sector data does not exist outside BIST and the US.** `_sector_of` knows
   those two markets only, so every UK stock would have returned "Diğer" — one
   single sector — and `sector_cap=2` would have silently capped every portfolio
   at **2 positions instead of 6**. The run would have completed, printed a
   plausible table, and measured a completely different strategy. Sectors are now
   frozen into the universe file alongside the tickers (20 distinct sectors for
   the UK), and `_pick_at` takes an explicit map via `kur={"sektor": ...}`.
2. **`us_transfer` was not copy-pasted.** It is now a thin wrapper over a
   parameterised `_transfer(piyasa, years)`. Duplicating 120 lines to change a
   suffix is the exact error class this file keeps recording. Verified by
   re-running `us 8` after the refactor: byte-identical to the pre-refactor
   numbers (+355.1%, NAV 4.55, 47/50 seeds).

**Cost is set deliberately against us:** 0.6% round-trip versus BIST's 0.4% and
the US's 0.2%, because UK stamp duty is 0.5% on purchases. Rounding a structural
assumption the *unfavourable* way is the safe direction — and, as the sensitivity
table below shows, it turned out not to matter.

#### Result — 292 stocks, 72/72 rebalances, 2018-08-31 → 2024-08-30, vault fingerprint `caca1fd8abc6`

| | Total | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|
| **Unmanaged (selection only)** | **−32.3%** | **−6.3%** | −47.3% | −0.18 |
| Managed (BIST rules) | −20.7% | −3.8% | −28.0% | −0.22 |
| FTSE 100 | +12.7% | +2.0% | −35.0% | +0.20 |
| Random-6, 50 seeds | median NAV 0.89 | | 5–95%: [0.55, 1.38] | |

System final NAV **0.68** against a random median of 0.89 — the **22nd
percentile**, beating 11 of 50 random draws. It loses money, loses to the index,
and loses to luck.

Note what the random control establishes. UK small and mid caps fell over this
window (random median 0.89) while the FTSE 100 rose 12.7%, so part of the loss is
simply the segment the strategy lives in. But the system finished **below** the
random median *drawn from that same segment*, so this is not only segment
exposure — the selection itself subtracted value.

#### It is not the cost assumption — checked, because the assumption was mine

Same window, same picks, only `COST_RT` varied:

| Round-trip cost | System NAV | Random median | System percentile |
|---|---|---|---|
| 0.6% (used above) | 0.68 | 0.89 | 22nd |
| 0.4% (BIST's) | 0.77 | 1.03 | 20th |
| 0.2% (the US test's) | 0.88 | 1.18 | 20th |
| **0.0% (frictionless)** | **1.00** | **1.36** | **20th** |

**Even with zero transaction costs the system is at the 20th percentile and still
below the random median.** The percentile is flat across the whole range, and for
a mechanical reason worth understanding: random-6 redraws six names every month,
so it turns over far more than the system does and is *hurt more* by costs. A
lower cost assumption therefore helps the control more than it helps us. My
conservative 0.6% was, if anything, flattering the system.

#### What this does to the "the method generalises" claim

Three markets, same parameters, same six-year development window, same control:

| Market | Selection-only percentile vs random-6 | Verdict |
|---|---|---|
| BIST | 10th | fails |
| **US** | **94th** | **works** |
| UK | 22nd | fails |

One win out of three. The honest reading is no longer "the selection method
transfers"; it is that **the US result is the outlier and needs its own
explanation.**

The explanation most consistent with all the evidence is that the system is a
**long-only momentum vehicle**, and its results track whether momentum paid in
that market over that window rather than any market-independent selection skill:

- US 2018–2024: momentum paid exceptionally (mega-cap tech). System wins big.
- UK 2018–2024: a value- and dividend-weighted market where momentum was weak.
  System loses.
- BIST 2018–2024: system loses to random.
- BIST 2024–2026 (reserved, known from pre-vault runs): a momentum melt-up.
  System wins big.

Every result lines up with "did momentum work here, then?" — which is a factor
exposure, not an edge. This is consistent with the existing calibration finding
that 52-week position is the single strongest signal in the scoring, and with the
regime-conditional US result (momentum IC +0.056 in Boğa, **−0.088 in Ayı**).

Stated carefully, because three markets is still a small sample: this does not
prove the system has no selection skill. It does mean **the burden of proof has
moved.** A claim of generalisation now requires showing the system beats a
momentum benchmark, not just an index and a random draw — a long-only momentum
factor portfolio is the control that is actually missing, and it is the natural
next measurement.

#### Benchmark dividend bias — found on the way, affects every "vs index" number in this file

Stock panels are downloaded with `auto_adjust=True`, which makes them **total
return** series (dividends reinvested). The benchmarks — `^GSPC`, `^FTSE`,
`XU100.IS` — are **price** indices, which exclude dividends. So every
index comparison in this file has been comparing a total-return portfolio against
a price-return benchmark, and the gap always favours the system.

Measured exactly for the US over the vault window, using `^SP500TR`:

| | Total | CAGR |
|---|---|---|
| S&P 500 price (`^GSPC`, used everywhere in this file) | +94.7% | +11.7% |
| S&P 500 total return (`^SP500TR`) | +115.4% | **+13.6%** |

**+1.9 percentage points per year of unearned advantage.** The UK is worse — the
FTSE 100 yields more than the S&P, so the understatement there is roughly 3.5–4
points a year, which makes the UK result above *even worse* than the table shows.
For BIST the yield is around 2–3 points.

No conclusion in this file reverses: the US win is far larger than 1.9 points and
the UK/BIST losses only deepen. It is **disclosed rather than fixed**, for the
same reason as the `min_tl_vol` mismatch — a total-return benchmark is not
available on our data source for BIST or the UK, so switching only the US would
make the three markets mutually incomparable while invalidating every published
table. State it in the presentation; a reviewer will ask.

## WHERE WE ARE AND WHAT IS NEXT (handoff, 2026-09-27)

Read this first in a new session. It supersedes the ordering in "Agreed next
steps" below, which is now partly done.

### What exists after the KAP work

| Asset | What it is |
|---|---|
| `kap_veri/` (gitignored) | **411,656 KAP disclosures**, 2018-01 → 2024-08, 1,031 stocks, 192 types, 90% with a summary. Rebuild with `tools/kap_gecmis.py`, ~35 min |
| `kap_sozluk.json` (tracked) | **114 disclosure types**, 98% of the archive, each with class (rutin / yonlu / icerik / sayisal) and an `olcum` block carrying n, median, pos%, t |
| `tools/kap_olay.py` | Event study — abnormal return by type, day-clustered t, vault-aware |
| `tools/kap_ayrim.py` | Splits one type into subgroups by summary text and measures each |
| `tools/kap_metin.py` | Fetches disclosure bodies (`oda_ExplanationTextBlock`), verified working |

### What is actually wired into the live score

Two measured signals, both live in `news_engine.py` as of 2026-09-27:

| Signal | n | 30d median | t |
|---|---|---|---|
| Payların Geri Alınması (buyback) | 5,158 | **+2.63%** | +4.6 |
| Bedelsiz sermaye artırımı (bonus issue) | 1,160 | **−3.86%** | −5.0 |

The dictionary is authoritative for KAP items (`_kap_turu` / `_kap_yon`, consulted
at the top of `_classify` and inside `_bilgisiz_baslik`), repeated notices of the
same type collapse to one event, and the 429-phrase weighted bank — which had
never executed in production — now runs with word-boundary matching.

### The open problem list

**Steering the system wrong**
1.1 **Price echo is counted twice.** "THY 300 TL'nin altına geriledi" classifies
    negative, but that is the price move the technical score already sees —
    `_is_technical_price_move` does not catch this phrasing. It creates
    correlation between two components that are supposed to be independent.
1.2 Live cohorts now carry **four config breaks**; the selection-vs-random clock
    keeps restarting. Managed-vs-passive is unaffected.
1.3 **35 portfolios hold 49 unique stocks** — a set that looks diversified is
    close to one bet, and the single-portfolio backtest cannot see it.
1.4 **Holding period 10 days against a measured 30–90 day edge**; cost burden
    went 1.4% → 4.2%.
1.5 **The relative-strength bonus never fires in production** (found
    2026-09-27). `compute_bist_score` adds +5/+10 to the technical score when a
    stock beats XU100 over 14 days, but `yf.download` now returns MultiIndex
    columns, so `xu_df["Close"]` is a DataFrame, `float(...)` raises under
    pandas 3, and the `except` swallows it — `relative_strength` is 0.0 for
    every stock (30/30 in the test log). **Not fixed on purpose:** fixing it
    changes live scores (a fifth cohort break), and the bonus was never in the
    backtest or the calibration, so "fixing" it would add an untested term and
    widen bias 3.5. The honest options are delete it or test it first; the
    user decides.

**Unmeasured — we do not know whether they work**
2.1 206 words + 429 phrases + 76 regex patterns have **never** been validated
    against returns, and cannot be historically (Google serves no news archive).
2.2 Only **2 of 114** KAP types carry a measured direction; 29% of the weekly
    flow gets an explicit "don't know".
2.3 ~~Component scores are not durably logged~~ — **DONE 2026-09-27**,
    `skor_gunlugu.py` → Turso `skor_gunlugu`, daily, whole BIST universe. The
    per-component IC becomes computable after a few months of rows.
2.4 Analyst premium (20% of the score) and valuation (10%) have never been
    IC-tested at all.
2.5 Credit ratings: 3,503 events, bodies fetchable, but 75% undetermined by
    keyword — needs the rating symbol parsed and compared to the company's
    previous rating.
2.6 The weighted bank's ±1 threshold means a single weight-1 term can never
    decide; it always returns neutral.

**Evidence**
3.1 Selection fails in 2 of 3 markets on the development window — BIST 10th
    percentile, UK 22nd, US 94th.
3.2 DSR 0.03–0.14 on the development window: no statistically defensible edge.
3.3 **The momentum benchmark has still not been run.** Every result across four
    market-periods lines up with "did momentum pay here, then?", and a
    long-only momentum portfolio is the control that would settle it.
3.4 Benchmark dividend bias: index comparisons flatter us by ~1.9 pts/yr in the
    US, more in the UK and BIST.
3.5 The backtest tests the technical core only; top-6 overlap with the live
    composite is 3 of 6.
3.6 The live liquidity filter is 20× more permissive than the backtest's.

**Accepted limits**
4.1 Capacity is a few million TL. 4.2 Survivorship bounded at ~2 pts real CAGR.
4.3 No Turkish financial sentiment model or labelled dataset exists.
4.4 KAP rate-limits; pull politely and never run two collectors at once.
4.5 Zaman Makinesi still simulates the old portfolio logic.
4.6 Actions disables cron after 60 days of repository inactivity.

### 2.3 DONE (2026-09-27) — `skor_gunlugu.py`, durable component logging

**The handoff's premise was wrong in one place, and it shaped the fix.** It said
"the robot scans ~500 stocks daily and records none of the component
breakdown". The robot does not compute the composite at all:
`PortfolioScanner.scan_all` produces the **technical** score only, and it runs
weekly (when a shadow cohort is missing), not daily. `compute_bist_score` — the
only place sentiment, premium and valuation exist — was called solely by the
Hisse Analizi page. So there was nothing to "also write to Turso"; the
composite had to be computed.

**What it does.** A separate Actions step after the robot (`python
skor_gunlugu.py yaz`) runs `compute_bist_score` — the live function itself, not
a copy — over the whole BIST scan universe (~480 names) and writes one row per
stock per day to Turso table `skor_gunlugu`, PK `(tarih, market, ticker)`.
Each row carries the four component scores, `SCORING_VERSION`, the **effective
weights actually applied that day** (`BISTScore._agirliklar`, exported from
`compute_bist_score` — deriving them later from flags would silently break the
moment `WEIGHTS` changes), the inputs behind them (news count / pos / neg /
content-free, analyst target and analyst count, F/K, PD/DD), the price with its
bar date (holidays and stale data are distinguishable), and `kaynaklar`, the
news sources that returned items. `python skor_gunlugu.py ozet` prints coverage
per day.

Forward returns are **not** joined from `accuracy_validation` as the handoff
proposed — that table is local SQLite too and dies with the container. The
analysis step (a few months from now) computes forward returns from price
history against `tarih`; the logged `fiyat` is a sanity check.

**Measured cost.** 8 threads, ~1.6 s per stock → ~13 min per day. 16 threads
gave ~1.2 s but news sources started returning empty, i.e. throttling. The step
has its own 28-minute limit and the job limit went 25 → 55, so management never
shares a budget with it; `if: !cancelled()` so a red robot step does not also
lose the day's log. Actions minutes: ~+300/month, total ~520 of 2000.

**Scope is BIST only, deliberately.** Analyst targets are auto-fetched for BIST
only (US premium would always be missing), the news engine is Turkish-sourced,
and the US universe (~1400) would triple the runtime.

**Exit code tells the truth**, per the 2026-07-31 incident: red if fewer than
80% of the universe was scored or if >90% of rows have `haber_n=0` (news
sources throttling the CI IP would otherwise produce a confident, empty
sentiment log). A day with **no KAP data at all** prints a loud warning but
stays green — KAP via borsapy is a known-flaky external source, like ENAG.

**Why the KAP column exists — observed while building this.** borsapy's KAP
route failed for every ticker for several minutes ("Server disconnected",
0.2 s), while KAP's own `byCriteria` API answered normally; minutes later it was
back. During the outage TUPRS's sentiment read 57.1, afterwards 51.6. The two
measured KAP signals (buyback, bonus issue) have **no input** during such a
window and nothing else would show it, so sentiment rows are only interpretable
alongside `kaynaklar`.

Also found: see 1.5 below (relative-strength bonus is dead in production).

**After 2.3, in order:** 3.3 the momentum benchmark (one day, settles the
biggest open question), then 1.4 turnover (measured problem, free gain), then
1.1 the price-echo double count.

The user has also flagged wanting to move into **fundamental/financial-statement
analysis** afterwards — `borsapy` exposes `income_stmt`, `balance_sheet`,
`cashflow` and their quarterly variants, so earnings growth is computable
numerically with no NLP at all. That is the first genuinely new data source
since the project started; everything else has been a re-slice of price.

### Agreed next steps (2026-08-31, carried into a new session)

The deflated-Sharpe result changed the plan. Testing more configurations on the same
8-year BIST panel now *lowers* confidence rather than raising it, so the work shifts
from searching to getting more independent evidence out of what we have. Three tasks,
in this order:

**1. Lock a data vault — DONE 2026-08-31.** Cutoff 2024-08-31, enforced by
default in both `pipeline_backtest.py` and `weight_calibration.py`; open it once
with `kasa-ac 8 EVET`. Full write-up, the new development-window baseline, and the
limits of what the vault can prove are in "The data vault" section above — read that
before running anything, because every command now returns different numbers than
the tables in this file.

**2. Combinatorial cross-validation (CPCV).**
The current design splits the window one way (train/test) or into expanding folds.
CPCV generates many train/test block combinations from the same data and returns a
*distribution* of out-of-sample results instead of one number, and it yields the
Probability of Backtest Overfitting directly. Better use of data already held, no
waiting. This is the natural companion to `dsr`.

**3. A third market — DONE 2026-09-13, and it came back negative.**
The UK was chosen over Germany on universe breadth. Same parameters, same window,
same control: the system lands at the **22nd percentile** of random draws and loses
32.3%. Not a cost artifact — it holds at every cost level including zero. Full
write-up in "Third market: the UK". The natural follow-up it created: **benchmark
the system against a long-only momentum portfolio**, since every result across four
market-periods now lines up with "did momentum pay here, then?" rather than with
selection skill.

**Standing rule agreed with the user:** stop grid-searching for the best value of a
parameter. Future changes should come from a stated hypothesis tested once (e.g.
"the trailing stop should be volatility-scaled because the US result suggests the
thresholds are volatility quantities"), not from sweeping a range and taking the
maximum. One motivated trial costs almost nothing in DSR terms; twenty do.

Also open, unchanged from before:
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
