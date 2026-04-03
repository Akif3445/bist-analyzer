# BIST Smart Investment Assistant

A comprehensive stock analysis platform for Borsa Istanbul (BIST) and US markets. The system combines technical analysis, news sentiment analysis, and valuation metrics to generate investment signals with a 0-100 composite scoring model.

Live demo: [bist-analyzer.streamlit.app](https://bist-analyzer-wrtwzcsmdv4z9fxrpkkrcz.streamlit.app)

---

## Overview

This platform provides automated stock analysis by processing multiple data sources simultaneously. It fetches real-time price data, analyzes news sentiment from 20+ Turkish and English sources, and computes technical indicators to produce a unified buy/sell signal for each stock.

The system covers 80+ BIST stocks and 50+ US stocks. All analysis runs client-side through Streamlit, with SQLite used for local caching and portfolio persistence.

---

## System Architecture

The application consists of two main modules:

**bist_analyzer.py** — Core application (~8500 lines). Contains all scoring engines, page renderers, database operations, backtesting logic, and the Streamlit UI.

**news_engine.py** — News sentiment engine (~1600 lines). Handles RSS fetching from 20+ sources, hybrid sentiment classification (BERT + keyword-based), source credibility weighting, and speculation filtering.

---

## Scoring Model

Each stock receives a composite score from 0 to 100, calculated from four components:

| Component | Weight | Source |
|-----------|--------|--------|
| Technical Score | 35% | SMA, RSI, MACD, Bollinger, Stochastic, OBV, ADX |
| News Sentiment | 35% | 20+ RSS sources, weighted by source credibility |
| Upside Potential | 20% | Analyst target price vs. current price |
| Valuation | 10% | P/E, P/B ratios relative to sector averages |

When analyst target prices are unavailable, the weight is redistributed to Technical and Sentiment components. The same applies when fundamental data is missing.

### Signal Thresholds

| Score Range | Signal | Meaning |
|-------------|--------|---------|
| 72 - 100 | GUCLU AL (Strong Buy) | Multiple indicators aligned positively |
| 57 - 71 | AL (Buy) | Majority of indicators positive |
| 43 - 56 | NOTR (Neutral) | Mixed signals, no clear direction |
| 30 - 42 | SAT (Sell) | Majority of indicators negative |
| 0 - 29 | GUCLU SAT (Strong Sell) | Multiple indicators aligned negatively |

---

## Technical Indicators

The TechnicalEngine computes the following indicators from OHLCV data:

- **SMA 50/200**: Trend direction, golden/death cross detection
- **RSI (14)**: Momentum, overbought/oversold levels
- **MACD (12, 26, 9)**: Signal line crossovers, histogram divergence
- **Bollinger Bands (20, 2 sigma)**: Volatility and mean reversion
- **Stochastic Oscillator (14, 3)**: Momentum confirmation
- **OBV (On-Balance Volume)**: Volume-price divergence
- **ADX (14)**: Trend strength filter (multiplier applied when ADX > 25)
- **ATR (14)**: Volatility measurement for risk calculations
- **52-Week Position**: Current price relative to yearly range

Each indicator group contributes a sub-score to the total technical score (max 100). The ADX acts as a multiplier rather than a direct contributor — strong trends amplify existing signals.

---

## News Sentiment Engine

The sentiment engine uses a three-layer classification approach:

**Layer 1 — Strong Pattern Override**: Regex patterns for definitive financial events (e.g., "rekor kar acikladi", "iflas basvurusu"). These bypass all other layers.

**Layer 2 — BERT Model**: When available, uses `savasy/bert-base-turkish-sentiment-cased` for Turkish language understanding. Predictions with confidence >= 70% are accepted directly.

**Layer 3 — Weighted Keywords**: 120+ positive and 98+ negative financial terms with severity weights (1x normal, 2x strong, 3x critical). Falls back to this when BERT is unavailable or uncertain.

### Source Credibility

News sources are categorized into three tiers:

| Tier | Weight | Sources |
|------|--------|---------|
| Tier 1 (3x) | Highest | Bloomberg HT, NTV Ekonomi, Haberturk |
| Tier 2 (2x) | Standard | Hurriyet, Dunya Gazetesi, Investing.com TR, CNN Turk |
| Tier 3 (1x) | Supplementary | Google News, Bing News, Yahoo Finance |

Additional factors that affect the final news score:
- **Recency**: News from the last 2 days gets 1.5x weight, 7+ day old news gets 0.7x
- **Official terms**: KAP disclosures, financial reports, and corporate actions receive confidence bonuses
- **Duplicate detection**: Same news from multiple sources increases confidence
- **Speculation filter**: Social media language, pump/dump terms, and unverified claims are excluded

---

## Backtesting

The BacktestEngine supports five strategy modes:

| Mode | Description |
|------|-------------|
| Swing | Short-term trades, 5-20 day holding period |
| Trend | Follows established trends using SMA crossovers |
| Universal | Balanced approach combining swing and trend signals |
| Investor | Long-term positions with wider stop-losses |
| Buy & Hold | Benchmark comparison, no active trading |

Backtests use point-in-time (PIT) data to avoid look-ahead bias. Each test point only uses data available at that historical moment.

---

## Signal Tracker

The Signal Tracker records every buy/sell signal the system generates and monitors price movement afterward. Signals are tracked across five time periods: 1 day, 3 days, 7 days, 14 days, and 30 days.

The tracker shows:
- Daily, weekly, and monthly signal views
- Per-period return percentages for each signal
- Overall accuracy statistics
- Signal distribution and performance breakdown

Both BIST and US market signals are tracked independently.

---

## Time Machine

The Time Machine module runs full historical simulations asking "what if we applied today's strategy in the past?" It supports six portfolio styles:

- Aggressive, Defensive, Momentum, Value, Stable, and Custom

Each simulation picks stocks using the scoring model at each historical point, then tracks portfolio performance forward with realistic entry/exit logic.

---

## Pages

### BIST Market (8 pages)
1. **Piyasa Ozeti** — Market overview with macro indicators (USD/TL, EUR/TL, Gold, BIST100) and today's signals summary
2. **BIST Listesi** — Full scan of 80+ stocks sorted by score
3. **Hisse Analizi** — Deep analysis for a single stock with all indicators
4. **Portfolyum** — Personal portfolio tracker with P&L
5. **Backtest** — Historical strategy testing
6. **Sistem Portfolyleri** — Pre-built smart portfolios
7. **Zaman Makinesi** — Historical "what if" simulations
8. **Sinyal Takip** — Signal accuracy tracker

### US Market (7 pages)
1. **US Analiz** — Single stock analysis for US equities
2. **US Backtest** — Backtesting for US stocks
3. **US Stock List** — US stock scanner
4. **US Portfolios** — US smart portfolios
5. **Portfolyum** — Shared portfolio page
6. **Zaman Makinesi** — Shared time machine
7. **US Sinyal Takip** — US signal tracker

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Data | yfinance |
| News | feedparser, requests (20+ RSS sources) |
| Sentiment | BERT (optional), keyword-based (default) |
| Database | SQLite |
| Charts | Plotly |
| Deployment | Streamlit Cloud |

---

## Installation

```bash
git clone https://github.com/Akif3445/bist-analyzer.git
cd bist-analyzer
pip install -r requirements.txt
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your-key-here
```

Run:
```bash
streamlit run bist_analyzer.py
```

For BERT sentiment (optional, requires ~2GB additional):
```bash
pip install transformers torch
```
The application automatically falls back to keyword-based sentiment when BERT is not installed.

---

## Configuration

The `.streamlit/config.toml` file contains theme and server settings. The application uses a dark theme by default.

Secrets can be configured through:
1. `.env` file (local development)
2. Streamlit Cloud secrets panel (production)

The application reads secrets using `st.secrets` first, falling back to `os.getenv`.

---

## Data Sources

**Price Data**: Yahoo Finance via yfinance library. Supports all BIST and major US exchange tickers.

**News Sources (Turkish)**: Bloomberg HT, NTV Ekonomi, Haberturk, Hurriyet, Milliyet, Sabah, Dunya Gazetesi, Investing.com TR, Finans Gundem, Ekonomim, Para Analiz, Foreks, CNN Turk Ekonomi, Google News TR

**News Sources (English)**: Google News EN, Bing News, Yahoo Finance, Yahoo Finance RSS

---

## Disclaimer

This software is for informational and educational purposes only. It does not constitute financial advice. All investment decisions carry risk. Past performance does not guarantee future results. Always do your own research before making investment decisions.

---

## License

This project is proprietary. All rights reserved.
