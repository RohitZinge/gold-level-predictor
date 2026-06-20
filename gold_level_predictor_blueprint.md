# 🟡 Gold Ladder-Level Strength Predictor — Full Project Blueprint

**Goal in one line:** for each trading day, predict **which of your 21 ladder levels is most likely to act as real support/resistance** — as a probability (e.g. 72%) with the reason why — so you trade only the strong levels instead of every line.

---

## 1. The big picture (architecture)

The project has **two separate parts**. The model lives in Python; TradingView stays only your chart.

```
        OFFLINE  (build once, retrain monthly)          ONLINE  (every new day)
   ┌────────────────────────────────────────┐      ┌──────────────────────────────┐
   │ 1  Get gold data (years of H1 bars)     │      │ A  New day opens              │
   │ 2  Build 21 levels per day (formula)    │      │ B  Read today's open          │
   │ 3  Find touches + label react / break   │      │ C  Build today's 21 levels    │
   │ 4  Build features (the fingerprint)     │ ───► │ D  Build features             │
   │ 5  Train + validate model               │ model│ E  Model scores each level    │
   │ 6  Save model + calibration + SHAP      │      │ F  Show scores + reasons      │
   └────────────────────────────────────────┘      └──────────────────────────────┘
                                                              │
                                                   TradingView = your chart only
```

- **Offline pipeline** — builds and trains the model from history. Run it once, then re-run (retrain) every month or so.
- **Online scoring** — runs each day, gives you the level scores + reasons.

> **Key timing fact:** all 21 levels come from the **day's open**, so scores can only be produced **once the new day opens** (when the open price is known) — not the night before.

---

## 2. Data — where, what, how much

| Source | Cost | Notes |
|---|---|---|
| Kaggle XAU/USD (2004–2026) | Free | Fastest start, ready-made CSV |
| Dukascopy (`dukascopy-node`) | Free | Clean, scripted, reproducible |
| MetaTrader 5 + Python | Free | If you already trade on MT5 |
| FirstRate / Barchart | Paid | Premium clean data |

- **Timeframe:** start with **1-hour (H1)** bars. Move to 15-min later for sharper reaction labels.
- **History:** **5+ years** (more = better learning; we normalize, so gold's big price jump doesn't matter).
- **Columns needed:** `datetime, open, high, low, close` (volume optional).
- **Validation before use:** remove duplicate timestamps, flag gaps (weekends/holidays), drop bad bars (high < low, zero prices), confirm timezone, sort by time.

---

## 3. The workflow — step by step

### Phase 1 — Data collection & cleaning
Get the CSV, load with pandas, run the checks above, save a clean file (parquet).
**Output:** one clean price table.

### Phase 2 — Build the 21 levels (your formula in Python)
For every day, take that day's open and run your formula (digit-sum → √ → step size → 21 prices). Same numbers as your chart.
**Output:** 21 level-prices per day, each tagged with its **step number** (−10…+10) and **thick/thin** type.

### Phase 3 — Detect touches & create labels (the target)
Walk forward through the intraday bars. A level is **touched** when a bar reaches it. After a touch, decide the outcome:
- **Reacted (1):** price turned away by at least *X* (e.g. one step size, or 1 ATR) **without** closing through it.
- **Broke (0):** price **closed through** the level by a buffer and kept going.

Each touch becomes one labelled example.
> Rule: the label looks at bars **after** the touch (allowed — it's the answer key). Features must **not**.

**Output:** every level-touch in history with a 1/0 outcome.

### Phase 4 — Build features (the fingerprint)
For each touch, compute facts using only **past** info (up to the touch bar):
- **Ladder position:** step number, thick/thin, distance from the open
- **Distance from current price** (in ATR)
- **Freshness:** untested today, or how many times tested
- **Break-and-retest state:** untested → broken → retest
- **Confluence:** distance to round number, to previous-day high/low, to weekly high/low, to other days' levels
- **Market context:** ATR (volatility), trend (price vs moving average / slope), today's step size, gap from yesterday's close
- **Time/session:** hour, session (Asia / London / NY), day of week

**Output:** the full feature table — one row per touch.

### Phase 5 — Assemble dataset & split by time
Join features + labels into the final dataset. Split by **time, never random:**
- **Train:** oldest years
- **Validation:** middle years (tuning + calibration)
- **Test:** most recent (untouched until the very end)

> Why: shuffling leaks the future into the past and gives fake-good results.

**Output:** train / validation / test sets.

### Phase 6 — Train the models
- **Baseline first:** Logistic Regression — simple, sets the bar to beat.
- **Main model:** Gradient-Boosted Trees (**LightGBM** or **XGBoost**) — best for this kind of table data; handles non-linear patterns and feature interactions.
- (Optional sanity check: Random Forest.)
- **Calibrate** the probabilities (isotonic / Platt) so "70%" really means ~70%.
- Handle class imbalance (likely more breaks than reactions) with class weights.

**Output:** trained, calibrated model.

### Phase 7 — Evaluate & validate (the "foolproof" part)
Check on the **test** set it never saw:
- **AUC / PR-AUC** — can it separate reactions from breaks?
- **Calibration** — Brier score + reliability curve (are the % honest?)
- **Beat the baseline** — must do better than "always pick the open" or random.
- **Walk-forward test** — works across several periods, not one lucky year.
- **Trading backtest** — if you only trade levels above a probability threshold, is the win-rate / expectancy better than trading all levels, **after spread/slippage**?

**Output:** an honest scorecard. If it doesn't beat the baseline, we improve the features — not the headline number.

### Phase 8 — Explain (why each level got its score)
Use **SHAP** to break each score into reasons, e.g. *"PDL-aligned +12%, fresh +9%, round number +7%."*
**Output:** a plain-English "why" for every level — your confirmation.

### Phase 9 — Daily use (inference)
At each new day's open:
1. Read the open → build today's 21 levels.
2. Compute features for each.
3. Model scores all 21 → ranked list + reasons.
4. You see it via: a **notebook** → a **Streamlit dashboard** (enter open, see scored levels) → later a scheduled job that sends the day's scores to **Telegram / email**.

### Phase 10 — Monitor & retrain (MLOps)
- Log every prediction + the real outcome later.
- Track live accuracy / calibration over time (drift watch).
- **Retrain monthly** (or when accuracy drops) on the newest data.
- Automate with **GitHub Actions** (you already use it).

---

## 4. Which model & why

| Stage | Model | Why |
|---|---|---|
| Baseline | Logistic Regression | Simple, interpretable, sets the bar |
| Main | LightGBM / XGBoost | Best for tabular data, non-linear, fast |
| Probabilities | Calibration (isotonic) | Makes the % trustworthy |
| Explain | SHAP | The "why" for each level |

We **don't** use deep learning — for table data of this size, gradient-boosted trees win, train faster, and are far easier to explain.

---

## 5. Tech stack

**Core (must-have)**
- Python, pandas, numpy
- scikit-learn, LightGBM / XGBoost
- SHAP
- matplotlib / plotly
- joblib (save model), parquet (store data)

**MLOps add-ons (great for your ML-Engineer / MLOps goal)**
- **MLflow** — track experiments & models
- **DVC** — version the data
- **pytest** — tests for the pipeline
- **GitHub Actions** — auto test + retrain (you already use it)
- **FastAPI + Docker** — serve scores as an API
- **Streamlit** — the dashboard you actually click

---

## 6. How we keep it trustworthy (and the honest limits)

- **No leakage:** features use only past bars; strict time order; time-based split.
- **Beat a baseline:** the model must prove it adds value, every time.
- **Honest %:** calibration so probabilities mean what they say.
- **Real backtest:** include spread / slippage; count only high-probability trades.
- **Retrain + monitor:** gold's regime changes, so we keep the model fresh.
- **Reality check:** this gives a probability **edge**, not certainty. No model predicts the market perfectly. Always size positions so a wrong call can't hurt you. This is a research tool — not financial advice.

---

## 7. Milestones (suggested order)

| # | Milestone | You'll have… |
|---|---|---|
| 1 | Clean data ready | a validated price file |
| 2 | Levels + labels + features | the full dataset |
| 3 | Baseline model + scorecard | your first honest result |
| 4 | Main model + tuning + calibration | the real model |
| 5 | SHAP + backtest | scores with reasons + an edge check |
| 6 | Streamlit dashboard | daily scores you can use |
| 7 | MLflow + GitHub Actions + monitoring | a complete MLOps project |

---

*Next decision: pick the data source (Phase 1). Everything else follows from there.*
