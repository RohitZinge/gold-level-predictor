# 🟡 Gold Ladder-Level Strength Predictor — MASTER HANDOFF

> **Purpose of this document.** This is a complete, self-contained brief written to be **pasted into a fresh AI chat** so a new assistant instantly understands: (1) what this project is, (2) everything that has been built so far and how the code works, (3) the results we have, and (4) what is left to do. Read it top to bottom. Every number, file, and design decision below is real and current as of **2026-06-20**.

---

## 0. HOW TO USE THIS DOCUMENT (instructions to the new AI)

You are taking over an existing, **working** machine-learning project. It is **not** a fresh start — 9 of 10 planned phases are built, tested, and producing honest results, and the model has since been materially improved (see §8.8). Your job is to help the user (a solo developer learning ML/MLOps, building this for his own gold trading) continue from here: extend it, harden it, or build the remaining MLOps phase.

Ground rules the project lives by — keep honoring them:
- **No look-ahead leakage.** Features may use only bars at/before the moment a level is touched. Enforced by an automated assertion (see §7.4). Never break it.
- **Time-ordered splits only** (never random shuffle) — shuffling leaks the future.
- **Beat a baseline** and **calibrate probabilities** so "70%" really means ~70%.
- **Backtest with costs.** A probability edge is not certainty. This is a research tool, **not financial advice**.
- **Measure before keeping.** Every accuracy change is accepted only if it improves the walk-forward / high-confidence backtest; things that don't help are reverted (we already did this with hyperparameter tuning — see §8.8).

The codebase is Python (pandas, LightGBM, scikit-learn, SHAP, Streamlit). TradingView is only the user's chart — no model runs there.

---

## 1. THE PROJECT IN ONE PARAGRAPH

For each trading day, the user draws **21 horizontal "ladder" levels** on his gold (XAU/USD) chart, all derived by a fixed formula from that day's opening price. Some levels act as strong support/resistance (price reacts off them); others get blown through (price breaks them). **This project trains a calibrated ML model that scores each of the 21 levels with a probability that it will hold (react) vs. break — with plain-English reasons (SHAP) — so the user trades only the strong levels instead of every line.** Scores can only be produced once the day opens (all 21 levels depend on the open price).

---

## 2. THE TRADING CONCEPT (domain knowledge the model encodes)

### 2.1 The 21 levels — the exact formula
Every day's 21 levels come **purely from that day's OPEN price**. Code: [src/levels.py](src/levels.py).

```
digit_sum   = sum of the digits of int(day_open)          # 4300 -> 4+3+0+0 = 7
step_factor = (sqrt(day_open) * (digit_sum / 10) / 2) / 20000
step_size   = day_open * step_factor                       # $ spacing between levels
level(n)    = day_open * (1 + n * step_factor)   for n = -10 .. +10
```

- `n = 0` is the open (the centre / "yellow line"). `n = -10..-1` are 10 levels below; `n = +1..+10` are 10 above. **21 levels total.**
- Each level is tagged: `type = "open"` at step 0, `"thick"` on even steps, `"thin"` on odd steps.
- **Worked example (open = 4300):** digit_sum 7, step_factor ≈ 0.00114755, step size ≈ **$4.93**; lowest level (n=−10) ≈ 4250.7, highest (n=+10) ≈ 4349.3.
- These numbers must match exactly what the user's TradingView indicator draws.

### 2.2 React vs. Break (what we predict)
When price reaches a level (a **touch**), one of two things happens:
- **Reacted (1):** price turned away from the level by a threshold **without** a committed close through it. (The level held.)
- **Broke (0):** price **closed through** the level by a buffer. (The level failed.)

The model's target is `outcome` = 1 (react) / 0 (break). Base rate in our data: **~41% react / ~59% break.**

### 2.3 Timeframes / market mechanics
- Instrument: **XAU/USD**, spot gold.
- The user watches the **15-minute** chart, confirms reactions on the **5-minute**, enters on the 1-minute. The labels mirror this: **touch detected on M15, react/break resolved on M5.**
- **Gold has a daily settlement break 17:00–18:00 New York** (no bars). The **daily OPEN = the 18:00 NY reopen price.**
- The user's chart is **FOREX.com**, whose daily boundary is **03:30 IST = 18:00 NY = 22:00 UTC**, and it **plots the BID** price.

---

## 3. PROJECT STATUS — PHASE CHECKLIST

The full plan (Phases 1–10) lives in [gold_level_predictor_blueprint.md](gold_level_predictor_blueprint.md).

| Phase | What | Status |
|---|---|---|
| 1 | Data collection & cleaning | ✅ DONE |
| 2 | Build the 21 levels (formula) | ✅ DONE |
| 3 | Detect touches + label react/break | ✅ DONE |
| 4 | Build features (the fingerprint) | ✅ DONE |
| 5 | Assemble dataset + time split | ✅ DONE |
| 6 | Train baseline + main model + calibrate | ✅ DONE |
| 7 | Evaluate + walk-forward + backtest | ✅ DONE |
| 8 | SHAP explanations | ✅ DONE |
| 9 | Daily tool (Streamlit dashboard) | ✅ DONE |
| 10 | MLOps (MLflow / DVC / CI / auto-retrain / monitoring) | ❌ NOT BUILT |
| + | **Accuracy-improvement work** (features, tuning, history) | ✅ DONE — see §8.8 |

**ALL 9 modelling/product phases are complete**, plus an accuracy-improvement pass. Only the MLOps automation layer (Phase 10) remains.

---

## 4. REPOSITORY LAYOUT (every file, what it does)

Root: `/Users/rohitzinge/Desktop/TreadingLevelML` (**not** a git repo yet).

```
TreadingLevelML/
├── gold_level_predictor_blueprint.md   # The original 10-phase plan/spec
├── FEATURES.md                         # Auto-generated feature dictionary (Phase 4)
├── PROJECT_MASTER_HANDOFF.md           # THIS FILE
├── conftest.py                         # pytest path setup
│
├── src/
│   ├── levels.py            # Phase 2: the 21-level formula (get_day_levels, step_size)
│   ├── fetch_data.py        # Phase 1: download XAUUSD M5 from Dukascopy (npx CLI)
│   ├── data_loader.py       # Phase 1: clean M5, resample M15/H1/H4, build D1 (NY 5PM boundary)
│   ├── convert_pine_opens.py# Phase 1: convert a TradingView Pine-log export -> forexcom_daily_opens.csv (+validate)
│   ├── build_levels.py      # Phase 2: write 21 levels/day -> levels.parquet (anchors to FOREX.com opens)
│   ├── build_labels.py      # Phase 3: touches (M15) + react/break (M5) -> labels.parquet
│   ├── build_features.py    # Phase 4: 33 leakage-safe features/touch -> dataset.parquet (+ leak assertion)
│   ├── model_config.py      # Shared config: DROP/CATEGORICAL, LightGBM params, monotone constraints (single source)
│   ├── tune_model.py        # Optional random hyperparameter search (honest split). NOTE: its result did NOT help — see §8.8
│   ├── train_model.py       # Phases 5-7: time-split, LogReg baseline, LightGBM, isotonic calib, scorecard
│   ├── walkforward_backtest.py  # Phase 7 harden: 4-fold walk-forward + costed backtest + operating-point sweep
│   └── explain.py           # Phase 8: SHAP global plots + LevelExplainer (plain-English reasons)
│
├── app/
│   └── daily_tool.py        # Phase 9: Streamlit dashboard + headless --demo mode
│
├── tests/
│   └── test_levels.py       # unit tests for the level formula
│
├── data/
│   ├── raw/
│   │   ├── xauusd_m5.csv             # combined raw M5 (Dukascopy bid), 2011-2026
│   │   └── _tmp/xauusd_m5_<year>_bid.csv   # cached yearly chunks 2011-2026 (resume-safe)
│   ├── external/
│   │   ├── forexcom_daily_opens.csv      # user's EXACT FOREX.com daily opens 2011->today (4,007 rows)
│   │   └── forexcom_daily_opens.csv.bak  # backup of the previous (2021->today) opens file
│   └── processed/
│       ├── m5.parquet   m15.parquet   h1.parquet   h4.parquet   d1.parquet   # 2011-2026
│       ├── levels.parquet    # 21 levels/day (78,603 rows = 3,743 days × 21)
│       ├── labels.parquet    # one row per touch + outcome (129,361 rows)
│       └── dataset.parquet   # labels + 33 features (129,361 × 39) — the training table
│
├── models/
│   ├── baseline_logreg.joblib    # Logistic Regression baseline (sklearn Pipeline)
│   ├── lgbm_calibrated.joblib    # dict{model(calibrated), features, categorical, base_rate}
│   ├── metrics.json              # single-split scorecard
│   └── feature_importance.csv    # LightGBM gain per feature
│       # NOTE: models/best_params.json is intentionally ABSENT — tuning didn't help (§8.8)
│
└── reports/
    ├── walkforward_metrics.csv   # per-fold AUC/PR/Brier/hit-rates (4 folds)
    ├── oos_predictions.parquet   # stitched out-of-sample preds 2023-2026
    ├── touch_scores_oos.csv      # human-readable OOS predictions (date, level, model%, actual, correct)
    ├── operating_points.csv      # confidence-threshold sweep (trades/win/expectancy/PF per threshold)
    ├── backtest_summary.csv      # strategy comparison (ALL vs RANDOM vs CONF>=0.65/0.70)
    ├── backtest_trades.parquet   # per-trade R results
    ├── equity_curve.png          # OOS cumulative R (net of costs)
    ├── shap_beeswarm.png         # global SHAP drivers (react+ / break−)
    ├── shap_importance.png       # mean|SHAP| feature importance
    └── daily_opens_2026.csv      # helper export
```

---

## 5. THE DATA (Phase 1) — sources, boundaries, gotchas

**Code:** [src/fetch_data.py](src/fetch_data.py), [src/data_loader.py](src/data_loader.py), [src/convert_pine_opens.py](src/convert_pine_opens.py).

### 5.1 Fetching
- Source: **Dukascopy** via the `dukascopy-node` npx CLI. Instrument `xauusd`, timeframe `m5`.
- `--price` ∈ {bid, ask, mid}; **default `bid`** (matches the user's FOREX.com/TradingView chart, which plots bid). `mid` = column-wise (bid+ask)/2.
- Downloaded in **yearly chunks**, cached in `data/raw/_tmp/` so re-runs resume and switching `--price` reuses cached sides. Combined → `data/raw/xauusd_m5.csv` (ISO-8601 UTC datetime).
- **GOTCHA:** dukascopy-node throttles aggressive batching → "Unknown error". Proven gentle recipe (the defaults): `--batch-size 10 --batch-pause 1000 --retries 10 --retry-pause 1500`, plus `-re -fr`. **Do NOT raise batch-size.**
- To extend history, just widen `--from`; caching reuses existing chunks. (This is exactly how we extended back to 2011 — see §8.8.)

### 5.2 Cleaning + resampling
`data_loader.py` parses UTC, sorts, drops duplicate timestamps, drops bad bars (high<low, price≤0, OHLC bound violations), and reports time gaps classified by cause (`daily_break` ≈55–70 min, `weekend`, `holiday`, `minor`).

- Intraday timeframes (M5/M15/H1/H4) are plain UTC resamples (`label="left", closed="left"`, empty buckets dropped).
- **D1 (daily) is special:** built on the **FOREX daily boundary 17:00 America/New_York → 17:00 next day**, DST-safe (groups on the *local wall clock*, never on fixed Timedeltas, so 17:00 never lands in the 02:00 DST gap). Index = the session-open instant (17:00 NY). The day's OPEN = first M5 open at/after the boundary. CLI: `--day-start-tz` (default America/New_York), `--day-start-hour` (default 17).

### 5.3 Current dataset state
- Range **2011-01-02 → 2026-06-19**, **1,101,963 clean M5 bars, 0 bad/dupes** (older years came in clean too, ~70–74k bars/year).
- Counts: m5 1,101,963 / m15 367,934 / h1 92,407 / h4 24,747 / **d1 3,998 sessions**.
- Price type is **bid** (verified). Daily opens match the chart to ~$0.1–0.25 on normal liquidity; residual = Dukascopy-vs-FOREX.com feed diff + thin-reopen noise (gold spread is $3–4 at the 6 PM reopen).

### 5.4 The exact-opens problem (important!)
The user could **not** reliably match Dukascopy's daily open to his FOREX.com chart on weekend-gap days (median diff $0.78, but up to **$114** on volatile Mondays). Fix:
- He exported his **exact FOREX.com daily opens via TradingView Pine `log.info`** per daily bar. The export now covers **2011 → today (4,007 days)**.
- The raw Pine export is `Date,Message` where `Message = "<NY session-open date>,<open price>"`. Convert it with **`src/convert_pine_opens.py <pine.csv> --out data/external/forexcom_daily_opens.csv --validate <existing>`** → cols `date, open_forexcom` where `date` = NY session-open date + 1 = chart candle day (exactly the key `build_levels.py` looks up). The converter was validated: the 1,412 overlapping days vs the previous file matched **exactly (Δ0.0000)**.
- `build_levels.py --use-forexcom-opens --max-gap 5` **anchors the 21 levels to these exact opens**, and **drops** days that have no FOREX.com open or where |FX − Dukascopy open| > $5 (255 such days — unreliable, since intraday bars used for labels are still Dukascopy).
- Live: the user **types the open** (exact). For new days he re-runs the Pine exporter (then `convert_pine_opens.py`) or types it live.

---

## 6. PHASE 2 — BUILD THE LEVELS

**Code:** [src/levels.py](src/levels.py) (formula) + [src/build_levels.py](src/build_levels.py) (writes the table).

- `build_levels.py` reads `d1.parquet`, applies `get_day_levels()` to each day's open, writes one row per level per day to `data/processed/levels.parquet`: columns `date, day_open, step_number, type, level_price`.
- `date` = the trading day's **close date** = NY session-open date + 1 (matches the TradingView candle date the user sees).
- With `--use-forexcom-opens`, levels are anchored to the user's exact opens (see §5.4). Current file: **78,603 rows = 3,743 days × 21 levels.**
- Tested in [tests/test_levels.py](tests/test_levels.py).

---

## 7. PHASE 3 & 4 — LABELS AND FEATURES (the heart of correctness)

### 7.1 Labels (Phase 3) — [src/build_labels.py](src/build_labels.py)
Output: `data/processed/labels.parquet`, one row per touch: `date, step_number, type, level_price, touch_time, outcome (1/0)`. **Labels only — no features here** (golden rule: only the label may look at bars after the touch).

Logic (matches the user's M15-watch / M5-confirm workflow):
- **TOUCH (on M15):** a level "comes into play" when an M15 bar's range enters the zone `[low − buf, high + buf]` after being outside. A touch is an **onset** (entering after being out), so a multi-bar hug = 1 touch and a genuine later retest = a new touch.
- **APPROACH SIDE:** from the M15 close *before* the touch (or the day open for the first bar): `ref ≥ level` ⇒ tested as **support** (came from above); else **resistance**. This sets which direction is "break" vs "react".
- **RESOLUTION (on M5)**, from the first M5 bar reaching the level after the M15 touch:
  - **BREAK (0):** an M5 bar **closes** beyond the level by `BREAK_BUFFER` on the through side (committed close-through; wicks don't count). May trigger on the M5 touch bar.
  - **REACT (1):** the away-side M5 extreme reaches `REACT_THRESHOLD` beyond the level, only on bars **after** the M5 touch bar.
  - **Earliest event wins; same-bar tie ⇒ break.** Unresolved by the 17:00-NY day end ⇒ **DROP** (ambiguous).
- **Thresholds (in step-size units, CLI-tunable):** `TOUCH_BUFFER 0.10`, `REACT_THRESHOLD 1.00`, `BREAK_BUFFER 0.25`.
- **Edge case:** step-0 (the open) is touched at bar 0 every day with `ref == level` → defaults to "support" side (~1/day, slight directional bias).

**Result (current, 2011–2026):** **129,361 touches**, react **41.0%** / break **59.0%**, ~2.2% dropped (ambiguous), **3,741 days, ~34.6 touches/day**.

### 7.2 Features (Phase 4) — [src/build_features.py](src/build_features.py)
Output: `data/processed/dataset.parquet` = label columns + **33 features**, one row per touch. **Shape: 129,361 × 39.** Feature dictionary auto-written to [FEATURES.md](FEATURES.md).

### 7.3 The 33 features (full dictionary)
Distances are normalized in **step-size units** (not raw $) so the model learns behaviour, not the price regime. The last 4 (★) were added in the accuracy round (§8.8).

| feature | meaning |
|---|---|
| `step_number` | ladder step −10..+10 (0 = open); the level's identity |
| `abs_step` | distance from the open in steps (\|step_number\|) |
| `type` | open / thick (even) / thin (odd) |
| `above_or_below_open` | +1 above the open, −1 below, 0 = open |
| `hours_since_open` | trading-hours since the 6 PM-NY open |
| `session` | Asia / London / NY at the touch (by UTC hour) |
| `day_of_week` | trading day's weekday (Mon=0) |
| `bars_since_day_open` | M15 bars elapsed since the day open |
| `step_size` | this day's ladder spacing in $ *(dropped before training)* |
| `atr_m15` | ATR over last 14 M15 bars, in $ *(dropped before training)* |
| `atr_steps` | ATR in step-size units (volatility vs the ladder) |
| `range_so_far_steps` | today's high−low so far, in step units |
| ★ `rel_volume` | touch-bar tick-volume vs its 20-bar rolling average (>1 = busier than usual) |
| ★ `vol_trend` | volume 8-bar vs 40-bar average (>1 = volume rising) |
| `price_vs_ema20_steps` | (price − EMA20 on M15) in step units |
| `price_vs_ema50_steps` | (price − EMA50 on M15) in step units |
| `ema20_slope_steps` | EMA20 change over 8 M15 bars, in step units |
| `recent_return_steps` | price change over last 8 M15 bars, in step units |
| `approached_from_above` | +1 came from above (support test), −1 below |
| `is_first_touch` | 1 if the level's first touch today |
| `prior_touches_today` | count of earlier touches of THIS level today |
| `levels_broken_before` | how many of the 21 levels price already closed through today |
| `bars_since_this_level_broken` | M15 bars since this level broke earlier today (NaN if not) |
| `dist_round_5_steps` | distance to nearest multiple of 5, in step units |
| `dist_round_10_steps` | distance to nearest multiple of 10, in step units |
| `dist_round_25_steps` | distance to nearest multiple of 25, in step units |
| `dist_round_50_steps` | distance to nearest multiple of 50, in step units |
| `dist_pdh_steps` | signed distance to previous-day high, in step units |
| `dist_pdl_steps` | signed distance to previous-day low, in step units |
| `dist_whigh_steps` | signed distance to prior-5-day high, in step units |
| `dist_wlow_steps` | signed distance to prior-5-day low, in step units |
| ★ `dist_nearest_prior_level_steps` | distance to the nearest level from the prior 5 days, in step units (small = confluence) |
| ★ `confluence_count` | # of prior-5-day levels within 0.5 step (stacked lines = stronger) |
| `gap_steps` | today's open − yesterday's close, in step units |
| `day_open` | the day's open price *(reference only; dropped before training)* |

**Label:** `outcome` — 1 = reacted, 0 = broke.

### 7.4 Leakage guard (the most important safeguard)
The **golden rule**: every feature uses only bars **≤ touch_time**. Enforced two ways:
1. **By construction:** rolling stats (ATR14, EMA20/50, slope, recent return, volume) are causal pandas ops computed **on M15** (not H1 — the H1 bar containing the touch would include post-touch ticks) and read at the touch bar; intraday state (cummax/cummin, `bars_since_open`, `levels_broken_before`) is read at the touch bar's expanding position; daily context (PDH/PDL, prior-5-day high/low, gap) and **cross-day confluence** use **only prior completed days**, never today's data.
2. **By assertion** (`assert_no_leakage`): rebuilds the features for 80 random touches on M15 **truncated at touch_time** and requires byte-identical values. If any feature peeked ahead, truncation would change it. **This assertion PASSES** (re-verified with the 4 new features and the extended history).

Missingness: `bars_since_this_level_broken` ~39% NaN (structural — level wasn't broken earlier today; LightGBM handles NaN natively); `dist_nearest_prior_level_steps` ~0.08% (first few days have no prior levels); everything else 0%.

---

## 8. PHASES 5–7 — TRAIN, CALIBRATE, EVALUATE

**Code:** [src/model_config.py](src/model_config.py) (shared config), [src/train_model.py](src/train_model.py) (single split), [src/walkforward_backtest.py](src/walkforward_backtest.py) (walk-forward + backtest + operating points).

### 8.1 What goes into the model
- Target: `outcome` (react 1 / break 0).
- **Dropped columns** (never fed to the model): ids/label + **raw-dollar** columns `level_price, day_open, step_size, atr_m15` — so the model learns level *behaviour*, not gold's price level. Everything fed in is **dimensionless**.
- Categorical: `type, session, day_of_week`.
- LightGBM params + (optional) monotonic constraints live in **`model_config.py`** so `train_model.py` and `walkforward_backtest.py` can never drift apart. Env toggles `USE_TUNED` / `USE_MONOTONE` exist for experiments; both default OFF for monotone, and there is no `best_params.json` (see §8.8).

### 8.2 Split (time-based, never shuffled)
`train_model.py` splits by **date** (whole days stay in one split): train `< 2024-01-01` (2011–2023), val `2024`, test `≥ 2025-01-01` (2025–2026). The walk-forward script does the more rigorous expanding-window version below.

### 8.3 Models
- **Baseline:** Logistic Regression (`class_weight="balanced"`, median/most-frequent imputation, StandardScaler + OneHot) — sets the bar to beat.
- **Main:** **LightGBM** (`n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=80, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, class_weight="balanced"`), early-stopped on validation (patience 150; it stops itself around iter ~2000).
- **Calibration:** **isotonic** on the validation set (via `CalibratedClassifierCV` + `FrozenEstimator`) so "70%" ≈ 70% real.

### 8.4 Single-split scorecard (current, `models/metrics.json`, full 2011–2026 history)
| model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| no-skill (base rate) | 0.500 | 0.387 | 0.238 |
| baseline LogReg | 0.552 | 0.439 | 0.244 |
| LightGBM (raw) | **0.697** | 0.598 | 0.213 |
| LightGBM (calibrated) | 0.697 | 0.584 | **0.210** |

### 8.5 Walk-forward validation (the honest test) — `reports/walkforward_metrics.csv`
Expanding windows, **retrain + recalibrate each fold**, calibration set carved from the end of the training window so the test year is fully out-of-sample (fit grows back to 2011):

```
fit 2011-2021 | cal 2022 | TEST 2023
fit 2011-2022 | cal 2023 | TEST 2024
fit 2011-2023 | cal 2024 | TEST 2025
fit 2011-2024 | cal 2025 | TEST 2026
```

| fold | n_test | react% | ROC-AUC | Brier |
|---|---|---|---|---|
| test 2023 | 6,073 | 38.2% | 0.703 | 0.207 |
| test 2024 | 7,376 | 39.5% | 0.718 | 0.205 |
| test 2025 | 6,444 | 38.4% | 0.697 | 0.208 |
| test 2026 | 2,411 | 39.7% | 0.697 | 0.212 |
| **mean ± std** | | | **0.704 ± 0.010** | 0.208 ± 0.003 |

Also: PR-AUC 0.591 ± 0.016, hit-rate@P≥.70 **0.793**, hit-rate@P≥.60 0.713. **Read:** AUC is high *and* very stable across all 4 years (0.697–0.718) — a real edge, not one lucky year, and leakage-free (the Phase-4 leak test passes). The stitched OOS predictions (2023–2026, 22,304 touches) are saved to `oos_predictions.parquet` and human-readably to **`reports/touch_scores_oos.csv`**.

### 8.6 Backtest + operating point (does the edge pay after costs?) — `reports/operating_points.csv`, `backtest_summary.csv`
Trade rule: at a touch with P(react) ≥ threshold, bet the level **holds** (long if approached from above, short if from below). Entry = level price, stop = `BREAK_BUFFER` (0.25 step) beyond, target = 1 step away. M5 fills; if a bar spans both stop and target → assume **stop** (conservative). Unresolved by day end → exit at day close. Costs: **$0.30 round-trip**. P&L in **R** (R = the risk = break buffer). Trading **all** touches ≈ random (~+0.5R, PF ~1.6); the model's **selection** is the edge:

| confidence threshold | trades (2023–26) | win% | expectancy | profit factor |
|---|---|---|---|---|
| P ≥ 0.65 | 1,831 | 67.1% | +1.98R | 5.39 |
| P ≥ 0.70 | 1,225 | 70.9% | +2.10R | 5.93 |
| P ≥ 0.75 | 844 | 72.2% | +2.20R | 6.55 |
| **P ≥ 0.80 (recommended)** | **514** | **74.7%** | **+2.31R** | **7.42** |

The walk-forward script prints a **recommended operating threshold** (max expectancy with ≥180 trades) → currently **P(react) ≥ 0.80**. That's roughly **2–3 trades/week** at ~75% win rate.

**CAVEAT (state this honestly):** the trade design mirrors the label (target ≈ react condition, stop ≈ break condition), so it validates "trading the react calls," not a fully independent system. Flat 1R sizing, limit fill at the level, ignores simultaneous-trade capital limits.

### 8.7 Top feature drivers (`models/feature_importance.csv`, full history)
By LightGBM gain: `price_vs_ema20_steps` (4813), `ema20_slope_steps` (3946), `price_vs_ema50_steps` (3727), `bars_since_this_level_broken` (3229), `atr_steps` (2707), `dist_pdl_steps` (2693), `dist_pdh_steps` (2648), `recent_return_steps` (2431), `range_so_far_steps` (2399), **`rel_volume` (2192)**, `gap_steps`, `dist_round_10/5_steps`, `dist_wlow_steps`, `hours_since_open`. **Trend (price vs EMAs, slope), freshness-after-break, volatility, prior-day highs/lows, and the new volume feature dominate.** SHAP plots in `reports/shap_beeswarm.png`, `shap_importance.png`.

### 8.8 ACCURACY-IMPROVEMENT ROUNDS (2026-06-20) — what was tried, what stuck
Goal set with the user: maximize **trading precision** (win-rate/expectancy when confident), using existing data, feature-first.

- **Round 1 — new features (KEPT ✅).** Added cross-day **confluence** (`dist_nearest_prior_level_steps`, `confluence_count`) and **volume** (`rel_volume`, `vol_trend`). Leak test still passes. Lifted matched-coverage precision (e.g. top-507 PF 4.95 → 5.52).
- **Round 2 — hyperparameter tuning + monotonic constraints (REVERTED ❌).** Built `tune_model.py` (random search, tuned honestly on 2021→2022 only so test years stay clean) and monotonic constraints in `model_config.py`. A clean 3-way A/B showed **plain features + default params won at every operating point** — tuning overfit the small tuning window. So we **deleted `best_params.json`** and set `USE_MONOTONE` default OFF. The infrastructure remains for future use.
- **Round 3 — extended history to 2011 (KEPT ✅, the big one).** Re-downloaded Dukascopy M5 back to 2011 and got the user's exact FOREX.com opens for 2011–2026. Dataset grew 35,493 → **129,361 touches**. Same 2023–2026 test years, just more training history.

**Net effect across the rounds (walk-forward, same test years):**
| | Before (2021–26 data, orig features) | After (2011–26 data, +features) |
|---|---|---|
| Walk-forward ROC-AUC | 0.649 ± 0.017 | **0.704 ± 0.010** |
| Brier | 0.222 | **0.208** |
| Backtest matched top-507: win / PF | 65.7% / 5.52 | **74.8% / 7.41** |
| Recommended threshold | P≥0.75 → 220 trades | **P≥0.80 → 514 trades, 74.7% win, PF 7.42** |

The history extension was the single biggest lever (the dimensionless features let the older regime transfer). Tuning was a dead end — keep that lesson.

### 8.9 Overfitting / underfitting check (measured)
| set | ROC-AUC |
|---|---|
| Train (2011–2023) | 0.834 |
| Validation (2024) | 0.720 |
| Test (2025–2026) | 0.697 |

**Not underfitting** (train 0.83 ≫ baseline 0.55). **Mild, controlled overfitting** (train−test ≈ 0.14 gap), but test/val are consistent and the walk-forward is 0.704 ± 0.010 across 4 separate years → the gap does not hurt generalization. Adding more regularization (Round 2) did NOT improve test, so the gap is cosmetic, not harmful.

---

## 9. PHASE 8 — SHAP EXPLANATIONS

**Code:** [src/explain.py](src/explain.py).

- Global: SHAP **beeswarm** + **mean-|SHAP| bar** → `reports/shap_beeswarm.png`, `reports/shap_importance.png`.
- `LevelExplainer.explain(row)` turns one level's score into plain-English reasons FOR and AGAINST a reaction. SHAP runs on the underlying LightGBM **booster** (exact TreeSHAP via `pred_contrib`); each contribution is mapped **through the isotonic calibrator** so the "% reasons" live in the same calibrated-probability space as the headline number.
- Example output: *"React 84% (base 41%) — fresh after a break (+12%), on heavy volume (+9%), at the weekly low (+8%), but against the short-term trend (−6%)."*
- `feat_phrase()` holds the human phrasing for every feature, including the new ones (e.g. `rel_volume` → "on heavy/light/normal volume"; `confluence_count` → "N prior-day levels stacked here").

---

## 10. PHASE 9 — THE DAILY TOOL

**Code:** [app/daily_tool.py](app/daily_tool.py).

- Run UI: `streamlit run app/daily_tool.py`. Headless: `python3 app/daily_tool.py --demo <open> [YYYY-MM-DD]`.
- Each morning the user **types the day's OPEN** (read off the FOREX.com chart so levels match exactly). The app:
  1. builds the 21 levels (`get_day_levels`),
  2. scores each with the calibrated model assuming a **first, untested touch** (start-of-day defaults; new-volume features default to 1.0 = normal),
  3. pulls market **context** (trend/volatility/gap refs + prior-day levels for confluence) from the latest `data/processed`,
  4. ranks levels, flags **≥65%** as "levels to watch", shows top SHAP reasons.
- `rescore_level(level, day_open, current_market_state, expl, ctx, dow)` gives a **sharper mid-day score** when price is actually testing a level (accepts overrides: `current_price, bars_since_open, prior_touches, levels_broken, bars_since_break, session, range_so_far, ema20/50, rel_volume, ...`).
- Permanent in-app disclaimer; warns when cached market data is stale (level **prices** stay exact since they only depend on the typed open; only the **context-based scores** drift).
- **Operating-point note:** the tool's ≥65% flag is the open-time *watchlist*. The measured *take-the-trade* bar from the backtest is **P≥0.80** at the real touch (see §8.6) — consider wiring that tier in if desired (not yet done).

---

## 11. HOW TO RUN THE WHOLE PIPELINE (in order)

```bash
# Phase 1 — fetch + clean (bid by default); --from 2011 for the full history
python3 src/fetch_data.py --from 2011-01-01 --to 2026-06-20
python3 src/data_loader.py

# Phase 1b — convert the user's TradingView Pine-log opens export (+validate vs existing)
python3 src/convert_pine_opens.py "<pine-logs ... .csv>" \
    --out data/external/forexcom_daily_opens.csv --validate data/external/forexcom_daily_opens.csv

# Phase 2 — levels (anchored to the user's exact FOREX.com opens)
python3 src/build_levels.py --use-forexcom-opens --max-gap 5

# Phase 3 — labels (touch on M15, react/break on M5)
python3 src/build_labels.py

# Phase 4 — features (+ leakage assertion; must print PASSED)
python3 src/build_features.py

# Phases 5-7 — train + scorecard, then walk-forward + backtest + operating points
python3 src/train_model.py
python3 src/walkforward_backtest.py --cost 0.30

# (optional) hyperparameter search — NOTE: last run did NOT beat defaults; writes models/best_params.json
# python3 src/tune_model.py --trials 30

# Phase 8 — SHAP plots + worked examples
python3 src/explain.py

# Phase 9 — daily tool
streamlit run app/daily_tool.py            # or: python3 app/daily_tool.py --demo 4300 2026-06-19

# tests
pytest
```

**Stack:** Python, pandas, numpy, scikit-learn, **LightGBM**, **SHAP**, matplotlib, joblib, parquet, Streamlit. Node.js + npx (for `dukascopy-node`).

---

## 12. KEY DESIGN DECISIONS & TRUSTWORTHINESS RULES (don't undo these)

1. **Leakage-safe by construction + assertion** (§7.4). The assertion must keep passing — re-run after any feature change.
2. **Time-ordered splits + expanding-window walk-forward** — never shuffle.
3. **Dimensionless features only** (drop raw-$ columns) — so the model doesn't memorize a price regime. (This is exactly what let 2011-era data help the 2025-26 model.)
4. **Trend EMAs on M15, not H1** — the H1 bar containing the touch would include post-touch ticks (a leak).
5. **Calibrated probabilities** (isotonic). 6. **Beat the baseline** (LightGBM 0.70 vs LogReg 0.55 vs no-skill 0.50).
7. **Backtest with costs** and compare to RANDOM — selection, not direction, is the edge. **Trade the high-confidence bar (P≥0.80).**
8. **Levels anchored to the user's exact FOREX.com opens** (§5.4) — drops days where feeds disagree > $5.
9. **D1 boundary = 17:00 NY, DST-safe** on the local wall clock.
10. **Measure, then keep.** Accept a change only if walk-forward / high-confidence backtest improves; revert otherwise (tuning was reverted for this reason).

---

## 13. KNOWN LIMITATIONS & CAVEATS (be honest about these)

- **Real but moderate edge.** Walk-forward AUC ≈ 0.70 — a genuine tilt, not a crystal ball; the usable money is in the high-confidence subset.
- **Mild overfitting** (train 0.83 vs test 0.70) — controlled, not harmful (§8.9).
- **Backtest mirrors the label** (§8.6): validates trading the react calls, not an independent system. Flat sizing, limit fill, ignores capital limits across simultaneous trades.
- **Intraday data is Dukascopy** even when opens are FOREX.com-anchored; days with > $5 open disagreement are dropped (255 days) to keep labels reliable.
- **Step-0 (open) touches** have a slight built-in "support" directional bias.
- **Scores at the open assume a first untested touch**; they sharpen when price actually reaches the level (use `rescore_level`).
- **Not financial advice.** A probability edge ≠ certainty; size positions so a wrong call can't hurt.

---

## 14. WHAT'S LEFT — PHASE 10 (MLOps) + IDEAS

**Not built yet (the only remaining blueprint phase):**
- **MLflow** — experiment/model tracking. **DVC** — version the data/parquet artifacts.
- **CI (GitHub Actions)** — run `pytest` + the leakage assertion on every push; scheduled monthly retrain.
- **Monitoring / drift** — log each live prediction + the real outcome later; track live AUC/calibration; alert + retrain when accuracy drops.
- **Serving** — optional FastAPI + Docker, or a scheduled job pushing the day's scored levels to Telegram/email.

**First practical step if continuing:** this is **not a git repo yet** — `git init`, commit, push, then add a CI workflow that runs the tests + leakage check. (Confirm with the user before any push.)

**Modelling ideas not yet tried (next levers, in rough priority):**
1. **Pre-2011 history** (same method) — more data was the biggest win so far.
2. **Meta-labeling** (primary model picks side, secondary sizes the bet) and **sample-uniqueness weights** for the ~35 overlapping touches/day (addresses non-IID labels).
3. Higher-timeframe structure features (H1/H4 swings), approach-velocity, prior-day behavior of each step.
4. Wiring the **P≥0.80 trade tier** into the daily tool.

*(Already tried and NOT worth repeating: LightGBM hyperparameter tuning and monotonic constraints — see §8.8.)*

---

## 15. GLOSSARY

- **Ladder level / step:** one of 21 daily horizontal lines from the open-price formula; `step_number` −10..+10.
- **Touch:** an M15 bar's range reaches a level (after being outside the zone).
- **React (1) / Break (0):** the level held vs. price closed through it (the prediction target).
- **Step size:** $ spacing between adjacent levels that day (everything distance-related is normalized by this).
- **Confluence:** today's level lining up with prior days' levels — stacked lines hold better.
- **R:** unit of risk in the backtest = the break buffer (0.25 step); P&L is measured in multiples of R.
- **PDH/PDL, WHigh/WLow:** previous-day high/low; prior-5-day high/low.
- **OOS:** out-of-sample (test data the model never trained on).
- **Calibration:** post-processing so predicted probabilities match observed frequencies.
- **Operating point / threshold:** the P(react) cutoff above which you actually trade (currently 0.80).

---

*End of master handoff. Everything above reflects the real, current state of `/Users/rohitzinge/Desktop/TreadingLevelML` as of 2026-06-20. The project is fully functional through Phase 9, with an accuracy-improvement pass complete (walk-forward AUC ≈ 0.70 on 2011–2026 data); only Phase 10 (MLOps) remains.*
