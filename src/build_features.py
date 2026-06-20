#!/usr/bin/env python3
"""
Phase 4 — features (the fingerprint).  One row per touch, joined to the label.

GOLDEN RULE
    Every feature uses ONLY bars up to AND INCLUDING the touch bar (touch_time).
    Nothing after touch_time is ever read here — that window belongs to the
    label only (Phase 3). This is enforced two ways:
      (1) by construction — rolling stats are causal pandas ops read AT the touch
          bar; intraday state is read at the touch bar's expanding position;
          daily context uses only PRIOR completed days from d1.
      (2) by an assertion — for a random sample of touches we rebuild the
          features on data TRUNCATED at touch_time and require them to be
          identical. If any feature peeked ahead, truncation would change it.

Trend EMAs are on M15 (not H1) so only fully-closed bars up to the touch are
used (the H1 bar containing the touch would include post-touch ticks = a leak).

Inputs : data/processed/{m15,d1}.parquet, levels.parquet, labels.parquet
Output : data/processed/dataset.parquet  (label columns + features, 1 row/touch)

Usage:
    python3 src/build_features.py
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levels import step_size  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT_PATH = PROC / "dataset.parquet"
NOTE_PATH = ROOT / "FEATURES.md"

DAY_START_TZ = "America/New_York"
DAY_START_HOUR = 17

ATR_N = 14            # M15 bars for ATR
EMA_FAST, EMA_SLOW = 20, 50   # M15 EMA spans
SLOPE_N = 8           # M15 bars for EMA slope
RET_N = 8             # M15 bars for recent return
WEEK_DAYS = 5         # prior trading days for the "weekly" high/low
BREAK_MULT = 0.25     # step-size units: a level is "broken" on a close this far through
ROUND_NUMBERS = (5, 10, 25, 50)
VOL_N = 20            # M15 bars for the rolling tick-volume baseline
VOL_FAST, VOL_SLOW = 8, 40    # M15 bars for the volume-trend ratio (fast/slow)
CONFLUENCE_K = 5      # prior trading days whose levels count as confluence
CONFLUENCE_TOL = 0.5  # step-size units: a prior-day level this close = confluence

LABEL_COLS = ["date", "step_number", "type", "level_price", "touch_time", "outcome"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def trading_day(idx_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map UTC bar timestamps to their trading-day label (same as labels.date)."""
    local = idx_utc.tz_convert(DAY_START_TZ)
    bump = np.where(local.hour >= DAY_START_HOUR, 1, 0)
    return pd.to_datetime(local.date) + pd.to_timedelta(bump, unit="D")


def session_of(ts: pd.Timestamp) -> str:
    """FX session from the UTC hour (fixed mapping; covers 24h)."""
    h = ts.hour
    if h >= 22 or h < 7:
        return "Asia"
    if h < 12:
        return "London"
    return "NY"


def precompute_m15(m15: pd.DataFrame) -> pd.DataFrame:
    """Attach causal rolling columns + trading-day to a copy of m15."""
    m = m15.copy()
    prev_close = m["close"].shift(1)
    tr = pd.concat([(m["high"] - m["low"]),
                    (m["high"] - prev_close).abs(),
                    (m["low"] - prev_close).abs()], axis=1).max(axis=1)
    m["atr"] = tr.rolling(ATR_N).mean()
    m["ema_fast"] = m["close"].ewm(span=EMA_FAST, adjust=False).mean()
    m["ema_slow"] = m["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    m["ema_fast_slope"] = m["ema_fast"] - m["ema_fast"].shift(SLOPE_N)
    m["recent_ret"] = m["close"] - m["close"].shift(RET_N)
    # tick-volume: touch-bar volume vs a causal rolling baseline, and a
    # short/long volume trend. Backward-looking only (safe to read at the touch).
    vol_ma = m["volume"].rolling(VOL_N).mean()
    m["rel_volume"] = m["volume"] / vol_ma
    m["vol_trend"] = (m["volume"].rolling(VOL_FAST).mean()
                      / m["volume"].rolling(VOL_SLOW).mean())
    m["tday"] = trading_day(m.index)
    return m


def daily_refs(d1: pd.DataFrame) -> pd.DataFrame:
    """Per trading-day prior-day references (use ONLY completed earlier days)."""
    d = d1.copy()
    d["date"] = pd.to_datetime([ts.date() + timedelta(days=1) for ts in d1.index])
    d = d.sort_values("date")
    d["pdh"] = d["high"].shift(1)
    d["pdl"] = d["low"].shift(1)
    d["prev_close"] = d["close"].shift(1)
    d["whigh"] = d["high"].shift(1).rolling(WEEK_DAYS).max()
    d["wlow"] = d["low"].shift(1).rolling(WEEK_DAYS).min()
    return d.set_index("date")[["open", "pdh", "pdl", "prev_close", "whigh", "wlow"]]


def prior_level_prices(levels: pd.DataFrame, k: int = CONFLUENCE_K) -> dict:
    """
    Per trading-day, the sorted level prices of the prior `k` days (confluence
    reference). Uses ONLY strictly-earlier days, so it is leakage-safe (prior
    levels are known at today's open). Empty array for the first `k` days.
    """
    dates = sorted(levels["date"].unique())
    by_day = {d: np.sort(levels.loc[levels["date"] == d, "level_price"].to_numpy())
              for d in dates}
    out = {}
    for i, d in enumerate(dates):
        prior = dates[max(0, i - k):i]
        out[d] = (np.sort(np.concatenate([by_day[p] for p in prior]))
                  if prior else np.empty(0))
    return out


def _break_positions(closes: np.ndarray, level_by_step: dict, break_buf: float) -> dict:
    """First bar index at which each level is decisively closed-through (else -1)."""
    out = {}
    for k, L in level_by_step.items():
        if k > 0:
            mask = closes >= L + break_buf
        elif k < 0:
            mask = closes <= L - break_buf
        else:  # the open: a decisive close either side
            mask = (closes >= L + break_buf) | (closes <= L - break_buf)
        hits = np.flatnonzero(mask)
        out[k] = int(hits[0]) if hits.size else -1
    return out


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def build_features(m15: pd.DataFrame, d1: pd.DataFrame, levels: pd.DataFrame,
                   labels: pd.DataFrame, days=None) -> pd.DataFrame:
    """
    Return a feature DataFrame indexed by labels' index (one row per touch).
    `days` optionally restricts the per-day processing (used by the leak test);
    rolling stats are always precomputed on the full passed m15.
    """
    m = precompute_m15(m15)
    refs = daily_refs(d1)
    day_groups = dict(tuple(m.groupby("tday")))

    # prior touches of THIS level earlier today (causal; from labels themselves)
    order = labels.sort_values(["date", "step_number", "touch_time"])
    prior = order.groupby(["date", "step_number"]).cumcount()
    prior_touches = prior.reindex(labels.index)

    levels_g = dict(tuple(levels.groupby("date")))
    prior_levels = prior_level_prices(levels)      # cross-day confluence reference
    dates = labels["date"].unique() if days is None else days
    rows = {}

    for date in dates:
        day_touches = labels[labels["date"] == date]
        if day_touches.empty or date not in day_groups or date not in levels_g:
            continue
        df_day = day_groups[date]
        idx_pos = {ts: i for i, ts in enumerate(df_day.index)}

        lv_day = levels_g[date]
        day_open = float(lv_day["day_open"].iloc[0])
        step = step_size(day_open)
        break_buf = BREAK_MULT * step
        conf_tol = CONFLUENCE_TOL * step
        prior_arr = prior_levels.get(date, np.empty(0))
        level_by_step = dict(zip(lv_day["step_number"].astype(int),
                                 lv_day["level_price"].astype(float)))

        highs = df_day["high"].to_numpy()
        lows = df_day["low"].to_numpy()
        closes = df_day["close"].to_numpy()
        hi_cummax = np.maximum.accumulate(highs)
        lo_cummin = np.minimum.accumulate(lows)
        atr = df_day["atr"].to_numpy()
        ema_f = df_day["ema_fast"].to_numpy()
        ema_s = df_day["ema_slow"].to_numpy()
        ema_slope = df_day["ema_fast_slope"].to_numpy()
        recent_ret = df_day["recent_ret"].to_numpy()
        rel_vol = df_day["rel_volume"].to_numpy()
        vol_trend = df_day["vol_trend"].to_numpy()
        break_pos = _break_positions(closes, level_by_step, break_buf)

        r = refs.loc[date] if date in refs.index else None
        pdh = pdl = prev_close = whigh = wlow = np.nan
        if r is not None:
            pdh, pdl, prev_close, whigh, wlow = (r["pdh"], r["pdl"],
                                                 r["prev_close"], r["whigh"], r["wlow"])
        gap_steps = (day_open - prev_close) / step if pd.notna(prev_close) else np.nan

        for t in day_touches.itertuples():
            p = idx_pos.get(t.touch_time)
            if p is None:
                continue
            k = int(t.step_number)
            L = float(t.level_price)
            price = closes[p]

            prev_ref = closes[p - 1] if p > 0 else day_open
            approached_from_above = 1 if prev_ref >= L else -1

            bp = break_pos.get(k, -1)
            this_broke_earlier = (bp != -1 and bp < p)
            levels_broken_before = sum(1 for v in break_pos.values()
                                       if v != -1 and v < p)

            # cross-day confluence: nearest prior-day level + how many stack here
            if prior_arr.size:
                ins = int(np.searchsorted(prior_arr, L))
                near = []
                if ins < prior_arr.size:
                    near.append(prior_arr[ins] - L)
                if ins > 0:
                    near.append(L - prior_arr[ins - 1])
                dist_prior = min(abs(x) for x in near) / step
                lo = np.searchsorted(prior_arr, L - conf_tol, side="left")
                hi = np.searchsorted(prior_arr, L + conf_tol, side="right")
                conf_count = int(hi - lo)
            else:
                dist_prior, conf_count = np.nan, 0

            def per_step(x):
                return x / step

            rows[t.Index] = {
                # --- level identity (known at the day open) ---
                "abs_step": abs(k),
                "above_or_below_open": int(np.sign(k)),
                # --- day context at the touch ---
                "hours_since_open": p * 0.25,
                "session": session_of(t.touch_time),
                "day_of_week": pd.Timestamp(t.date).dayofweek,
                "bars_since_day_open": p,
                # --- volatility / size ---
                "step_size": step,
                "atr_m15": atr[p],
                "atr_steps": per_step(atr[p]),
                "range_so_far_steps": per_step(hi_cummax[p] - lo_cummin[p]),
                # --- volume (tick-volume up to the touch) ---
                "rel_volume": rel_vol[p],
                "vol_trend": vol_trend[p],
                # --- trend / momentum (up to the touch only) ---
                "price_vs_ema20_steps": per_step(price - ema_f[p]),
                "price_vs_ema50_steps": per_step(price - ema_s[p]),
                "ema20_slope_steps": per_step(ema_slope[p]),
                "recent_return_steps": per_step(recent_ret[p]),
                "approached_from_above": approached_from_above,
                # --- touch / state history (this day, before the touch) ---
                "is_first_touch": int(prior_touches[t.Index] == 0),
                "prior_touches_today": int(prior_touches[t.Index]),
                "levels_broken_before": levels_broken_before,
                "bars_since_this_level_broken": (p - bp) if this_broke_earlier else np.nan,
                # --- confluence (normalised in step-size units) ---
                "dist_round_5_steps": per_step(abs(L - round(L / 5) * 5)),
                "dist_round_10_steps": per_step(abs(L - round(L / 10) * 10)),
                "dist_round_25_steps": per_step(abs(L - round(L / 25) * 25)),
                "dist_round_50_steps": per_step(abs(L - round(L / 50) * 50)),
                "dist_pdh_steps": per_step(L - pdh) if pd.notna(pdh) else np.nan,
                "dist_pdl_steps": per_step(L - pdl) if pd.notna(pdl) else np.nan,
                "dist_whigh_steps": per_step(L - whigh) if pd.notna(whigh) else np.nan,
                "dist_wlow_steps": per_step(L - wlow) if pd.notna(wlow) else np.nan,
                "dist_nearest_prior_level_steps": dist_prior,
                "confluence_count": conf_count,
                # --- gap ---
                "gap_steps": gap_steps,
                # reference (known at open, handy downstream)
                "day_open": day_open,
            }

    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- #
# Leakage assertion
# --------------------------------------------------------------------------- #
def assert_no_leakage(m15, d1, levels, labels, full_feats, n_sample=80, seed=0):
    """Rebuild a sample of touches on data truncated at touch_time; require equality."""
    rng = np.random.default_rng(seed)
    sample = labels.sample(min(n_sample, len(labels)), random_state=seed)
    feat_cols = full_feats.columns
    mism = []
    for tid, row in sample.iterrows():
        tt = row["touch_time"]
        m_trunc = m15[m15.index <= tt]                       # <= touch bar only
        same_day = labels[(labels["date"] == row["date"]) &
                          (labels["touch_time"] <= tt)]       # earlier same-day touches
        one = build_features(m_trunc, d1, levels, same_day, days=[row["date"]])
        a = full_feats.loc[tid]
        b = one.loc[tid]
        for c in feat_cols:
            va, vb = a[c], b[c]
            if isinstance(va, float) or isinstance(vb, float):
                if not (pd.isna(va) and pd.isna(vb)) and not np.isclose(
                        float(va), float(vb), rtol=1e-9, atol=1e-9, equal_nan=True):
                    mism.append((tid, c, va, vb))
            elif va != vb:
                mism.append((tid, c, va, vb))
    assert not mism, f"LEAKAGE/MISMATCH in {len(mism)} feature(s): {mism[:5]}"
    return len(sample)


FEATURE_NOTES = [
    ("step_number", "ladder step −10..+10 (0 = open); the level's identity"),
    ("abs_step", "distance from the open in steps (|step_number|)"),
    ("type", "open / thick (even step) / thin (odd step)"),
    ("above_or_below_open", "+1 above the open, −1 below, 0 = open"),
    ("hours_since_open", "trading-hours elapsed since the 6 PM-NY open"),
    ("session", "Asia / London / NY at the touch (by UTC hour)"),
    ("day_of_week", "trading day's weekday (Mon=0)"),
    ("bars_since_day_open", "M15 bars elapsed since the day open"),
    ("step_size", "this day's ladder spacing in $ (from the formula)"),
    ("atr_m15", f"ATR over last {ATR_N} M15 bars, in $"),
    ("atr_steps", "ATR in step-size units (volatility vs the ladder)"),
    ("range_so_far_steps", "today's high−low so far, in step units"),
    ("rel_volume", f"touch-bar tick-volume vs its {VOL_N}-bar rolling average (>1 = busier than usual)"),
    ("vol_trend", f"volume {VOL_FAST}-bar vs {VOL_SLOW}-bar average (>1 = volume rising)"),
    ("price_vs_ema20_steps", f"(price − EMA{EMA_FAST} on M15) in step units"),
    ("price_vs_ema50_steps", f"(price − EMA{EMA_SLOW} on M15) in step units"),
    ("ema20_slope_steps", f"EMA{EMA_FAST} change over {SLOPE_N} M15 bars, in step units"),
    ("recent_return_steps", f"price change over last {RET_N} M15 bars, in step units"),
    ("approached_from_above", "+1 if price came from above (support test), −1 if below"),
    ("is_first_touch", "1 if this is the level's first touch today"),
    ("prior_touches_today", "count of earlier touches of THIS level today"),
    ("levels_broken_before", "how many of the 21 levels price already closed through today"),
    ("bars_since_this_level_broken", "M15 bars since this level broke earlier today (NaN if not)"),
    ("dist_round_5_steps", "distance to nearest multiple of 5, in step units"),
    ("dist_round_10_steps", "distance to nearest multiple of 10, in step units"),
    ("dist_round_25_steps", "distance to nearest multiple of 25, in step units"),
    ("dist_round_50_steps", "distance to nearest multiple of 50, in step units"),
    ("dist_pdh_steps", "signed distance to previous-day high, in step units"),
    ("dist_pdl_steps", "signed distance to previous-day low, in step units"),
    ("dist_whigh_steps", f"signed distance to prior-{WEEK_DAYS}-day high, in step units"),
    ("dist_wlow_steps", f"signed distance to prior-{WEEK_DAYS}-day low, in step units"),
    ("dist_nearest_prior_level_steps", f"distance to the nearest level from the prior {CONFLUENCE_K} days, in step units (small = confluence)"),
    ("confluence_count", f"# of prior-{CONFLUENCE_K}-day levels within {CONFLUENCE_TOL} step (stacked lines = stronger)"),
    ("gap_steps", "today's open − yesterday's close, in step units"),
    ("day_open", "the day's open price (reference; known at the open)"),
]


def write_note():
    lines = ["# Feature dictionary (Phase 4)\n",
             "One row per touch in `data/processed/dataset.parquet`. Every feature "
             "uses only data up to and including the touch bar.\n",
             "| feature | meaning |", "|---|---|"]
    lines += [f"| `{n}` | {d} |" for n, d in FEATURE_NOTES]
    lines.append("\n**Label:** `outcome` — 1 = reacted, 0 = broke (from Phase 3).")
    NOTE_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    m15 = pd.read_parquet(PROC / "m15.parquet")
    d1 = pd.read_parquet(PROC / "d1.parquet")
    levels = pd.read_parquet(PROC / "levels.parquet")
    labels = pd.read_parquet(PROC / "labels.parquet").reset_index(drop=True)

    feats = build_features(m15, d1, levels, labels)
    feats = feats.reindex(labels.index)            # align to label order

    dataset = pd.concat([labels[LABEL_COLS], feats], axis=1)
    dataset.to_parquet(OUT_PATH)
    write_note()

    # ---- leakage assertion ------------------------------------------------
    print("Running leakage assertion (rebuild on data truncated at touch_time)...")
    n_checked = assert_no_leakage(m15, d1, levels, labels, feats, n_sample=80)
    print(f"  PASSED — {n_checked} sampled touches reproduce identically; "
          "no feature used data after touch_time.\n")

    # ---- summary ----------------------------------------------------------
    feat_cols = [c for c in dataset.columns if c not in LABEL_COLS]
    print(f"dataset.parquet shape: {dataset.shape[0]:,} rows × {dataset.shape[1]} cols "
          f"({len(feat_cols)} features + {len(LABEL_COLS)} label/id cols)")
    print(f"saved -> {OUT_PATH.relative_to(ROOT)}   note -> {NOTE_PATH.name}")

    print("\n5 sample rows (subset of columns):")
    show = dataset[["date", "step_number", "type", "outcome", "abs_step",
                    "session", "atr_steps", "approached_from_above",
                    "is_first_touch", "levels_broken_before", "dist_pdh_steps"]].head()
    with pd.option_context("display.width", 170, "display.max_columns", 20,
                           "display.float_format", lambda x: f"{x:,.3f}"):
        print(show.to_string())

    print("\n% missing per feature:")
    miss = (dataset[feat_cols].isna().mean() * 100).round(2)
    for c, v in miss.items():
        flag = "  <- structural (level not broken earlier)" \
            if c == "bars_since_this_level_broken" else ""
        if v > 0:
            print(f"  {c:32s} {v:6.2f}%{flag}")
    if (miss == 0).all():
        print("  (none)")
    else:
        print(f"  all other features: 0.00% missing")

    print("\nDone. Dataset + label assembled. NOT training yet — review then confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
