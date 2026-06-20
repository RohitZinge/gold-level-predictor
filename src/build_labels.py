#!/usr/bin/env python3
"""
Phase 3 — touches + labels (the answer key).  LABELS ONLY — no features here.

Matches the way the levels are actually traded: watch the 15-min, confirm the
reaction on the 5-min.

  TOUCH  (on M15): a level "comes into play" when a 15-min bar's range reaches it
         within TOUCH_BUFFER (low-buf <= level <= high+buf). A touch is an ONSET
         (15-min bar entering the zone after being outside), so a multi-bar hug =
         one touch and a genuine later retest = a new touch.

  RESOLUTION (on M5): from the touch forward, the react/break outcome is judged on
         5-MINUTE bars (finer than the 15-min touch):
           broke (0)   : an M5 bar CLOSES beyond the level by BREAK_BUFFER on the
                         through side (a committed 5-min close-through, not a wick).
           reacted (1) : the AWAY-side M5 extreme reaches REACT_THRESHOLD beyond
                         the level, with no earlier break-close. Only on M5 bars
                         AFTER the first M5 bar that reached the level (the touch
                         bar still holds the approach move).
         Earliest event wins; a same-bar tie counts as a break. Unresolved by the
         17:00-NY day end -> DROP the touch (ambiguous).

APPROACH SIDE (from the M15 close before the touch, or the day open for the first
  bar): ref >= level -> tested as SUPPORT (came from above); else RESISTANCE.

GOLDEN RULE: only the label may look at bars AFTER the touch. (Features, Phase 4,
may not.)

THRESHOLDS (in units of that day's step size; tune via CLI):
    TOUCH_BUFFER 0.10   REACT_THRESHOLD 1.00   BREAK_BUFFER 0.25

Saves data/processed/labels.parquet, one row per touch:
    date, step_number, type, level_price, touch_time (M15), outcome (1/0)

Usage:
    python3 src/build_labels.py
    python3 src/build_labels.py --react-threshold 1.0 --break-buffer 0.25
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levels import step_size  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M15_PATH = ROOT / "data" / "processed" / "m15.parquet"
M5_PATH = ROOT / "data" / "processed" / "m5.parquet"
LEVELS_PATH = ROOT / "data" / "processed" / "levels.parquet"
OUT_PATH = ROOT / "data" / "processed" / "labels.parquet"

DAY_START_TZ = "America/New_York"
DAY_START_HOUR = 17

SAVE_COLS = ["date", "step_number", "type", "level_price", "touch_time", "outcome"]


def trading_day(idx_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map UTC bar timestamps to their trading-day label (same as levels.date)."""
    local = idx_utc.tz_convert(DAY_START_TZ)
    bump = np.where(local.hour >= DAY_START_HOUR, 1, 0)  # evening bars -> next day
    return pd.to_datetime(local.date) + pd.to_timedelta(bump, unit="D")


def _first(mask: np.ndarray, start: int) -> int:
    """Index of first True at position >= start, or a large sentinel."""
    hits = np.flatnonzero(mask[start:])
    return hits[0] + start if hits.size else 10**9


def label_day(levels_day, m15, m5, touch_mult, react_mult, break_mult):
    """Touches detected on M15, react/break resolved on M5. Returns rich records."""
    day_open = float(levels_day["day_open"].iloc[0])
    step = step_size(day_open)
    touch_buf, react_thr, break_buf = (touch_mult * step,
                                       react_mult * step, break_mult * step)

    # --- M15 (touch detection) ---
    h15, l15, c15 = (m15["high"].to_numpy(), m15["low"].to_numpy(),
                     m15["close"].to_numpy())
    t15 = m15.index

    # --- M5 (resolution) ---
    h5, l5, c5 = (m5["high"].to_numpy(), m5["low"].to_numpy(),
                  m5["close"].to_numpy())
    t5 = m5.index
    t5_int = t5.asi8                 # UTC ns, for searchsorted by touch time
    T5 = len(m5)

    rows, n_onsets = [], 0
    for lv in levels_day.itertuples(index=False):
        L = float(lv.level_price)
        in15 = (L >= l15 - touch_buf) & (L <= h15 + touch_buf)
        if not in15.any():
            continue
        prev_out = np.r_[True, ~in15[:-1]]
        onsets = np.flatnonzero(in15 & prev_out)
        n_onsets += onsets.size

        in5 = (L >= l5 - touch_buf) & (L <= h5 + touch_buf)
        broke_sup = c5 <= L - break_buf
        broke_res = c5 >= L + break_buf
        react_sup = h5 >= L + react_thr
        react_res = l5 <= L - react_thr

        for i in onsets:
            ref = c15[i - 1] if i > 0 else day_open
            support = ref >= L
            # first M5 bar at/after the M15 touch bar that reaches the level
            s = int(np.searchsorted(t5_int, t15[i].value, side="left"))
            j0 = _first(in5, s)
            if j0 >= T5:
                continue                      # no M5 coverage -> drop
            break_arr = broke_sup if support else broke_res
            react_arr = react_sup if support else react_res
            bj = _first(break_arr, j0)        # break may hit on the M5 touch bar
            rj = _first(react_arr, j0 + 1)    # react only AFTER the M5 touch bar
            if bj == rj == 10**9:
                continue                      # ambiguous -> drop
            if bj <= rj:
                outcome, resolve, trig = 0, bj, c5[bj]
            else:
                outcome, resolve, trig = 1, rj, (h5[rj] if support else l5[rj])
            rows.append({
                "date": levels_day["date"].iloc[0],
                "step_number": int(lv.step_number), "type": lv.type,
                "level_price": round(L, 4), "touch_time": t15[i],
                "outcome": outcome,
                "side": "support" if support else "resistance",
                "resolve_time": t5[resolve], "resolve_price": round(float(trig), 3),
                "day_open": round(day_open, 3), "step_size": round(step, 3),
            })
    return rows, n_onsets


def build_labels(m15, m5, levels, touch_mult, react_mult, break_mult):
    m15 = m15.copy(); m15["tday"] = trading_day(m15.index)
    m5 = m5.copy(); m5["tday"] = trading_day(m5.index)
    m15_by_day = dict(tuple(m15.groupby("tday")))
    m5_by_day = dict(tuple(m5.groupby("tday")))
    levels_by_day = dict(tuple(levels.groupby("date")))

    all_rows, total_onsets = [], 0
    for tday, lv_day in levels_by_day.items():
        d15, d5 = m15_by_day.get(tday), m5_by_day.get(tday)
        if d15 is None or d5 is None:
            continue
        rows, n = label_day(lv_day, d15, d5, touch_mult, react_mult, break_mult)
        all_rows.extend(rows)
        total_onsets += n
    return pd.DataFrame(all_rows), total_onsets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--touch-buffer", type=float, default=0.10)
    p.add_argument("--react-threshold", type=float, default=1.0)
    p.add_argument("--break-buffer", type=float, default=0.25)
    args = p.parse_args()

    for path in (M15_PATH, M5_PATH, LEVELS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (run data_loader + build_levels).")
    m15 = pd.read_parquet(M15_PATH)
    m5 = pd.read_parquet(M5_PATH)
    levels = pd.read_parquet(LEVELS_PATH)

    print(f"Thresholds (x step): touch={args.touch_buffer} react={args.react_threshold} "
          f"break={args.break_buffer}  | touch=M15, resolve=M5")
    rich, total_onsets = build_labels(m15, m5, levels,
                                      args.touch_buffer, args.react_threshold,
                                      args.break_buffer)
    rich[SAVE_COLS].to_parquet(OUT_PATH)

    n = len(rich)
    n_react = int((rich["outcome"] == 1).sum())
    n_break = int((rich["outcome"] == 0).sum())
    dropped = total_onsets - n
    print(f"\nSaved {n:,} touches -> {OUT_PATH.relative_to(ROOT)}")
    print(f"  reacted (1): {n_react:,}  ({n_react/n:6.2%})")
    print(f"  broke   (0): {n_break:,}  ({n_break/n:6.2%})")
    print(f"  dropped (ambiguous): {dropped:,} of {total_onsets:,} "
          f"({dropped/total_onsets:.2%})")
    print(f"  days with touches: {rich['date'].nunique():,}  "
          f"| avg touches/day: {n/max(rich['date'].nunique(),1):.1f}")

    last_day = rich["date"].max()
    recent = [d for d in sorted(rich["date"].unique()) if d != last_day]
    day = next(d for d in reversed(recent) if (rich["date"] == d).sum() >= 8)
    day_all = rich[rich["date"] == day].sort_values("touch_time")
    ex = day_all.head(8)
    print(f"\n===== WORKED EXAMPLE — {pd.Timestamp(day).date()} "
          f"(open {ex['day_open'].iloc[0]:,.2f}, step ${ex['step_size'].iloc[0]:,.2f}; "
          f"{len(day_all)} touches, first {len(ex)} shown) =====")
    show = ex[["touch_time", "step_number", "type", "level_price",
               "side", "resolve_time", "resolve_price", "outcome"]].copy()
    show["touch_time"] = show["touch_time"].dt.tz_convert(DAY_START_TZ).dt.strftime("%m-%d %H:%M")
    show["resolve_time"] = show["resolve_time"].dt.tz_convert(DAY_START_TZ).dt.strftime("%m-%d %H:%M")
    show["outcome"] = show["outcome"].map({1: "reacted", 0: "broke"})
    with pd.option_context("display.width", 160, "display.max_columns", 12,
                           "display.float_format", lambda x: f"{x:,.3f}"):
        print(show.to_string(index=False))
    print("\nDone. Labels saved (M15 touch, M5 reaction).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
