#!/usr/bin/env python3
"""
Phase 1 — Load, clean, and resample.

Reads the raw XAUUSD M5 mid-price CSV produced by src/fetch_data.py, cleans it,
resamples it up into M15 / H1 / H4 / D1, and saves each timeframe as parquet in
data/processed/. Prints a per-timeframe report (rows, date range, 5 sample rows,
data-quality warnings).

Cleaning steps (reported as it goes):
    - parse the datetime as UTC, sort ascending
    - drop duplicate timestamps
    - drop bad bars: high < low, or any OHLC price <= 0
    - report time gaps between consecutive M5 bars (weekend gaps flagged as
      expected; unexpected intraday gaps highlighted)

DAY-START BOUNDARY (must match your TradingView indicator)
----------------------------------------------------------
The D1 (daily) bars are NOT UTC midnight. They roll over at the FOREX daily
boundary used by your Pine `timeframe.change("D")`: 17:00 America/New_York
(5 PM NY). Each trading day runs 17:00 NY -> 17:00 NY the next day, and the
day's OPEN is the first M5 open at/after 17:00 NY. The 21 ladder levels are
built from that open, so this boundary must line up with your chart exactly.

    DAY_START_TZ   = "America/New_York"   # default
    DAY_START_HOUR = 17                   # 5 PM local rollover

This is handled timezone-aware so it stays correct across daylight-saving
changes: 17:00 NY is 21:00 UTC in summer (EDT) and 22:00 UTC in winter (EST),
and a session is 23 or 25 hours long on the two DST-transition days. Grouping
is done on the local wall clock, so those transitions are handled correctly.

Intraday timeframes (M5/M15/H1/H4) are always stored in UTC. Only the daily
bars use the NY boundary; their index is the session-open instant (17:00 NY).

Usage:
    python3 src/data_loader.py
    python3 src/data_loader.py --day-start-tz America/New_York --day-start-hour 17
    python3 src/data_loader.py --raw data/raw/xauusd_m5.csv
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "xauusd_m5.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# Daily boundary for the D1 bars (see module docstring). Must match the Pine
# `timeframe.change("D")` rollover on your gold chart: 5 PM New York.
DAY_START_TZ = "America/New_York"
DAY_START_HOUR = 17  # 17:00 local = 5 PM NY

OHLC_AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum"}

# Intraday timeframes are plain UTC resamples off the M5 base. The daily bar is
# handled separately (NY session boundary) -- see resample_daily_session().
TIMEFRAME_RULES = {
    "m15": "15min",
    "h1": "1h",
    "h4": "4h",
}

M5_FREQ = pd.Timedelta(minutes=5)


# --------------------------------------------------------------------------- #
# Load + clean
# --------------------------------------------------------------------------- #
def load_m5(raw_path: Path) -> pd.DataFrame:
    """Load the raw M5 CSV, clean it, and report what was removed + time gaps."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run `python3 src/fetch_data.py` first."
        )

    df = pd.read_csv(raw_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    n_raw = len(df)
    print(f"\n=== Cleaning M5 ({n_raw:,} raw rows) ===")

    # Sort by time (stable) so 'first'/'last' aggregations are meaningful.
    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)

    # 1) Duplicate timestamps -------------------------------------------------
    dup_mask = df["datetime"].duplicated(keep="first")
    n_dups = int(dup_mask.sum())
    df = df[~dup_mask]

    # 2) Bad bars: high < low, or any price <= 0 ------------------------------
    price_cols = ["open", "high", "low", "close"]
    bad_hl = df["high"] < df["low"]
    bad_zero = (df[price_cols] <= 0).any(axis=1)
    # Also flag bars where high/low don't bound open/close (corrupt OHLC).
    bad_bounds = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
    )
    bad_mask = bad_hl | bad_zero | bad_bounds
    n_bad = int(bad_mask.sum())
    df = df[~bad_mask].reset_index(drop=True)

    print(f"  duplicate timestamps removed : {n_dups:,}")
    print(f"  bad bars removed             : {n_bad:,} "
          f"(high<low={int(bad_hl.sum())}, price<=0={int(bad_zero.sum())}, "
          f"bad-bounds={int(bad_bounds.sum())})")
    print(f"  clean rows kept              : {len(df):,} "
          f"({n_raw - len(df):,} total removed)")

    report_gaps(df)
    return df.set_index("datetime")


def classify_gap(duration: pd.Timedelta, start_dow: int) -> str:
    """
    Classify an M5 gap by how the gold market actually closes:
      - 'daily_break' : the ~1h daily settlement break (≈55-70 min)
      - 'weekend'     : a multi-day close starting Fri or spanning Sat/Sun
      - 'holiday'     : another long close (> 2h) on a non-weekend day
      - 'minor'       : a short gap (thin liquidity); worth a glance only
    """
    mins = duration.total_seconds() / 60
    if 55 <= mins <= 70:
        return "daily_break"
    if duration > pd.Timedelta(days=1) or start_dow in (4, 5, 6):
        return "weekend"
    if duration > pd.Timedelta(hours=2):
        return "holiday"
    return "minor"


def report_gaps(df: pd.DataFrame) -> None:
    """Report time gaps between consecutive M5 bars, classified by cause."""
    dt = df["datetime"]
    deltas = dt.diff()
    gaps = deltas[deltas > M5_FREQ]  # anything bigger than one M5 step
    if gaps.empty:
        print("  time gaps                    : none")
        return

    gap_tbl = pd.DataFrame({
        "gap_start": dt.shift(1)[gaps.index],
        "gap_end": dt[gaps.index],
        "duration": gaps.values,
    })
    start_dow = gap_tbl["gap_start"].dt.dayofweek  # Mon=0 .. Sun=6
    gap_tbl["kind"] = [classify_gap(d, w)
                       for d, w in zip(gap_tbl["duration"], start_dow)]
    counts = gap_tbl["kind"].value_counts()

    print(f"  time gaps (> 5 min)          : {len(gap_tbl):,}")
    for kind, label in (("daily_break", "~1h daily settlement break (expected)"),
                        ("weekend", "weekend closes (expected)"),
                        ("holiday", "holiday closes (expected)"),
                        ("minor", "minor sub-hour gaps (thin liquidity)")):
        if kind in counts:
            print(f"      {label:42s}: {counts[kind]:,}")

    biggest = gap_tbl.sort_values("duration", ascending=False).head(5)
    print("  largest gaps:")
    for _, row in biggest.iterrows():
        print(f"    {row['gap_start']} -> {row['gap_end']}  "
              f"({row['duration']}, {row['kind']})")

    # The only thing worth a real warning: short gaps inside active hours that
    # are NOT the daily break (could hint at missing artifacts).
    minor = gap_tbl[gap_tbl["kind"] == "minor"]
    if len(minor):
        print(f"  NOTE: {len(minor):,} minor sub-hour gap(s) "
              f"(largest {minor['duration'].max()}); typically thin-liquidity, "
              f"not missing data.")


# --------------------------------------------------------------------------- #
# Resample
# --------------------------------------------------------------------------- #
def resample_tf(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample the clean M5 frame up to an intraday `rule` (stays in UTC)."""
    out = m5.resample(rule, label="left", closed="left").agg(OHLC_AGG)
    # Resampling creates empty buckets over weekends/holidays -> drop them.
    return out.dropna(subset=["open"])


def _session_start_dates(local_idx: pd.DatetimeIndex, hour: int) -> list:
    """
    Trading-day (session-start) date for each bar, on the local WALL CLOCK.

    A bar belongs to the session that began at the most recent `hour`:00 local:
    if its local hour is >= hour the session started today, otherwise yesterday.
    Working on the wall-clock date (not on fixed Timedeltas) keeps this correct
    across DST transitions -- `hour`=17 never lands in the 02:00 DST gap/overlap.
    """
    dates = local_idx.date          # datetime.date per bar, in local tz
    hours = local_idx.hour          # local wall-clock hour per bar
    return [d if h >= hour else d - timedelta(days=1)
            for d, h in zip(dates, hours)]


def _localize_at_hour(dates, tz: str, hour: int) -> pd.DatetimeIndex:
    """Turn an iterable of dates into tz-aware Timestamps at `hour`:00 local."""
    naive = pd.to_datetime(pd.Index(dates)) + pd.Timedelta(hours=hour)
    # hour=17 is never ambiguous/nonexistent under US DST (transitions at 02:00).
    return naive.tz_localize(tz, nonexistent="shift_forward", ambiguous=False)


def resample_daily_session(m5: pd.DataFrame, tz: str = DAY_START_TZ,
                           hour: int = DAY_START_HOUR) -> pd.DataFrame:
    """
    Build daily bars whose day runs `hour`:00 `tz` -> `hour`:00 `tz` next day
    (the forex / TradingView daily boundary, default 17:00 New York).

      open  = first M5 open at/after the boundary
      high  = max, low = min, close = last, volume = sum across the session

    The returned frame is indexed by the session-OPEN instant (e.g. the 17:00 NY
    timestamp), tz-aware in `tz`. DST-correct (grouping uses the local wall clock).
    """
    local_idx = m5.index.tz_convert(tz)
    session_dates = _session_start_dates(local_idx, hour)

    grouper = pd.Index(session_dates, name="session_date")
    daily = m5.groupby(grouper).agg(OHLC_AGG).dropna(subset=["open"])

    # Re-index by the actual session-open instant (date @ hour:00, localized).
    daily.index = _localize_at_hour(daily.index, tz, hour)
    daily.index.name = "datetime"
    return daily


def recent_complete_opens(daily: pd.DataFrame, last_data_ts: pd.Timestamp,
                          tz: str, hour: int, n: int = 5) -> pd.DataFrame:
    """
    Return the `n` most recent COMPLETE daily sessions (the in-progress current
    session, whose 17:00 close has not yet passed, is excluded). Columns:
    session_open_NY, session_close_NY, open.
    """
    open_dates = daily.index.tz_convert(tz).date
    closes = _localize_at_hour([d + timedelta(days=1) for d in open_dates], tz, hour)
    complete = closes <= last_data_ts            # instant comparison (tz-aware)

    out = daily.loc[complete, ["open"]].copy()
    out.insert(0, "session_close_NY",
               closes[complete].tz_convert(tz).strftime("%Y-%m-%d %H:%M %Z"))
    out.insert(0, "session_open_NY",
               out.index.tz_convert(tz).strftime("%Y-%m-%d %H:%M %Z"))
    return out.tail(n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Report + save
# --------------------------------------------------------------------------- #
def report_and_save(name: str, df: pd.DataFrame) -> None:
    out_path = PROCESSED_DIR / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    idx = df.index
    print(f"\n===== {name.upper()} =====")
    print(f"  rows       : {len(df):,}")
    print(f"  date range : {idx.min()}  ..  {idx.max()}  (tz={idx.tz})")
    print(f"  saved      : {out_path.relative_to(ROOT)}")
    # Quality checks on the resampled frame.
    warnings = []
    if df.index.has_duplicates:
        warnings.append("duplicate index timestamps present")
    if (df["high"] < df["low"]).any():
        warnings.append("high < low in some bars")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        warnings.append("non-positive prices present")
    n_nan = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if n_nan:
        warnings.append(f"{n_nan} bars with NaN OHLC")
    print(f"  warnings   : {', '.join(warnings) if warnings else 'none'}")
    print("  sample (5 rows):")
    with pd.option_context("display.width", 120,
                           "display.max_columns", 10,
                           "display.float_format", lambda x: f"{x:,.4f}"):
        print(df.head().to_string())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(DEFAULT_RAW),
                        help="Raw M5 CSV path (default: data/raw/xauusd_m5.csv).")
    parser.add_argument("--day-start-tz", default=DAY_START_TZ,
                        help="Timezone of the daily rollover "
                             f"(default: {DAY_START_TZ}).")
    parser.add_argument("--day-start-hour", type=int, default=DAY_START_HOUR,
                        help="Local hour of the daily rollover "
                             f"(default: {DAY_START_HOUR} = 5 PM).")
    args = parser.parse_args()

    print(f"Daily boundary: {args.day_start_hour:02d}:00 {args.day_start_tz} "
          f"-> {args.day_start_hour:02d}:00 next day")
    m5 = load_m5(Path(args.raw))

    # Save the cleaned M5 base, then the intraday resamples (UTC).
    report_and_save("m5", m5)
    for name, rule in TIMEFRAME_RULES.items():
        report_and_save(name, resample_tf(m5, rule))

    # Daily bars on the NY 5 PM session boundary.
    d1 = resample_daily_session(m5, args.day_start_tz, args.day_start_hour)
    report_and_save("d1", d1)

    # Comparison table: opens of the most recent COMPLETE sessions, to check
    # against the TradingView chart's open / yellow line.
    recent = recent_complete_opens(d1, m5.index.max(),
                                   args.day_start_tz, args.day_start_hour, n=5)
    print("\n===== DAILY OPENS — 5 most recent COMPLETE sessions =====")
    print("  (compare each 'open' to your TradingView open / yellow line)")
    with pd.option_context("display.width", 140, "display.max_columns", 10,
                           "display.float_format", lambda x: f"{x:,.3f}"):
        print(recent.to_string(index=False))

    print("\nDone. Timeframes written to data/processed/ "
          "(m5, m15, h1, h4, d1). Levels NOT built yet — confirm the opens above "
          "match your chart first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
