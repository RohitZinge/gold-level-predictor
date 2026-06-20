#!/usr/bin/env python3
"""
Phase 2 (finish) — build the 21 ladder levels for every trading day.

Reads data/processed/d1.parquet (daily bars on the 5 PM-NY forex boundary),
applies get_day_levels() from src/levels.py to each day's OPEN, and saves one
row per level per day to data/processed/levels.parquet:

    date         trading-day label (matches your TradingView candle date)
    day_open     that day's open price (the yellow centre line)
    step_number  -10 .. +10  (0 = open)
    type         "open" (step 0) | "thick" (even steps) | "thin" (odd steps)
    level_price  the level's price

DATE LABEL
    A trading day runs 17:00 NY -> 17:00 NY next day, and your chart (IST) dates
    each candle by its CLOSE day. E.g. the session that opens 18:00 NY on Jun 15
    is your "16 Jun" candle. So `date` = the session's close date = the NY
    session-open date + 1, which lines up with what you see on TradingView.

Usage:
    python3 src/build_levels.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Make src/ importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from levels import get_day_levels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D1_PATH = ROOT / "data" / "processed" / "d1.parquet"
OUT_PATH = ROOT / "data" / "processed" / "levels.parquet"
FX_OPENS = ROOT / "data" / "external" / "forexcom_daily_opens.csv"

COLUMNS = ["date", "day_open", "step_number", "type", "level_price"]


def build_levels(d1: pd.DataFrame, fx_opens: dict | None = None,
                 max_gap: float = 5.0) -> tuple[pd.DataFrame, dict]:
    """
    One row per level per day, using get_day_levels() on each day's open.

    If `fx_opens` (a {trading_day Timestamp -> FOREX.com open} map) is given, the
    levels are anchored to YOUR exact chart opens instead of Dukascopy's. Days
    are then DROPPED when (a) there is no FOREX.com open, or (b) the FOREX.com and
    Dukascopy opens disagree by more than `max_gap` $ — because on those days the
    Dukascopy intraday bars (used for touches/labels) don't line up with the
    FOREX.com level grid, so the labels would be unreliable.
    """
    rows = []
    stats = {"kept": 0, "drop_no_fx": 0, "drop_gap": 0, "duka": 0}
    for ts, day_open in d1["open"].items():
        duka_open = float(day_open)
        trading_day = pd.Timestamp(ts.date() + timedelta(days=1))   # = chart date

        if fx_opens is not None:
            fx = fx_opens.get(trading_day)
            if fx is None:
                stats["drop_no_fx"] += 1
                continue
            if abs(fx - duka_open) > max_gap:
                stats["drop_gap"] += 1
                continue
            use_open = float(fx)
            stats["kept"] += 1
        else:
            use_open = duka_open
            stats["duka"] += 1

        for lv in get_day_levels(use_open):
            rows.append((trading_day, use_open,
                         lv["step_number"], lv["type"], round(lv["price"], 4)))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["step_number"] = df["step_number"].astype("int8")
    df["type"] = df["type"].astype("category")
    return df, stats


def print_recent(df: pd.DataFrame, n_days: int = 3) -> None:
    """Print the full 21 levels for the n most recent days, side by side."""
    recent_dates = sorted(df["date"].unique())[-n_days:]
    sub = df[df["date"].isin(recent_dates)]

    # Pivot: rows = step_number (+10 at top), columns = each day's price.
    prices = (sub.pivot(index="step_number", columns="date", values="level_price")
                 .sort_index(ascending=False))
    prices.columns = [pd.Timestamp(c).date() for c in prices.columns]
    types = (sub.drop_duplicates("step_number").set_index("step_number")["type"]
                .reindex(prices.index))

    table = pd.concat([types.rename("type"), prices], axis=1)
    opens = sub.drop_duplicates("date").set_index("date")["day_open"]

    print(f"\n===== 21 LEVELS — {n_days} most recent days =====")
    print("  day_open (centre / yellow line):")
    for d in recent_dates:
        print(f"    {pd.Timestamp(d).date()} : {opens[d]:,.3f}")
    print("  (step +10 = highest price, step 0 = open, step -10 = lowest)\n")
    with pd.option_context("display.width", 140, "display.max_columns", 12,
                           "display.float_format", lambda x: f"{x:,.3f}"):
        print(table.to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-forexcom-opens", action="store_true",
                    help=f"Anchor levels to your exact opens in {FX_OPENS.name}.")
    ap.add_argument("--max-gap", type=float, default=5.0,
                    help="Drop days where FOREX.com vs Dukascopy open differ by "
                         "more than this many $ (default 5).")
    args = ap.parse_args()

    if not D1_PATH.exists():
        raise FileNotFoundError(f"{D1_PATH} not found. Run src/data_loader.py first.")
    d1 = pd.read_parquet(D1_PATH)

    fx_opens = None
    if args.use_forexcom_opens:
        if not FX_OPENS.exists():
            raise FileNotFoundError(f"{FX_OPENS} not found.")
        fxdf = pd.read_csv(FX_OPENS, parse_dates=["date"])
        fx_opens = dict(zip(fxdf["date"], fxdf["open_forexcom"]))
        print(f"Anchoring to FOREX.com opens ({len(fx_opens)} days), "
              f"max-gap=${args.max_gap}")

    levels, stats = build_levels(d1, fx_opens, args.max_gap)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    levels.to_parquet(OUT_PATH)

    n_days = levels["date"].nunique()
    print(f"Built levels for {n_days:,} days "
          f"-> {len(levels):,} rows ({len(levels)//max(n_days,1)} per day) "
          f"-> {OUT_PATH.relative_to(ROOT)}")
    if fx_opens is not None:
        print(f"  kept (FOREX.com): {stats['kept']:,} | "
              f"dropped no-FX: {stats['drop_no_fx']:,} | "
              f"dropped gap>${args.max_gap}: {stats['drop_gap']:,}")

    print_recent(levels, n_days=3)
    print("\nDone. Levels saved. NOT building labels yet — confirm the centre "
          "and a few levels match your TradingView chart first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
