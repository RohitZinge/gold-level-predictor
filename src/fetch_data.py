#!/usr/bin/env python3
"""
Phase 1 — Data collection.

Download XAUUSD M5 OHLC from Dukascopy and save the raw history to
data/raw/xauusd_m5.csv.

PRICE TYPE (--price, default "bid")
    Dukascopy (via the `dukascopy-node` CLI) serves `bid` and `ask` candles.
    - "bid" (default): matches a TradingView / FOREX.com gold chart, which plots
      bid — so the day-open and your 21 levels line up with what you see.
    - "ask": the ask side.
    - "mid": download both and average column-wise, mid = (bid + ask) / 2 for
      O/H/L/C and the mean of the two volumes. Sits ~half-a-spread above bid
      (≈ $1.5 on gold at the thin 6 PM reopen), so it reads high vs a bid chart.

The download is done in YEARLY chunks so a single network call never has to
pull six years at once (avoids timeouts). Each chunk is cached as a temp CSV in
data/raw/_tmp/, so re-running resumes (and switching --price re-uses cached
bid/ask sides without re-downloading).

Requirements:
    - Node.js + npx on PATH (you already have Node v22). The first run will
      auto-fetch the `dukascopy-node` package via `npx --yes`.
    - Python: pandas.

Usage:
    python3 src/fetch_data.py                       # last ~6 years, bid
    python3 src/fetch_data.py --price mid
    python3 src/fetch_data.py --from 2020-01-01 --to 2026-06-18
    python3 src/fetch_data.py --force               # ignore cache, re-download all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / defaults
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TMP_DIR = RAW_DIR / "_tmp"
DEFAULT_OUT = RAW_DIR / "xauusd_m5.csv"

INSTRUMENT = "xauusd"
TIMEFRAME = "m5"
PRICE_TYPES = ("bid", "ask")  # downloaded then averaged into mid

# OHLC columns we average; volume handled separately.
OHLC_COLS = ["open", "high", "low", "close"]


def default_from_date() -> str:
    """~6 years back from today, clamped to a clean year start."""
    today = date.today()
    return f"{today.year - 6}-01-01"


# --------------------------------------------------------------------------- #
# Dukascopy CLI download (one price side, one date range)
# --------------------------------------------------------------------------- #
def download_chunk(
    price_type: str,
    date_from: str,
    date_to: str,
    out_path: Path,
    *,
    batch_size: int,
    batch_pause: int,
    retries: int,
    retry_pause: int,
) -> Path:
    """
    Download a single [date_from, date_to) M5 chunk for one price side
    (bid/ask) into out_path via the dukascopy-node CLI. `date_to` is exclusive.

    Returns the path to the written CSV. Raises on failure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dukascopy-node appends ".csv" to the -fn value, so strip the suffix.
    file_stem = out_path.with_suffix("").name

    cmd = [
        "npx", "--yes", "dukascopy-node@latest",
        "-i", INSTRUMENT,
        "-from", date_from,
        "-to", date_to,
        "-t", TIMEFRAME,
        "-p", price_type,
        "-v",                         # include volumes
        "-f", "csv",
        "-dir", str(out_path.parent),
        "-fn", file_stem,
        "-bs", str(batch_size),
        "-bp", str(batch_pause),
        "-r", str(retries),
        "-rp", str(retry_pause),
        "-re",                        # retry on empty (0-byte) responses
        "-fr",                        # don't abort if some artifacts stay bad
    ]

    print(f"    $ dukascopy-node -p {price_type} {date_from}..{date_to}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        sys.stderr.write(result.stdout[-2000:] + "\n")
        sys.stderr.write(result.stderr[-2000:] + "\n")
        raise RuntimeError(
            f"dukascopy-node failed for {price_type} {date_from}..{date_to} "
            f"(exit {result.returncode})"
        )
    return out_path


def load_side_csv(path: Path) -> pd.DataFrame:
    """Read a dukascopy CSV (timestamp ms + OHLCV), indexed by timestamp."""
    return pd.read_csv(path).set_index("timestamp")


def _side_path(year: int, side: str) -> Path:
    return TMP_DIR / f"{INSTRUMENT}_{TIMEFRAME}_{year}_{side}.csv"


# --------------------------------------------------------------------------- #
# Per-year orchestration
# --------------------------------------------------------------------------- #
def fetch_year(
    year: int,
    date_from: str,
    date_to: str,
    *,
    price: str,
    force: bool,
    cli_kwargs: dict,
) -> pd.DataFrame:
    """
    Get one [date_from, date_to) window for the requested `price` type and
    return a tidy DataFrame: timestamp(ms), open, high, low, close, volume.

    price="bid"/"ask" downloads that side directly; price="mid" downloads both
    bid and ask and averages them column-wise. Already-downloaded chunks in
    data/raw/_tmp/ are reused (so switching price type re-uses cached sides).
    """
    sides = ("bid", "ask") if price == "mid" else (price,)
    frames = {}
    for side in sides:
        side_path = _side_path(year, side)
        if side_path.exists() and side_path.stat().st_size > 0 and not force:
            print(f"    cached: {side_path.name}")
        else:
            download_chunk(side, date_from, date_to, side_path, **cli_kwargs)
        frames[side] = load_side_csv(side_path)

    if price != "mid":
        out = frames[price]
        return out.reset_index() if not out.empty else out.reset_index().iloc[0:0]

    merged = frames["bid"].join(frames["ask"], how="inner",
                                lsuffix="_bid", rsuffix="_ask")
    if merged.empty:
        return merged.reset_index().iloc[0:0]
    mid = pd.DataFrame(index=merged.index)
    for col in OHLC_COLS:
        mid[col] = (merged[f"{col}_bid"] + merged[f"{col}_ask"]) / 2.0
    mid["volume"] = (merged["volume_bid"] + merged["volume_ask"]) / 2.0
    return mid.reset_index()  # timestamp back as a column


def yearly_windows(date_from: str, date_to: str):
    """Yield (year, chunk_from, chunk_to) tiling [date_from, date_to) by calendar year."""
    start = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date()
    for year in range(start.year, end.year + 1):
        chunk_from = max(start, date(year, 1, 1))
        chunk_to = min(end, date(year + 1, 1, 1))
        if chunk_from < chunk_to:
            yield year, chunk_from.isoformat(), chunk_to.isoformat()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="date_from", default=default_from_date(),
                        help="Start date yyyy-mm-dd (default: ~6 years ago).")
    parser.add_argument("--to", dest="date_to",
                        default=date.today().isoformat(),
                        help="End date yyyy-mm-dd, exclusive (default: today).")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output CSV path (default: data/raw/xauusd_m5.csv).")
    parser.add_argument("--price", choices=("bid", "ask", "mid"), default="bid",
                        help="Price type. 'bid' matches a TradingView/FOREX.com "
                             "chart (default); 'mid' = bid+ask average.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached chunks and re-download everything.")
    # CLI tuning (kept gentle; Dukascopy throttles aggressive batching).
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--batch-pause", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--retry-pause", type=int, default=1500)
    args = parser.parse_args()

    cli_kwargs = dict(
        batch_size=args.batch_size,
        batch_pause=args.batch_pause,
        retries=args.retries,
        retry_pause=args.retry_pause,
    )

    print(f"Fetching {INSTRUMENT.upper()} {TIMEFRAME.upper()} {args.price.upper()} "
          f"prices {args.date_from} .. {args.date_to} (yearly chunks)")
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for year, cfrom, cto in yearly_windows(args.date_from, args.date_to):
        print(f"\n[{year}] {cfrom} .. {cto}")
        df = fetch_year(year, cfrom, cto, price=args.price,
                        force=args.force, cli_kwargs=cli_kwargs)
        print(f"    -> {len(df):,} M5 {args.price} bars")
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("\nNo data downloaded.")
        return 1

    full = pd.concat(all_frames, ignore_index=True)
    full = full.drop_duplicates(subset="timestamp").sort_values("timestamp")

    # Convert the epoch-ms timestamp into an explicit ISO-8601 UTC datetime so
    # the raw file is unambiguous and timezone is never guessed downstream.
    full["datetime"] = pd.to_datetime(full["timestamp"], unit="ms", utc=True)
    full = full[["datetime"] + OHLC_COLS + ["volume"]]
    # Round to sane precision (gold quotes ~3 dp; mid adds one).
    full[OHLC_COLS] = full[OHLC_COLS].round(4)
    full["volume"] = full["volume"].round(4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)

    print(f"\nSaved {len(full):,} rows -> {out_path}")
    print(f"Range: {full['datetime'].min()}  ..  {full['datetime'].max()}")
    print("Done. (Run src/data_loader.py next to clean + build timeframes.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
