#!/usr/bin/env python3
"""
Convert a TradingView Pine-log export of FOREX.com daily opens into the
data/external/forexcom_daily_opens.csv format used by build_levels.py.

Pine-log input columns:
    Date     : IST timestamp when the log fired (e.g. 2021-01-04T04:30:00.000+05:30)
    Message  : "<session-open NY date>,<open price>"   e.g.  "2021-01-03,1898.750"

Output columns (one row per trading/chart day):
    date          : chart candle date = session-open NY date + 1 day
                    (this is exactly the key build_levels.py looks up: the D1
                    session-open NY date + 1)
    open_forexcom : the FOREX.com daily open price

Usage:
    python3 src/convert_pine_opens.py "<pine.csv>" --out data/external/forexcom_daily_opens.csv
    python3 src/convert_pine_opens.py "<pine.csv>" --out NEW.csv --validate data/external/forexcom_daily_opens.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def convert(pine_path: Path) -> pd.DataFrame:
    pine = pd.read_csv(pine_path)
    msg = pine["Message"].astype(str).str.strip().str.strip('"')
    parts = msg.str.split(",", expand=True)
    session_date = pd.to_datetime(parts[0].str.strip())
    open_val = parts[1].str.strip().astype(float)
    chart_day = (session_date + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    out = (pd.DataFrame({"date": chart_day, "open_forexcom": open_val})
           .dropna()
           .drop_duplicates("date", keep="last")
           .sort_values("date")
           .reset_index(drop=True))
    return out


def validate(new: pd.DataFrame, existing_path: Path, tol: float = 0.05) -> None:
    old = pd.read_csv(existing_path)
    m = new.merge(old, on="date", suffixes=("_new", "_old"))
    diff = (m["open_forexcom_new"] - m["open_forexcom_old"]).abs()
    n_over = len(m)
    n_bad = int((diff > tol).sum())
    print(f"  validation vs {existing_path.name}: {n_over:,} overlapping days, "
          f"max |diff| = {diff.max():.4f}, mismatches >{tol} = {n_bad}")
    if n_bad:
        print("  WORST 5 mismatches:")
        worst = m.assign(diff=diff).sort_values("diff", ascending=False).head(5)
        for _, r in worst.iterrows():
            print(f"    {r['date']}: new {r['open_forexcom_new']:.3f} "
                  f"vs old {r['open_forexcom_old']:.3f}  (Δ{r['diff']:.3f})")
    else:
        print("  ✓ overlap matches the known-good file — mapping is correct.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pine", help="Pine-log CSV (Date,Message).")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--validate", default=None,
                    help="Existing forexcom_daily_opens.csv to validate the "
                         "overlap against (does not modify it).")
    args = ap.parse_args()

    new = convert(Path(args.pine))
    print(f"Converted {len(new):,} daily opens: {new['date'].iloc[0]} .. "
          f"{new['date'].iloc[-1]}")
    if args.validate and Path(args.validate).exists():
        validate(new, Path(args.validate))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
