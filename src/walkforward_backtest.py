#!/usr/bin/env python3
"""
Phase 7 (harden) — walk-forward validation + realistic backtest.

PART A — Walk-forward (is the edge consistent, or one lucky split?)
    Expanding windows, strict time order, RETRAIN + RE-CALIBRATE every fold.
    The calibration set is carved from the END of each training window so the
    test year stays fully out-of-sample:
        fit 2020-2021 | calib 2022 | TEST 2023
        fit 2020-2022 | calib 2023 | TEST 2024
        fit 2020-2023 | calib 2024 | TEST 2025
        fit 2020-2024 | calib 2025 | TEST 2026
    Per fold: ROC-AUC, PR-AUC, Brier, hit-rate at P(react) >= .70 and >= .60.
    All test-fold predictions are stitched into one out-of-sample (OOS) series.

PART B — Backtest on the OOS predictions (does confidence pay after costs?)
    At a touch with P(react) >= THRESHOLD, trade betting the level HOLDS:
      long  if price came DOWN into it (support),  short if it came UP (resistance)
      entry = level price ; stop = just beyond by BREAK_BUFFER (0.25 step)
      target = 1 step away  (also 1.5R and 2R variants)
    Bar-by-bar fills on M15; if a bar spans both stop and target -> assume STOP
    (conservative). Unresolved by the 17:00-NY day end -> exit at the day's close.
    Costs: spread+slippage, default $0.30 round-trip. P&L normalised in R
    (R = the risk = BREAK_BUFFER) so it is comparable across the 2020-2026 price
    regime. Compares high-confidence vs ALL touches vs RANDOM selection.

Outputs (reports/): walkforward_metrics.csv, oos_predictions.parquet,
    backtest_summary.csv, backtest_trades.parquet, equity_curve.png

Usage:
    python3 src/walkforward_backtest.py
    python3 src/walkforward_backtest.py --cost 0.30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib  # noqa: F401  (kept for parity / optional model dump)
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_config import lgbm_params, monotone_list  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TARGET = "outcome"
DROP = ["date", "touch_time", "outcome", "level_price",
        "day_open", "step_size", "atr_m15"]
CATEGORICAL = ["type", "session", "day_of_week"]
DAY_START_TZ = "America/New_York"
DAY_START_HOUR = 17
BREAK_MULT = 0.25
SEED = 42
TEST_YEARS = [2023, 2024, 2025, 2026]


# --------------------------------------------------------------------------- #
# PART A — walk-forward
# --------------------------------------------------------------------------- #
def fit_calibrated(fit_df, calib_df, features, cats):
    """LightGBM early-stopped on calib, then isotonic-calibrated on calib."""
    def Xy(part):
        X = part[features].copy()
        for c in cats:
            X[c] = X[c].astype("category")
        return X, part[TARGET]

    Xf, yf = Xy(fit_df)
    Xc, yc = Xy(calib_df)
    params = lgbm_params()
    mc = monotone_list(features)
    if mc is not None:
        params["monotone_constraints"] = mc
    model = lgb.LGBMClassifier(**params)
    model.fit(Xf, yf, eval_set=[(Xc, yc)], eval_metric="auc",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    try:
        from sklearn.frozen import FrozenEstimator
        calib = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except Exception:
        calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calib.fit(Xc, yc)
    return calib, model.best_iteration_


def hit_rate(y, p, t):
    m = p >= t
    return (int(m.sum()), float(y[m].mean()) if m.sum() else np.nan)


def walk_forward(df, features, cats):
    rows, oos = [], []
    for ty in TEST_YEARS:
        fit_df = df[df["year"] < ty - 1]
        calib_df = df[df["year"] == ty - 1]
        test_df = df[df["year"] == ty]
        if test_df.empty or calib_df.empty or fit_df.empty:
            continue
        calib, best_it = fit_calibrated(fit_df, calib_df, features, cats)

        Xte = test_df[features].copy()
        for c in cats:
            Xte[c] = Xte[c].astype("category")
        p = calib.predict_proba(Xte)[:, 1]
        y = test_df[TARGET]

        n70, h70 = hit_rate(y, p, 0.70)
        n60, h60 = hit_rate(y, p, 0.60)
        rows.append({
            "fold": f"fit≤{ty-2}|cal{ty-1}|test{ty}", "test_year": ty,
            "n_test": len(test_df), "react%": y.mean(),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "hit>=.70": h70, "n>=.70": n70, "hit>=.60": h60, "n>=.60": n60,
            "best_iter": best_it,
        })
        o = test_df[["date", "touch_time", "level_price", "step_size",
                     "approached_from_above", TARGET]].copy()
        o["p_react"] = p
        o["test_year"] = ty
        oos.append(o)
    return pd.DataFrame(rows), pd.concat(oos, ignore_index=True)


# --------------------------------------------------------------------------- #
# PART B — backtest
# --------------------------------------------------------------------------- #
def trading_day(idx_utc):
    local = idx_utc.tz_convert(DAY_START_TZ)
    bump = np.where(local.hour >= DAY_START_HOUR, 1, 0)
    return pd.to_datetime(local.date) + pd.to_timedelta(bump, unit="D")


def strat_stats(net_R):
    net_R = pd.Series(net_R).dropna()
    if net_R.empty:
        return None
    wins = net_R[net_R > 0]
    losses = net_R[net_R <= 0]
    eq = net_R.cumsum()
    dd = (eq.cummax() - eq).max()
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    return {
        "n_trades": len(net_R), "win_rate": (net_R > 0).mean(),
        "avg_win_R": wins.mean() if len(wins) else 0.0,
        "avg_loss_R": losses.mean() if len(losses) else 0.0,
        "expectancy_R": net_R.mean(), "total_R": net_R.sum(),
        "profit_factor": pf, "max_dd_R": dd,
    }


def operating_points(trades, floor_trades=180):
    """
    Confidence-threshold sweep on the OOS backtest (1-step, net of costs).
    Recommends the threshold that MAXIMISES expectancy while keeping at least
    `floor_trades` trades over the period (~1/week) — the trading-precision
    operating point.
    """
    p = trades["p_react"].to_numpy()
    nr = trades["net_R_1step"]
    rows = []
    for thr in (0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80):
        s = nr[p >= thr].dropna()
        if s.empty:
            continue
        wins, losses = s[s > 0], s[s <= 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
        rows.append({"thresh": thr, "n_trades": len(s), "win_rate": (s > 0).mean(),
                     "expectancy_R": s.mean(), "profit_factor": pf, "total_R": s.sum()})
    tbl = pd.DataFrame(rows)
    elig = tbl[tbl["n_trades"] >= floor_trades]
    rec = (elig.loc[elig["expectancy_R"].idxmax()] if len(elig)
           else tbl.loc[tbl["expectancy_R"].idxmax()])
    return tbl, rec


def backtest(oos, cost, plot_path):
    # target variants in $: 1 step away; or 1.5R / 2R (R = buf = 0.25 step).
    oos = oos.copy()
    step = oos["step_size"].to_numpy()
    buf = BREAK_MULT * step
    variant_dist = {"1step": step, "1.5R": 1.5 * buf, "2R": 2.0 * buf}
    # fills simulated on M5 (matches the M5 reaction labels / your 5m-1m entry)
    oos = _simulate_all(oos, pd.read_parquet(PROC / "m5.parquet"),
                        variant_dist, cost)

    # --- strategy comparison on the 1-step target ---
    p = oos["p_react"].to_numpy()
    rng = np.random.default_rng(SEED)
    n70 = int((p >= 0.70).sum())
    rand_exp = []
    if n70:
        for _ in range(200):
            idx = rng.choice(len(oos), size=n70, replace=False)
            rand_exp.append(pd.Series(oos["net_R_1step"].to_numpy()[idx]).dropna().mean())
    strategies = {
        "ALL touches": oos["net_R_1step"],
        f"RANDOM (n={n70})": None,
        "CONF >= 0.65": oos.loc[p >= 0.65, "net_R_1step"],
        "CONF >= 0.70": oos.loc[p >= 0.70, "net_R_1step"],
    }
    summ = {}
    for name, s in strategies.items():
        if name.startswith("RANDOM"):
            summ[name] = {"n_trades": n70, "win_rate": np.nan, "avg_win_R": np.nan,
                          "avg_loss_R": np.nan, "expectancy_R": float(np.mean(rand_exp)),
                          "total_R": float(np.mean(rand_exp)) * n70,
                          "profit_factor": np.nan, "max_dd_R": np.nan}
        else:
            summ[name] = strat_stats(s)
    summary = pd.DataFrame(summ).T

    # --- target-variant comparison for CONF>=0.70 ---
    var_rows = {}
    for v in ("1step", "1.5R", "2R"):
        st = strat_stats(oos.loc[p >= 0.70, f"net_R_{v}"])
        if st:
            var_rows[v] = {k: st[k] for k in ("n_trades", "win_rate",
                                              "expectancy_R", "profit_factor",
                                              "max_dd_R")}
    variants = pd.DataFrame(var_rows).T

    # --- equity curves (1-step target), ordered in time ---
    oos_sorted = oos.sort_values("touch_time")
    _plot_equity(oos_sorted, plot_path)
    return oos, summary, variants


def _simulate_all(oos, m15, variant_dist, cost):
    m = m15.copy()
    m["tday"] = trading_day(m.index)
    day_groups = dict(tuple(m.groupby("tday")))
    day_pos = {d: {ts: i for i, ts in enumerate(g.index)} for d, g in day_groups.items()}
    variants = list(variant_dist.keys())
    out = {f"net_R_{v}": [] for v in variants}
    out.update({f"won_{v}": [] for v in variants})

    step_arr = oos["step_size"].to_numpy()
    dist_arr = {v: (variant_dist[v] if np.ndim(variant_dist[v]) else
                    np.full(len(oos), variant_dist[v])) for v in variants}

    for r, t in enumerate(oos.itertuples()):
        g = day_groups.get(t.date)
        pos = day_pos.get(t.date, {}).get(t.touch_time)
        entry, step = float(t.level_price), float(step_arr[r])
        buf = BREAK_MULT * step
        long = t.approached_from_above == 1
        if g is None or pos is None or buf <= 0:
            for v in variants:
                out[f"net_R_{v}"].append(np.nan); out[f"won_{v}"].append(np.nan)
            continue
        highs = g["high"].to_numpy()[pos + 1:]
        lows = g["low"].to_numpy()[pos + 1:]
        last_close = g["close"].to_numpy()[-1]
        stop = entry - buf if long else entry + buf
        s_hit_idx = (np.flatnonzero(lows <= stop) if long
                     else np.flatnonzero(highs >= stop))
        si = s_hit_idx[0] if s_hit_idx.size else 10**9
        for v in variants:
            tdist = float(dist_arr[v][r])
            target = entry + tdist if long else entry - tdist
            t_hit = (np.flatnonzero(highs >= target) if long
                     else np.flatnonzero(lows <= target))
            ti = t_hit[0] if t_hit.size else 10**9
            if si == ti == 10**9:
                gross = (last_close - entry) if long else (entry - last_close)
                win = gross > 0
            elif si <= ti:
                gross, win = -buf, False
            else:
                gross, win = +tdist, True
            out[f"net_R_{v}"].append((gross - cost) / buf)
            out[f"won_{v}"].append(bool(win))
    for k, vals in out.items():
        oos[k] = vals
    return oos


def _plot_equity(oos_sorted, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = oos_sorted["p_react"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, mask in [("ALL touches", np.ones(len(oos_sorted), bool)),
                       ("CONF >= 0.65", p >= 0.65), ("CONF >= 0.70", p >= 0.70)]:
        s = oos_sorted.loc[mask, ["touch_time", "net_R_1step"]].dropna()
        if len(s):
            ax.plot(s["touch_time"], s["net_R_1step"].cumsum(), label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Out-of-sample equity (1-step target, net of costs)")
    ax.set_ylabel("cumulative R"); ax.set_xlabel("date"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.30,
                    help="Round-trip spread+slippage in $ (default 0.30).")
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_parquet(PROC / "dataset.parquet").sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    features = [c for c in df.columns if c not in DROP and c != "year"]
    cats = [c for c in CATEGORICAL if c in features]

    print("PART A — Walk-forward (retrain + recalibrate per fold)\n")
    wf, oos = walk_forward(df, features, cats)
    cols = ["fold", "n_test", "react%", "roc_auc", "pr_auc", "brier",
            "hit>=.70", "n>=.70", "hit>=.60", "n>=.60", "best_iter"]
    with pd.option_context("display.width", 170, "display.float_format", lambda x: f"{x:.3f}"):
        print(wf[cols].to_string(index=False))
    avg = wf[["roc_auc", "pr_auc", "brier", "hit>=.70", "hit>=.60"]].agg(["mean", "std"])
    print("\nAcross folds (mean ± std):")
    for c in ["roc_auc", "pr_auc", "brier", "hit>=.70", "hit>=.60"]:
        print(f"  {c:10s}: {avg.loc['mean', c]:.3f} ± {avg.loc['std', c]:.3f}")
    wf.to_csv(REPORTS / "walkforward_metrics.csv", index=False)
    oos.to_parquet(REPORTS / "oos_predictions.parquet")
    print(f"\nStitched OOS predictions: {len(oos):,} touches "
          f"({oos['date'].min().date()} .. {oos['date'].max().date()}) "
          f"-> reports/oos_predictions.parquet")

    print(f"\nPART B — Backtest on OOS predictions (cost ${args.cost:.2f} round-trip)\n")
    trades, summary, variants = backtest(oos, args.cost, REPORTS / "equity_curve.png")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print("Strategy comparison (1-step target, R = risk = 0.25 step, net of costs):")
        print(summary.to_string())
        print("\nTarget-variant comparison for CONF >= 0.70:")
        print(variants.to_string())

    tbl, rec = operating_points(trades)
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print("\nOperating-point sweep (1-step target, net of costs):")
        print(tbl.to_string(index=False))
    print(f"\nRECOMMENDED operating threshold (max expectancy, >=180 trades): "
          f"P(react) >= {rec['thresh']:.2f}  ->  {int(rec['n_trades'])} trades, "
          f"win {rec['win_rate']:.1%}, expectancy {rec['expectancy_R']:+.2f}R, "
          f"PF {rec['profit_factor']:.2f}")
    tbl.to_csv(REPORTS / "operating_points.csv", index=False)
    summary.to_csv(REPORTS / "backtest_summary.csv")
    trades.to_parquet(REPORTS / "backtest_trades.parquet")
    print(f"\nSaved -> reports/ (walkforward_metrics.csv, oos_predictions.parquet, "
          f"backtest_summary.csv, backtest_trades.parquet, equity_curve.png)")
    print("\nDone (Phase 7 harden). Review walk-forward stability + backtest edge, then stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
