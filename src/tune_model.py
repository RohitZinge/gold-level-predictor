#!/usr/bin/env python3
"""
Light hyperparameter search (random search) optimised for TRADING PRECISION.

HONEST PROTOCOL — tune ONLY on early data so every walk-forward test year
(2023-2026) stays untouched:
    fit on year <= 2021   |   validate (early-stop + objective) on 2022

OBJECTIVE — react-precision among the top `TOP_FRAC` most-confident validation
touches (a matched-coverage proxy for the high-confidence calls we actually
trade). Isotonic calibration is monotonic and so does NOT change the top-k
ranking, so we score on raw probabilities here for speed.

Writes the best tunable params to models/best_params.json, which
model_config.lgbm_params() then overlays onto the defaults.

Usage:
    python3 src/tune_model.py            # 30 trials
    python3 src/tune_model.py --trials 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_config import (BEST_PARAMS_PATH, CATEGORICAL, DATA, DEFAULT_PARAMS,
                          DROP, TARGET, TUNABLE_KEYS)

FIT_MAX_YEAR = 2021       # fit on <= this year
VAL_YEAR = 2022           # validate on this year (no walk-forward test year touched)
TOP_FRAC = 0.10           # objective: react-precision among top 10% most-confident
SEED = 42

# Random-search space (only TUNABLE_KEYS; biased to faster learning rates).
GRID = {
    "learning_rate": [0.02, 0.03, 0.05],
    "num_leaves": [15, 23, 31, 47, 63],
    "min_child_samples": [40, 60, 80, 120, 200],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_lambda": [1.0, 3.0, 5.0, 10.0, 20.0],
    "reg_alpha": [0.0, 1.0, 5.0],
    "min_split_gain": [0.0, 0.01],
}


def topk_precision(y, p, frac: float) -> float:
    """React-precision among the top `frac` highest-probability rows."""
    k = max(1, int(round(len(p) * frac)))
    order = np.argsort(p)[::-1][:k]
    return float(np.asarray(y)[order].mean())


def fit_score(params, Xf, yf, Xv, yv):
    """Train early-stopped on val, return (top-k precision, val AUC, best_iter)."""
    m = lgb.LGBMClassifier(**params)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150, verbose=False)])
    p = m.predict_proba(Xv)[:, 1]
    from sklearn.metrics import roc_auc_score
    return topk_precision(yv, p, TOP_FRAC), roc_auc_score(yv, p), m.best_iteration_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_parquet(DATA)
    df["year"] = df["date"].dt.year
    features = [c for c in df.columns if c not in DROP and c != "year"]
    cats = [c for c in CATEGORICAL if c in features]
    fit = df[df["year"] <= FIT_MAX_YEAR]
    val = df[df["year"] == VAL_YEAR]

    def Xy(part):
        X = part[features].copy()
        for c in cats:
            X[c] = X[c].astype("category")
        return X, part[TARGET]

    Xf, yf = Xy(fit)
    Xv, yv = Xy(val)
    print(f"Tuning honestly: fit<= {FIT_MAX_YEAR} ({len(fit):,} touches) | "
          f"val {VAL_YEAR} ({len(val):,}). Objective = react-precision in the "
          f"top {TOP_FRAC:.0%} most-confident val touches (val react rate "
          f"{yv.mean():.1%}).\n")

    # Baseline (current hand-set params) for reference.
    base_prec, base_auc, base_it = fit_score(DEFAULT_PARAMS, Xf, yf, Xv, yv)
    print(f"baseline params : top-{TOP_FRAC:.0%} precision {base_prec:.3f} | "
          f"val AUC {base_auc:.3f} (best_iter {base_it})\n")

    rng = random.Random(SEED)
    best = {"prec": base_prec, "auc": base_auc, "params": {}, "trial": "baseline"}
    for i in range(args.trials):
        cand = {k: rng.choice(GRID[k]) for k in TUNABLE_KEYS}
        params = dict(DEFAULT_PARAMS)
        params.update(cand)
        prec, auc, it = fit_score(params, Xf, yf, Xv, yv)
        tag = ""
        if prec > best["prec"] + 1e-9:
            best = {"prec": prec, "auc": auc, "params": cand, "trial": i}
            tag = "  <-- new best"
        print(f"trial {i:>2}: prec {prec:.3f} auc {auc:.3f} it {it:>4}  "
              f"lr={cand['learning_rate']} leaves={cand['num_leaves']} "
              f"mcs={cand['min_child_samples']} ss={cand['subsample']} "
              f"cs={cand['colsample_bytree']} l2={cand['reg_lambda']}{tag}")

    print(f"\nBEST: top-{TOP_FRAC:.0%} precision {best['prec']:.3f} "
          f"(baseline {base_prec:.3f}, +{best['prec']-base_prec:.3f}) "
          f"| trial {best['trial']}")
    if best["params"]:
        BEST_PARAMS_PATH.write_text(json.dumps(best["params"], indent=2))
        print(f"Saved tuned params -> {BEST_PARAMS_PATH.relative_to(BEST_PARAMS_PATH.parents[1])}")
        print(json.dumps(best["params"], indent=2))
    else:
        print("No candidate beat the baseline; keeping default params "
              "(best_params.json not written).")
    print("\nNext: retrain (train_model.py) + walk-forward (walkforward_backtest.py) "
          "to confirm the lift on the untouched 2023-2026 folds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
