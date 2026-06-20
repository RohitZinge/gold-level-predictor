#!/usr/bin/env python3
"""
Phases 5-7 — split, train, calibrate, evaluate (the honest scorecard).

Pipeline (professional, leakage-safe):
  5. TIME-BASED split (never shuffled): oldest years train, middle validate,
     most-recent test (held out until the end). Splitting by DATE keeps each
     trading day wholly in one split.
  6. BASELINE  : Logistic Regression (simple, sets the bar to beat).
     MAIN MODEL : LightGBM gradient-boosted trees (early-stopped on validation).
     Both use class weights for the mild react/break imbalance.
     CALIBRATION: isotonic on the validation set so a "70%" really means ~70%.
  7. EVALUATE on the untouched TEST set: ROC-AUC, PR-AUC, Brier (calibration),
     reliability table, and a TRADING-USEFUL view (precision & lift when the
     model is confident vs. trading every level blindly).

Model inputs are the DIMENSIONLESS features only — raw-dollar columns
(level_price, day_open, step_size, atr_m15) are dropped so the model learns
level behaviour, not gold's 2020-2026 price regime.

Inputs : data/processed/dataset.parquet
Outputs: models/{baseline_logreg,lgbm_calibrated}.joblib, models/metrics.json,
         models/feature_importance.csv

Usage:
    python3 src/train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_config import lgbm_params, monotone_list  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dataset.parquet"
MODELS = ROOT / "models"

TARGET = "outcome"                       # 1 = reacted, 0 = broke
# Columns never fed to the model: ids/label + raw-dollar (price-regime) columns.
DROP = ["date", "touch_time", "outcome", "level_price",
        "day_open", "step_size", "atr_m15"]
CATEGORICAL = ["type", "session", "day_of_week"]

VAL_START = "2024-01-01"                  # train < this
TEST_START = "2025-01-01"                 # val < this <= test
SEED = 42


# --------------------------------------------------------------------------- #
def time_split(df: pd.DataFrame):
    """Split by date: train (oldest) / val (middle) / test (most recent)."""
    d = df["date"]
    train = df[d < VAL_START]
    val = df[(d >= VAL_START) & (d < TEST_START)]
    test = df[d >= TEST_START]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s}: {len(part):>7,} touches  "
              f"{part['date'].min().date()} .. {part['date'].max().date()}  "
              f"react={part[TARGET].mean():.1%}")
    return train, val, test


def evaluate(name, y, p):
    """Core probability metrics for predictions p of P(react) against truth y."""
    return {
        "model": name,
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
    }


def reliability_table(y, p, bins=10):
    """Are the probabilities honest? mean predicted vs actual, per bin."""
    df = pd.DataFrame({"y": y.values, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    return g


def trading_view(y, p, base_rate):
    """When the model is confident, how much better than trading every level?"""
    rows = []
    for t in (0.50, 0.55, 0.60, 0.65, 0.70):
        # confident the level REACTS (would trade the bounce)
        m = p >= t
        if m.sum():
            rows.append(("react", t, int(m.sum()), m.mean(), y[m].mean(),
                         y[m].mean() / base_rate))
        # confident the level BREAKS (would trade the break-through)
        mb = p <= (1 - t)
        if mb.sum():
            rows.append(("break", t, int(mb.sum()), mb.mean(),
                         1 - y[mb].mean(), (1 - y[mb].mean()) / (1 - base_rate)))
    return pd.DataFrame(rows, columns=["call", "conf>=", "n", "coverage",
                                       "precision", "lift"])


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA).sort_values("date").reset_index(drop=True)
    features = [c for c in df.columns if c not in DROP]
    cats = [c for c in CATEGORICAL if c in features]
    nums = [c for c in features if c not in cats]
    print(f"Dataset: {len(df):,} touches, {len(features)} model features "
          f"({len(cats)} categorical). Target: react(1) vs break(0).")

    print("\nTime-based split (no shuffle):")
    train, val, test = time_split(df)
    base_rate = train[TARGET].mean()       # P(react) prior, from train only

    def Xy(part):
        X = part[features].copy()
        for c in cats:
            X[c] = X[c].astype("category")
        return X, part[TARGET]

    Xtr, ytr = Xy(train); Xva, yva = Xy(val); Xte, yte = Xy(test)

    # ---- 1) Baseline: Logistic Regression --------------------------------
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), nums),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    baseline = Pipeline([("pre", pre),
                         ("lr", LogisticRegression(max_iter=2000,
                                                   class_weight="balanced",
                                                   random_state=SEED))])
    # LogReg can't use pandas 'category' OHE-NaN well -> feed plain frames.
    baseline.fit(train[features], ytr)
    p_base = baseline.predict_proba(test[features])[:, 1]

    # ---- 2) Main model: LightGBM (params from model_config + monotone) ----
    params = lgbm_params()
    mc = monotone_list(features)
    if mc is not None:
        params["monotone_constraints"] = mc
    print(f"\nLightGBM params: lr={params['learning_rate']} "
          f"leaves={params['num_leaves']} mcs={params['min_child_samples']} "
          f"l2={params['reg_lambda']} | monotone={'on' if mc else 'off'}")
    lgbm = lgb.LGBMClassifier(**params)
    lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
             callbacks=[lgb.early_stopping(150, verbose=False),
                        lgb.log_evaluation(0)])
    best_iter = lgbm.best_iteration_
    p_lgb = lgbm.predict_proba(Xte)[:, 1]

    # ---- 3) Calibrate the main model on validation (isotonic) ------------
    try:
        from sklearn.frozen import FrozenEstimator
        calib = CalibratedClassifierCV(FrozenEstimator(lgbm), method="isotonic")
    except Exception:
        calib = CalibratedClassifierCV(lgbm, method="isotonic", cv="prefit")
    calib.fit(Xva, yva)
    p_cal = calib.predict_proba(Xte)[:, 1]

    # ---- 4) Scorecard on TEST --------------------------------------------
    p_const = np.full(len(yte), base_rate)         # no-skill: always base rate
    results = [
        evaluate("no-skill (base rate)", yte, p_const),
        evaluate("baseline LogReg", yte, p_base),
        evaluate("LightGBM (raw)", yte, p_lgb),
        evaluate("LightGBM (calibrated)", yte, p_cal),
    ]
    res = pd.DataFrame(results).set_index("model")

    print(f"\nLightGBM best iteration (early-stopped on val): {best_iter}")
    print("\n===== TEST SCORECARD (held-out, most recent data) =====")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(res.to_string())

    print("\nCalibration check (LightGBM calibrated) — mean predicted vs actual:")
    rel = reliability_table(yte, p_cal)
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(rel.to_string())

    print(f"\nTrading-useful view (test base rate react={yte.mean():.1%}, "
          f"break={1-yte.mean():.1%}):")
    tv = trading_view(yte, p_cal, base_rate=yte.mean())
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(tv.to_string(index=False))

    # ---- 5) Feature importance -------------------------------------------
    imp = (pd.DataFrame({"feature": features, "gain": lgbm.feature_importances_})
           .sort_values("gain", ascending=False).reset_index(drop=True))
    print("\nTop 15 features (LightGBM gain):")
    print(imp.head(15).to_string(index=False))

    # ---- 6) Save artifacts -----------------------------------------------
    joblib.dump(baseline, MODELS / "baseline_logreg.joblib")
    joblib.dump({"model": calib, "features": features, "categorical": cats,
                 "base_rate": float(base_rate)},
                MODELS / "lgbm_calibrated.joblib")
    imp.to_csv(MODELS / "feature_importance.csv", index=False)
    res.reset_index().to_json(MODELS / "metrics.json", orient="records", indent=2)

    auc_gain = res.loc["LightGBM (calibrated)", "roc_auc"] - \
        res.loc["baseline LogReg", "roc_auc"]
    print(f"\nSaved models + metrics -> {MODELS.relative_to(ROOT)}/")
    print(f"LightGBM beats LogReg on AUC by {auc_gain:+.4f}; "
          f"beats no-skill (0.5000) by {res.loc['LightGBM (calibrated)','roc_auc']-0.5:+.4f}.")
    print("\nDone (Phases 5-7). Review the scorecard. SHAP explanations (Phase 8) next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
