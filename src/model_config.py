#!/usr/bin/env python3
"""
Shared model configuration — single source of truth for both train_model.py and
walkforward_backtest.py so they never drift apart.

- DROP / CATEGORICAL / TARGET : the column contract for the model.
- DEFAULT_PARAMS              : hand-set LightGBM params (the original baseline).
- lgbm_params()              : DEFAULT_PARAMS overlaid with any tuned values from
                               models/best_params.json (written by tune_model.py).
- MONOTONE / monotone_list() : monotonic constraints on the features whose effect
                               on P(react) has a clear, known direction. They make
                               the model generalise better and stay intuitive. Set
                               USE_MONOTONE = False to disable for an A/B check.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dataset.parquet"
BEST_PARAMS_PATH = ROOT / "models" / "best_params.json"

TARGET = "outcome"                       # 1 = reacted, 0 = broke
# Never fed to the model: ids/label + raw-dollar (price-regime) columns.
DROP = ["date", "touch_time", "outcome", "level_price",
        "day_open", "step_size", "atr_m15"]
CATEGORICAL = ["type", "session", "day_of_week"]

# The original, hand-tuned LightGBM configuration.
DEFAULT_PARAMS = dict(
    n_estimators=3000, learning_rate=0.02, num_leaves=31,
    min_child_samples=80, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=5.0, class_weight="balanced",
    random_state=42, n_jobs=-1, verbose=-1,
)
# Keys that tune_model.py is allowed to override (search space). Infra keys
# (class_weight/random_state/n_jobs/verbose/n_estimators) are always pinned.
TUNABLE_KEYS = ("learning_rate", "num_leaves", "min_child_samples", "subsample",
                "colsample_bytree", "reg_lambda", "reg_alpha", "min_split_gain")

# Monotonic constraints: +1 = feature raises P(react) monotonically, -1 = lowers.
# Only features with a clear domain direction; everything else is free (0).
# A/B (2026-06-20) found monotone constraints did NOT help the features-only
# model, so they are OFF by default. Re-enable for experiments via USE_MONOTONE=1.
USE_MONOTONE = os.environ.get("USE_MONOTONE", "0") != "0"
MONOTONE = {
    "confluence_count": +1,                 # more stacked prior levels -> holds more
    "dist_nearest_prior_level_steps": -1,   # farther from prior levels -> holds less
    "prior_touches_today": -1,              # worn-down level -> breaks more
    "is_first_touch": +1,                   # fresh first touch -> holds more
}


def lgbm_params() -> dict:
    """DEFAULT_PARAMS overlaid with tuned values from best_params.json (if present)."""
    p = dict(DEFAULT_PARAMS)
    if BEST_PARAMS_PATH.exists() and os.environ.get("USE_TUNED", "1") != "0":
        tuned = json.loads(BEST_PARAMS_PATH.read_text())
        p.update({k: v for k, v in tuned.items() if k in TUNABLE_KEYS})
    # Always pin the infrastructure keys.
    p.update(dict(n_estimators=3000, class_weight="balanced",
                  random_state=42, n_jobs=-1, verbose=-1))
    return p


def monotone_list(features) -> list | None:
    """Monotone-constraint vector aligned to `features`, or None if disabled."""
    if not USE_MONOTONE:
        return None
    return [int(MONOTONE.get(f, 0)) for f in features]
