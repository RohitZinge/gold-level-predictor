#!/usr/bin/env python3
"""
Phase 8 — SHAP explanations: turn each level's score into plain-English reasons.

SHAP is run on the underlying LightGBM booster (exact TreeSHAP via native
pred_contrib). Calibration is a monotonic isotonic layer on top, so the DRIVERS
are the same; we map each contribution THROUGH the isotonic calibrator so the
"% reasons" live in the same calibrated probability space as the headline.

1. Global  : SHAP beeswarm + mean-|SHAP| bar  -> reports/shap_beeswarm.png,
             reports/shap_importance.png  (the trustworthy feature importance).
2. explain_level(features) -> readable reasons FOR and AGAINST a reaction, e.g.
   "React 72% (base 44%) — at the weekly low (+12%), fresh after a break (+8%),
    near a round number (+5%), but against the short-term trend (-6%)."
3. Three worked examples from recent out-of-sample days (high-react, high-break,
   uncertain).

Usage:
    python3 src/explain.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dataset.parquet"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
TEST_START = "2025-01-01"          # recent out-of-sample period for examples/plots
# Never display a calibrated 0%/100% — no level is ever a guaranteed react/break.
# Isotonic calibration can saturate to exactly 0/1 on its top/bottom step, so we
# clip the shown probability to a sane band.
PROB_CLIP = (0.01, 0.99)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# Human phrases: describe each feature's CURRENT STATE in plain English.
# --------------------------------------------------------------------------- #
_DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _near(v, name, tol=0.6):
    return f"at {name}" if abs(v) < tol else (f"below {name}" if v < 0 else f"above {name}")


def _round_phrase(v, label):
    return f"on a {label} round number" if abs(v) < 0.3 else f"off {label} round numbers"


def feat_phrase(name, v):
    """A readable phrase describing feature `name` at value `v`."""
    try:
        if name == "price_vs_ema20_steps":
            return "above the short-term trend" if v > 0.25 else ("below the short-term trend" if v < -0.25 else "on the short-term trend")
        if name == "price_vs_ema50_steps":
            return "above the medium-term trend" if v > 0.25 else ("below the medium-term trend" if v < -0.25 else "on the medium-term trend")
        if name == "ema20_slope_steps":
            return "trend rising" if v > 0.05 else ("trend falling" if v < -0.05 else "trend flat")
        if name == "recent_return_steps":
            return "a recent push up" if v > 0.1 else ("a recent push down" if v < -0.1 else "flat momentum")
        if name == "dist_wlow_steps":
            return _near(v, "the weekly low")
        if name == "dist_whigh_steps":
            return _near(v, "the weekly high")
        if name == "dist_pdl_steps":
            return _near(v, "yesterday's low")
        if name == "dist_pdh_steps":
            return _near(v, "yesterday's high")
        if name == "dist_round_5_steps":
            return _round_phrase(v, "$5")
        if name == "dist_round_10_steps":
            return _round_phrase(v, "$10")
        if name == "dist_round_25_steps":
            return _round_phrase(v, "$25")
        if name == "dist_round_50_steps":
            return _round_phrase(v, "$50")
        if name == "bars_since_this_level_broken":
            return f"fresh after a break ({int(v)} bars ago)" if pd.notna(v) else "not broken earlier today"
        if name == "is_first_touch":
            return "the first touch today" if v >= 0.5 else "a retest"
        if name == "prior_touches_today":
            return f"tested {int(v)}x earlier today" if v > 0 else "untested so far today"
        if name == "levels_broken_before":
            return f"{int(v)} levels already broken today"
        if name == "atr_steps":
            return "high volatility" if v > 1.2 else ("low volatility" if v < 0.5 else "normal volatility")
        if name == "range_so_far_steps":
            return "a wide day so far" if v > 6 else ("a tight day so far" if v < 2 else "an average day so far")
        if name == "hours_since_open":
            return "early in the day" if v < 4 else ("late in the day" if v > 16 else "mid-day")
        if name == "session":
            return f"the {v} session"
        if name == "day_of_week":
            return _DOW[int(v)]
        if name == "step_number":
            return f"ladder step {int(v):+d}"
        if name == "abs_step":
            return f"{int(v)} steps from the open"
        if name == "above_or_below_open":
            return "above the open" if v > 0 else ("below the open" if v < 0 else "the open level")
        if name == "approached_from_above":
            return "tested as support" if v > 0 else "tested as resistance"
        if name == "type":
            return f"a {v} level"
        if name == "gap_steps":
            return "an up-gap open" if v > 0.3 else ("a down-gap open" if v < -0.3 else "a flat open")
        if name == "rel_volume":
            return "on heavy volume" if v > 1.3 else ("on light volume" if v < 0.7 else "on normal volume")
        if name == "vol_trend":
            return "volume rising" if v > 1.1 else ("volume fading" if v < 0.9 else "steady volume")
        if name == "dist_nearest_prior_level_steps":
            return "stacked on a prior-day level" if (pd.notna(v) and v < 0.5) else "clear of prior-day levels"
        if name == "confluence_count":
            return (f"{int(v)} prior-day levels stacked here"
                    if (pd.notna(v) and v >= 1) else "no prior-day confluence")
    except Exception:
        pass
    return name


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #
class LevelExplainer:
    def __init__(self):
        b = joblib.load(MODELS / "lgbm_calibrated.joblib")
        self.features = list(b["features"])
        self.cats = list(b["categorical"])
        self.calib = b["model"]
        est = self.calib.calibrated_classifiers_[0].estimator
        self.lgbm = getattr(est, "estimator", est)         # unwrap FrozenEstimator
        self.booster = self.lgbm.booster_
        # isotonic calibrator (booster prob -> honest prob); identity if absent
        cc = self.calib.calibrated_classifiers_[0]
        cals = getattr(cc, "calibrators", None) or getattr(cc, "calibrators_", None)
        self.iso = cals[0] if cals else None
        # category levels exactly as used at training
        ds = pd.read_parquet(DATA)
        self.cat_levels = {c: ds[c].astype("category").cat.categories for c in self.cats}
        self.base_logodds = float(self.booster.predict(
            self._prep(ds.head(1)), pred_contrib=True)[0, -1])

    # -- internals ----------------------------------------------------------
    def _prep(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows[self.features].copy()
        for c in self.features:
            if c in self.cats:
                X[c] = pd.Categorical(X[c], categories=self.cat_levels[c])
            else:                                  # robust if row came in as object
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def _cal(self, p_boost):
        """Map booster probability -> calibrated (honest) probability, clipped."""
        p = np.atleast_1d(p_boost)
        out = self.iso.predict(p) if self.iso is not None else p
        return np.clip(out, *PROB_CLIP)

    def react_pct(self, rows: pd.DataFrame) -> np.ndarray:
        p = self.calib.predict_proba(self._prep(rows))[:, 1]
        return np.clip(p, *PROB_CLIP)

    def shap_values(self, rows: pd.DataFrame):
        contrib = self.booster.predict(self._prep(rows), pred_contrib=True)
        return contrib[:, :-1]                              # (n, n_features)

    # -- the headline API ---------------------------------------------------
    def explain(self, row: pd.Series, n_for=4, n_against=3, min_pct=0.005) -> dict:
        """Plain-English reasons for one touch (a Series of the model features)."""
        rowdf = row.to_frame().T
        shap = self.shap_values(rowdf)[0]
        raw = self.base_logodds + shap.sum()
        final = float(self._cal(_sigmoid(raw))[0])
        base = float(self._cal(_sigmoid(self.base_logodds))[0])

        # each feature's calibrated-probability impact (leave-its-contribution-out)
        contribs = []
        for f, s in zip(self.features, shap):
            impact = final - float(self._cal(_sigmoid(raw - s))[0])
            contribs.append((f, impact, row[f]))

        fors = sorted([c for c in contribs if c[1] >= min_pct],
                      key=lambda c: -c[1])[:n_for]
        against = sorted([c for c in contribs if c[1] <= -min_pct],
                         key=lambda c: c[1])[:n_against]

        def phr(lst):
            return [f"{feat_phrase(f, v)} ({imp:+.0%})" for f, imp, v in lst]

        parts = phr(fors)
        text = f"React {final:.0%} (base {base:.0%})"
        if parts:
            text += " — " + ", ".join(parts)
        if against:
            text += (", but " if parts else " — but ") + ", ".join(phr(against))
        return {"react_pct": final, "base_pct": base, "text": text,
                "for": fors, "against": against}


# --------------------------------------------------------------------------- #
# Global SHAP plots
# --------------------------------------------------------------------------- #
def global_plots(expl: LevelExplainer, sample: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap_vals = expl.shap_values(sample)
    X_plot = sample[expl.features].copy()
    for c in expl.cats:                                     # codes so colour works
        X_plot[c] = pd.Categorical(X_plot[c],
                                   categories=expl.cat_levels[c]).codes

    shap.summary_plot(shap_vals, X_plot, feature_names=expl.features,
                      max_display=20, show=False)
    plt.title("SHAP beeswarm — drivers of React(+) vs Break(-)")
    plt.tight_layout(); plt.savefig(REPORTS / "shap_beeswarm.png", dpi=110)
    plt.close()

    shap.summary_plot(shap_vals, X_plot, feature_names=expl.features,
                      plot_type="bar", max_display=20, show=False)
    plt.title("Mean |SHAP| — feature importance (trustworthy)")
    plt.tight_layout(); plt.savefig(REPORTS / "shap_importance.png", dpi=110)
    plt.close()
    return shap_vals


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    expl = LevelExplainer()
    df = pd.read_parquet(DATA)
    test = df[df["date"] >= TEST_START].reset_index(drop=True)

    # ---- global plots on a recent OOS sample ----
    rng = np.random.default_rng(0)
    samp = test.iloc[rng.choice(len(test), size=min(3000, len(test)), replace=False)]
    shap_vals = global_plots(expl, samp)
    mean_abs = pd.Series(np.abs(shap_vals).mean(0), index=expl.features) \
        .sort_values(ascending=False)
    print("Global SHAP — top 12 drivers by mean |SHAP| (log-odds):")
    for f, v in mean_abs.head(12).items():
        print(f"  {f:30s} {v:.3f}")
    print("Saved -> reports/shap_beeswarm.png, reports/shap_importance.png")

    # ---- 3 worked examples from recent OOS days ----
    p = expl.react_pct(test)
    test = test.assign(p_react=p)
    def nearest(target):                       # representative, not saturated
        return (test["p_react"] - target).abs().idxmin()
    picks = {
        "HIGH-CONFIDENCE REACT": nearest(0.80),
        "HIGH-CONFIDENCE BREAK": nearest(0.20),
        "UNCERTAIN (~50%)": nearest(0.50),
    }
    print("\n===== 3 worked examples (recent out-of-sample) =====")
    for label, idx in picks.items():
        r = test.loc[idx]
        out = expl.explain(r[expl.features])
        actual = "reacted" if r["outcome"] == 1 else "broke"
        print(f"\n{label}")
        print(f"  {r['date'].date()}  step {int(r['step_number']):+d} ({r['type']}) "
              f"@ {r['level_price']:.2f}   actual: {actual}")
        print(f"  {out['text']}")

    print("\nDone (Phase 8). Explainer in src/explain.py; plots in reports/. "
          "Reasons are calibrated-probability contributions (approx).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
