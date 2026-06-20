#!/usr/bin/env python3
"""
Phase 9 — the daily tool (Streamlit dashboard).

Each morning: type today's daily OPEN (from your FOREX.com chart so the levels
match exactly) -> the app builds your 21 ladder levels, scores each with the
trained, calibrated model assuming a FIRST, untested touch, ranks them, and
shows the reasons. Strong levels (>=65%) are pulled to the top.

Run:
    streamlit run app/daily_tool.py

Headless sample (no browser):
    python3 app/daily_tool.py --demo 3300            # score an open of 3300
    python3 app/daily_tool.py --demo 3300 2026-06-18 # ... for a given date
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from levels import get_day_levels, step_size          # noqa: E402
from explain import LevelExplainer, feat_phrase        # noqa: E402

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
STRONG = 0.65
TRADE_BAR = 0.80          # measured high-confidence "take the trade" threshold (see backtest)
DISCLAIMER = (
    "Scores are taken at the day's OPEN and assume a first, untested touch. "
    "They are sharpest when price actually reaches the level — re-score then. "
    "This tilts the odds; it is not a guarantee."
)

ATR_N, EMA_FAST, EMA_SLOW, SLOPE_N, RET_N, WEEK_DAYS = 14, 20, 50, 8, 8, 5
CONFLUENCE_K, CONFLUENCE_TOL = 5, 0.5    # prior days + step-tolerance for confluence


DAY_START_TZ, DAY_START_HOUR = "America/New_York", 17


def _trading_day(idx_utc):
    """Map UTC bar timestamps to their trading-day label (same rule as the pipeline)."""
    local = idx_utc.tz_convert(DAY_START_TZ)
    bump = np.where(local.hour >= DAY_START_HOUR, 1, 0)
    return pd.to_datetime(local.date) + pd.to_timedelta(bump, unit="D")


# --------------------------------------------------------------------------- #
# Market context — AS OF a chosen day's open (replay), or latest data (live)
# --------------------------------------------------------------------------- #
def build_context(chart_date=None) -> dict:
    """
    Slow-moving $ context (trend, volatility, gap refs, confluence).
    If `chart_date` is a trading day present in the data, the context is computed
    strictly AS OF that day's open (true historical replay — only bars before the
    open are used). Otherwise it falls back to the latest data (live use).
    """
    m = pd.read_parquet(PROC / "m15.parquet")
    d = pd.read_parquet(PROC / "d1.parquet")
    lv = pd.read_parquet(PROC / "levels.parquet")

    asof = None
    if chart_date is not None:
        cd = pd.Timestamp(chart_date)
        sess = m.index[_trading_day(m.index) == cd]
        if len(sess):
            asof = sess[0]                          # first M15 bar of that session
            m = m[m.index < asof]                   # strictly BEFORE the open
            d = d[_trading_day(d.index) < cd]       # only prior completed sessions
            lv = lv[lv["date"] < cd]                # only prior days' levels

    close = m["close"]
    prev = close.shift(1)
    tr = pd.concat([(m["high"] - m["low"]), (m["high"] - prev).abs(),
                    (m["low"] - prev).abs()], axis=1).max(axis=1)
    ema_f = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_s = close.ewm(span=EMA_SLOW, adjust=False).mean()
    dd = d.sort_index()
    last_days = sorted(lv["date"].unique())[-CONFLUENCE_K:]
    prior_levels = np.sort(lv.loc[lv["date"].isin(last_days), "level_price"].to_numpy())
    return {
        "as_of": asof if asof is not None else m.index[-1],
        "as_of_kind": f"as of {pd.Timestamp(chart_date).date()} open"
                      if asof is not None else "latest data",
        "prior_levels": prior_levels,
        "last_close": float(close.iloc[-1]),
        "latest_open": float(dd["open"].iloc[-1]),
        "latest_chart_day": dd.index[-1].date() + timedelta(days=1),
        "atr": float(tr.rolling(ATR_N).mean().iloc[-1]),
        "ema20": float(ema_f.iloc[-1]),
        "ema50": float(ema_s.iloc[-1]),
        "ema20_slope": float(ema_f.iloc[-1] - ema_f.iloc[-1 - SLOPE_N]),
        "recent_ret": float(close.iloc[-1] - close.iloc[-1 - RET_N]),
        "prev_close": float(dd["close"].iloc[-1]),
        "pdh": float(dd["high"].iloc[-1]),
        "pdl": float(dd["low"].iloc[-1]),
        "whigh": float(dd["high"].iloc[-WEEK_DAYS:].max()),
        "wlow": float(dd["low"].iloc[-WEEK_DAYS:].min()),
    }


# --------------------------------------------------------------------------- #
# Feature row for one level
# --------------------------------------------------------------------------- #
def make_row(level: dict, day_open: float, step: float, ctx: dict,
             dow: int, state: dict | None = None) -> dict:
    """
    Build the full model-feature row for one level.

    Defaults = START OF DAY (first, untested touch, nothing happened yet).
    `state` overrides the dynamic fields for a mid-day RE-SCORE (real touch-time
    values): current_price, bars_since_open, prior_touches, levels_broken,
    bars_since_break, atr (in $), session, approached_from_above, range_so_far_$.
    """
    state = state or {}
    k = int(level["step_number"])
    L = float(level.get("level_price", level.get("price")))   # get_day_levels uses 'price'
    price = float(state.get("current_price", day_open))     # at open: price = open
    atr = float(state.get("atr", ctx["atr"]))
    bars = int(state.get("bars_since_open", 0))
    prior = int(state.get("prior_touches", 0))
    bsb = state.get("bars_since_break", np.nan)
    appr = state.get("approached_from_above", 1 if k <= 0 else -1)
    sess = state.get("session", "Asia")                     # day opens 18:00 NY
    rng = float(state.get("range_so_far", 0.0))
    # trend context: from latest data by default; overridable for a live re-score
    ema20 = float(state.get("ema20", ctx["ema20"]))
    ema50 = float(state.get("ema50", ctx["ema50"]))
    ema20_slope = float(state.get("ema20_slope", ctx["ema20_slope"]))
    recent_ret = float(state.get("recent_ret", ctx["recent_ret"]))

    def ps(x):
        return x / step
    nr = lambda R: abs(L - round(L / R) * R)                # dist to round number

    # cross-day confluence vs the prior days' levels (same logic as build_features)
    prior_arr = ctx.get("prior_levels")
    if prior_arr is not None and len(prior_arr):
        ins = int(np.searchsorted(prior_arr, L))
        near = []
        if ins < len(prior_arr):
            near.append(prior_arr[ins] - L)
        if ins > 0:
            near.append(L - prior_arr[ins - 1])
        dist_prior = min(abs(x) for x in near) / step
        lo = np.searchsorted(prior_arr, L - CONFLUENCE_TOL * step, side="left")
        hi = np.searchsorted(prior_arr, L + CONFLUENCE_TOL * step, side="right")
        conf_count = int(hi - lo)
    else:
        dist_prior, conf_count = np.nan, 0

    return {
        "step_number": k, "type": level["type"], "abs_step": abs(k),
        "above_or_below_open": int(np.sign(k)),
        "hours_since_open": bars * 0.25, "session": sess, "day_of_week": dow,
        "bars_since_day_open": bars,
        "atr_steps": ps(atr), "range_so_far_steps": ps(rng),
        "rel_volume": float(state.get("rel_volume", 1.0)),
        "vol_trend": float(state.get("vol_trend", 1.0)),
        "price_vs_ema20_steps": ps(price - ema20),
        "price_vs_ema50_steps": ps(price - ema50),
        "ema20_slope_steps": ps(ema20_slope),
        "recent_return_steps": ps(recent_ret),
        "approached_from_above": int(appr),
        "is_first_touch": int(prior == 0), "prior_touches_today": prior,
        "levels_broken_before": int(state.get("levels_broken", 0)),
        "bars_since_this_level_broken": bsb,
        "dist_round_5_steps": ps(nr(5)), "dist_round_10_steps": ps(nr(10)),
        "dist_round_25_steps": ps(nr(25)), "dist_round_50_steps": ps(nr(50)),
        "dist_pdh_steps": ps(L - ctx["pdh"]), "dist_pdl_steps": ps(L - ctx["pdl"]),
        "dist_whigh_steps": ps(L - ctx["whigh"]), "dist_wlow_steps": ps(L - ctx["wlow"]),
        "dist_nearest_prior_level_steps": dist_prior, "confluence_count": conf_count,
        "gap_steps": ps(day_open - ctx["prev_close"]),
    }


def side_of(k: int) -> str:
    return "support" if k < 0 else ("resistance" if k > 0 else "open")


# --------------------------------------------------------------------------- #
# TradingView Pine code generator — your indicator, only the % array filled in
# --------------------------------------------------------------------------- #
def pct_by_step(day_open: float, dow: int, expl: LevelExplainer, ctx: dict) -> list:
    """The 21 react % in STEP order (idx 0 = step -10 … idx 20 = step +10)."""
    step = step_size(day_open)
    levels = get_day_levels(day_open)               # already ordered -10..+10
    rows = [make_row(lv, day_open, step, ctx, dow) for lv in levels]
    X = pd.DataFrame(rows)[expl.features]
    probs = expl.react_pct(X)                        # clipped 1-99
    return [int(round(float(p) * 100)) for p in probs]


def generate_pine(day_open: float, dow: int, expl: LevelExplainer, ctx: dict) -> str:
    """Your Pine template with ONLY the modelPct array filled for this open."""
    pcts = pct_by_step(day_open, dow, expl, ctx)
    tmpl = (Path(__file__).resolve().parent / "pine_template.pine").read_text()
    return (tmpl.replace("{{PCTS}}", ", ".join(str(v) for v in pcts))
                .replace("{{OPEN}}", f"{day_open:.3f}"))


def reasons_short(expl: LevelExplainer, row: pd.Series, k: int = 3) -> str:
    out = expl.explain(row, n_for=k, n_against=k, min_pct=0.01)
    merged = sorted(out["for"] + out["against"], key=lambda c: -abs(c[1]))[:k]
    return ", ".join(f"{feat_phrase(f, v)} ({imp:+.0%})" for f, imp, v in merged)


# --------------------------------------------------------------------------- #
# Score all 21 levels at the open
# --------------------------------------------------------------------------- #
def score_open(day_open: float, dow: int, expl: LevelExplainer, ctx: dict) -> pd.DataFrame:
    step = step_size(day_open)
    levels = get_day_levels(day_open)
    rows = [make_row(lv, day_open, step, ctx, dow) for lv in levels]
    X = pd.DataFrame(rows)[expl.features]
    probs = expl.react_pct(X)
    recs = []
    for lv, p, (_, r) in zip(levels, probs, X.iterrows()):
        k = int(lv["step_number"])
        recs.append({"level_price": round(lv["price"], 2), "step": k,
                     "type": lv["type"], "side": side_of(k),
                     "react_%": round(100 * p, 1),
                     "reasons": reasons_short(expl, r)})
    out = pd.DataFrame(recs).sort_values("react_%", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


def rescore_level(level: dict, day_open: float, current_market_state: dict,
                  expl: LevelExplainer, ctx: dict, dow: int) -> dict:
    """Sharper score for a level price is ACTUALLY testing now (real state)."""
    step = step_size(day_open)
    row = pd.Series(make_row(level, day_open, step, ctx, dow,
                             state=current_market_state))[expl.features]
    out = expl.explain(row)
    return {"react_pct": out["react_pct"], "text": out["text"],
            "side": side_of(int(level["step_number"]))}


# --------------------------------------------------------------------------- #
# Exact open per trading day, straight from our dataset (FOREX.com-anchored)
# --------------------------------------------------------------------------- #
def opens_by_date() -> dict:
    """{date -> day_open} from levels.parquet — the exact opens we anchored to."""
    lv = pd.read_parquet(PROC / "levels.parquet")
    s = lv.drop_duplicates("date").set_index("date")["day_open"]
    return {pd.Timestamp(k).date(): float(v) for k, v in s.items()}


# --------------------------------------------------------------------------- #
# Model fingerprint — so the app visibly proves WHICH model it is using
# --------------------------------------------------------------------------- #
def model_fingerprint() -> dict:
    """Metadata about the trained model file the app loads (for provenance)."""
    import datetime
    import json
    import os

    p = MODELS / "lgbm_calibrated.joblib"
    if not p.exists():
        return {"ok": False}
    import joblib
    b = joblib.load(p)
    info = {
        "ok": True,
        "trained": datetime.datetime.fromtimestamp(os.path.getmtime(p)),
        "n_features": len(b["features"]),
        "base_rate": float(b.get("base_rate", float("nan"))),
        "kind": type(b["model"]).__name__,
        "auc": None, "brier": None, "wf_auc": None, "wf_years": None,
    }
    try:
        m = json.load(open(MODELS / "metrics.json"))
        cal = [r for r in m if "calibrated" in r["model"]][0]
        info["auc"], info["brier"] = cal["roc_auc"], cal["brier"]
    except Exception:
        pass
    try:
        wf = pd.read_csv(REPORTS / "walkforward_metrics.csv")
        info["wf_auc"] = float(wf["roc_auc"].mean())
        info["wf_years"] = f"{int(wf['test_year'].min())}–{int(wf['test_year'].max())}"
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
#MainMenu, footer {visibility: hidden;}
.hero {background: linear-gradient(110deg,#3a2f00 0%,#b8860b 55%,#ffd86b 100%);
       padding: 18px 24px; border-radius: 14px; margin-bottom: 6px;
       box-shadow: 0 4px 18px rgba(0,0,0,.25);}
.hero h1 {color:#fff; margin:0; font-size:1.7rem; letter-spacing:.3px;}
.hero p  {color:#fff4d6; margin:.25rem 0 0 0; font-size:.9rem;}
[data-testid="stMetricValue"] {font-size:1.5rem;}
[data-testid="stMetric"] {background:#1c1c1c0d; border:1px solid #8884; border-radius:12px;
                          padding:10px 14px;}
section[data-testid="stSidebar"] {min-width: 310px;}
.modelcard {background:#10331a; border:1px solid #2e7d32; border-radius:12px;
            padding:14px 16px; color:#d7f5dd;}
.modelcard b {color:#9be7a8;}
</style>
"""


def run_app():
    import streamlit as st

    st.set_page_config(page_title="Gold Ladder Levels", page_icon="🟡",
                       layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>🟡 Gold Ladder Levels — daily strength scores</h1>'
        '<p>Type the day\'s open → get all 21 levels ranked by the probability '
        'each one HOLDS (reacts), with the reasons why.</p></div>',
        unsafe_allow_html=True)

    expl = st.cache_resource(LevelExplainer)()
    fp = st.cache_data(model_fingerprint)()
    opens = st.cache_data(opens_by_date)()
    dmin, dmax = min(opens), max(opens)

    # ---- INPUT ROW (pick the day + open) -----------------------------------
    if "open_val" not in st.session_state:
        st.session_state.open_val = round(opens[dmax], 2)

    def _load_for_date():                  # runs before widgets re-instantiate
        sel = st.session_state.get("trade_date")
        o = opens.get(sel)
        if o is not None:
            st.session_state.open_val = round(o, 2)
            st.session_state.load_msg = (
                "success", f"Loaded exact open **{o:,.2f}** for **{sel}** "
                f"(FOREX.com-anchored — matches your chart).")
        else:
            st.session_state.load_msg = (
                "warning", f"No open stored for **{sel}** (weekend / holiday / "
                f"gap-dropped day). Pick a trading day between {dmin} and {dmax}.")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        day_open = st.number_input(
            "Daily OPEN (type your chart's yellow-line / 03:30-IST candle open)",
            step=0.1, format="%.2f", key="open_val")
    with c2:
        d = st.date_input("Trading day", value=dmax, key="trade_date",
                          help="Pick a day, then click the button to pull that "
                               "day's exact open AND replay that day's context.")
    with c3:
        st.write(""); st.write("")
        st.button("📅 Load open for this date", on_click=_load_for_date,
                  use_container_width=True,
                  help=f"Fills the open from our dataset for the selected day "
                       f"(available {dmin} → {dmax}). For a brand-new live day "
                       f"not in the data yet, type the open instead.")

    msg = st.session_state.pop("load_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])

    # ---- CONTEXT: as of the picked day (replay) or latest (live) -----------
    replay = d in opens
    ctx = st.cache_data(build_context)(d if replay else None)

    # ---- SIDEBAR: model provenance + context + help ------------------------
    with st.sidebar:
        st.markdown("### 🧠 Model in use")
        if fp.get("ok"):
            st.markdown(
                f'<div class="modelcard">'
                f'📦 <b>models/lgbm_calibrated.joblib</b><br>'
                f'🗓️ trained <b>{fp["trained"]:%Y-%m-%d %H:%M}</b><br>'
                f'🎯 walk-forward AUC <b>{fp["wf_auc"]:.3f}</b> '
                f'({fp["wf_years"]} out-of-sample)<br>'
                f'🧩 <b>{fp["n_features"]}</b> features · base rate '
                f'<b>{fp["base_rate"]:.0%}</b> react<br>'
                f'⚙️ {fp["kind"]} (isotonic-calibrated LightGBM)'
                f'</div>', unsafe_allow_html=True)
            st.caption("Every % is this model's calibrated output — verified: "
                       "when it says 70%, ~70% actually react.")
        else:
            st.error("No trained model found. Run `python3 src/train_model.py`.")

        st.markdown("---")
        st.markdown("### 🧭 Scores computed")
        st.write(f"**{ctx['as_of_kind']}**")
        if replay:
            st.caption(f"Replaying **{d}**: the levels AND the react % reflect "
                       f"that day's market context (only data before that day's "
                       f"open is used — a true historical replay).")
        else:
            st.caption("Live mode (date not in the data) — uses the latest "
                       "available context. Refresh the data for a new day:")
            st.code("python3 src/fetch_data.py --from 2011-01-01 --to <date>\n"
                    "python3 src/data_loader.py", language="bash")

        st.markdown("---")
        st.markdown("### ❓ How to read the open")
        st.caption("The open is the price at the daily rollover — **18:00 New "
                   "York = 03:30 IST** — i.e. the **yellow line** your indicator "
                   "draws. Type that exact number for an exact match.")

    # ---- CONTEXT NOTE ------------------------------------------------------
    if replay and d < dmax:
        st.success(f"📌 Replaying **{d}** — levels and react % are computed "
                   f"**as of that day's open** (true historical replay).")
    elif not replay:
        st.warning(
            f"⚠️ **{d}** isn't a trading day in the dataset. Level **prices are "
            f"exact** from the open you typed, but **react %** use the latest "
            f"available context — refresh the data (sidebar) for a live day.")

    # ---- SCORE -------------------------------------------------------------
    table = score_open(day_open, d.weekday(), expl, ctx)
    step = step_size(day_open)
    strong = table[table["react_%"] >= STRONG * 100].copy()
    weak = table[table["react_%"] < STRONG * 100].copy()
    trade = table[table["react_%"] >= TRADE_BAR * 100]

    # ---- KPI METRICS -------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Day open", f"{day_open:,.2f}")
    m2.metric("Step size", f"${step:,.2f}")
    m3.metric(f"Levels to watch (≥{int(STRONG*100)}%)", f"{len(strong)} / 21")
    m4.metric(f"Trade-grade (≥{int(TRADE_BAR*100)}%)", f"{len(trade)} / 21",
              help="The backtested high-confidence bar where react calls win ~75%.")

    def _emoji_side(s):
        return {"support": "🟢 support", "resistance": "🔴 resistance",
                "open": "🟡 open"}.get(s, s)

    def show(df):
        v = df.copy()
        v["side"] = v["side"].map(_emoji_side)
        st.dataframe(
            v, hide_index=True, use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "level_price": st.column_config.NumberColumn("Price", format="%.2f"),
                "step": st.column_config.NumberColumn("Step", format="%+d", width="small"),
                "type": st.column_config.TextColumn("Type", width="small"),
                "side": st.column_config.TextColumn("Role @ open"),
                "react_%": st.column_config.ProgressColumn(
                    "React probability", format="%.0f%%",
                    min_value=0, max_value=100),
                "reasons": st.column_config.TextColumn("Why", width="large"),
            })

    st.markdown(f"#### ⭐ Levels to watch — {len(strong)} scored ≥{int(STRONG*100)}%")
    if len(strong):
        show(strong)
    else:
        st.info("No level scored ≥65% at the open today — a low-conviction day.")

    with st.expander(f"Show the other {len(weak)} levels"):
        show(weak)

    st.caption("Role = the level's job at the open (below open ⇒ support, above "
               "⇒ resistance). Scores assume a **first, untested touch** and are "
               "sharpest when price actually reaches the level. " + DISCLAIMER)

    # ---- TRADINGVIEW PINE CODE (copy → paste) ------------------------------
    st.markdown("---")
    with st.expander("📋 TradingView code — copy & paste into your Pine editor"):
        st.caption("This is **your indicator, logic unchanged** — only the "
                   "`modelPct` array is filled with the 21 react % for the open "
                   "above. Paste it over your script; the % show above each line "
                   "for the current day. Re-copy whenever you change the open.")
        st.code(generate_pine(day_open, d.weekday(), expl, ctx),
                language="javascript")


# --------------------------------------------------------------------------- #
def demo(open_price: float, dow_date: date):
    expl = LevelExplainer()
    ctx = build_context()
    print(f"Context as of {ctx['as_of']:%Y-%m-%d %H:%M UTC}  "
          f"(prev_close={ctx['prev_close']:.2f}, atr=${ctx['atr']:.2f}, "
          f"ema20={ctx['ema20']:.2f})")
    print(f"\nOPEN = {open_price:.2f}  | trading day {dow_date} "
          f"({dow_date.strftime('%A')}) | step ${step_size(open_price):.2f}\n")
    table = score_open(open_price, dow_date.weekday(), expl, ctx)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(table.to_string(index=False))
    n_strong = (table["react_%"] >= STRONG * 100).sum()
    print(f"\n⭐ {n_strong} level(s) ≥{int(STRONG*100)}% to watch.")
    print("\nNOTE: " + DISCLAIMER)

    # demo the mid-day re-score on a near-open level actually being tested
    step = step_size(open_price)
    lvl = next(l for l in get_day_levels(open_price) if l["step_number"] == 2)
    rs = rescore_level(
        lvl, open_price,
        {"current_price": lvl["price"], "bars_since_open": 20, "prior_touches": 1,
         "levels_broken": 3, "session": "London", "range_so_far": 5 * step,
         "ema20": lvl["price"] - 0.3 * step},   # fresh trend: EMA caught up near price
        expl, ctx, dow_date.weekday())
    print(f"\nExample re-score (level {lvl['price']:.2f}, step +2, actually tested "
          f"mid-day in London after a wide range + 3 levels broke, on a retest):"
          f"\n  {rs['text']}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        i = sys.argv.index("--demo")
        op = float(sys.argv[i + 1])
        dd = (date.fromisoformat(sys.argv[i + 2])
              if len(sys.argv) > i + 2 else date.today())
        demo(op, dd)
    else:
        run_app()
