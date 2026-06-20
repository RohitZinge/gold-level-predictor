# Feature dictionary (Phase 4)

One row per touch in `data/processed/dataset.parquet`. Every feature uses only data up to and including the touch bar.

| feature | meaning |
|---|---|
| `step_number` | ladder step −10..+10 (0 = open); the level's identity |
| `abs_step` | distance from the open in steps (|step_number|) |
| `type` | open / thick (even step) / thin (odd step) |
| `above_or_below_open` | +1 above the open, −1 below, 0 = open |
| `hours_since_open` | trading-hours elapsed since the 6 PM-NY open |
| `session` | Asia / London / NY at the touch (by UTC hour) |
| `day_of_week` | trading day's weekday (Mon=0) |
| `bars_since_day_open` | M15 bars elapsed since the day open |
| `step_size` | this day's ladder spacing in $ (from the formula) |
| `atr_m15` | ATR over last 14 M15 bars, in $ |
| `atr_steps` | ATR in step-size units (volatility vs the ladder) |
| `range_so_far_steps` | today's high−low so far, in step units |
| `rel_volume` | touch-bar tick-volume vs its 20-bar rolling average (>1 = busier than usual) |
| `vol_trend` | volume 8-bar vs 40-bar average (>1 = volume rising) |
| `price_vs_ema20_steps` | (price − EMA20 on M15) in step units |
| `price_vs_ema50_steps` | (price − EMA50 on M15) in step units |
| `ema20_slope_steps` | EMA20 change over 8 M15 bars, in step units |
| `recent_return_steps` | price change over last 8 M15 bars, in step units |
| `approached_from_above` | +1 if price came from above (support test), −1 if below |
| `is_first_touch` | 1 if this is the level's first touch today |
| `prior_touches_today` | count of earlier touches of THIS level today |
| `levels_broken_before` | how many of the 21 levels price already closed through today |
| `bars_since_this_level_broken` | M15 bars since this level broke earlier today (NaN if not) |
| `dist_round_5_steps` | distance to nearest multiple of 5, in step units |
| `dist_round_10_steps` | distance to nearest multiple of 10, in step units |
| `dist_round_25_steps` | distance to nearest multiple of 25, in step units |
| `dist_round_50_steps` | distance to nearest multiple of 50, in step units |
| `dist_pdh_steps` | signed distance to previous-day high, in step units |
| `dist_pdl_steps` | signed distance to previous-day low, in step units |
| `dist_whigh_steps` | signed distance to prior-5-day high, in step units |
| `dist_wlow_steps` | signed distance to prior-5-day low, in step units |
| `dist_nearest_prior_level_steps` | distance to the nearest level from the prior 5 days, in step units (small = confluence) |
| `confluence_count` | # of prior-5-day levels within 0.5 step (stacked lines = stronger) |
| `gap_steps` | today's open − yesterday's close, in step units |
| `day_open` | the day's open price (reference; known at the open) |

**Label:** `outcome` — 1 = reacted, 0 = broke (from Phase 3).
