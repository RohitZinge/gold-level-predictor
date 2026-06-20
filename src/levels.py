#!/usr/bin/env python3
"""
Phase 2 (formula only) — the 21 ladder levels for a trading day.

Every day's 21 levels are derived purely from that day's OPEN price:

    digit_sum   = sum of the digits of int(day_open)
    step_factor = (sqrt(day_open) * (digit_sum / 10) / 2) / 20000
    level(n)    = day_open * (1 + n * step_factor)   for n = -10 .. +10

n = 0 is the open (centre); n = -10..-1 are the 10 levels below; n = +1..+10
the 10 above. That's 21 levels total. Each level is tagged with:

    step_number : -10 .. +10  (0 = the open)
    type        : "open" at the centre (0), "thick" on even steps, "thin" on odd

Worked example (open = 4300):
    digit_sum   = 4+3+0+0 = 7
    step_factor = (sqrt(4300) * 0.7 / 2) / 20000 ≈ 0.00114755
    step size   ≈ 4300 * 0.00114755 ≈ $4.93  (spacing between adjacent levels)
    lowest  (n=-10) ≈ 4250.7    centre (n=0) = 4300    highest (n=+10) ≈ 4349.3

These must match the same numbers drawn on your TradingView chart.
"""

from __future__ import annotations

import math

# 10 levels below + the open + 10 levels above = 21 levels.
N_STEPS = 10


def digit_sum(day_open: float) -> int:
    """Sum of the base-10 digits of int(day_open). E.g. 4300 -> 4+3+0+0 = 7."""
    return sum(int(d) for d in str(abs(int(day_open))))


def step_factor(day_open: float) -> float:
    """Fractional spacing between adjacent levels (the formula's core)."""
    return (math.sqrt(day_open) * (digit_sum(day_open) / 10) / 2) / 20000


def step_size(day_open: float) -> float:
    """Dollar spacing between two adjacent levels = day_open * step_factor."""
    return day_open * step_factor(day_open)


def level_type(step_number: int) -> str:
    """'open' at the centre (step 0), 'thick' on even steps, 'thin' on odd."""
    if step_number == 0:
        return "open"
    return "thick" if step_number % 2 == 0 else "thin"


def get_day_levels(day_open: float) -> list[dict]:
    """
    Build the 21 ladder levels for a day given its open price.

    Returns a list of 21 dicts ordered by step_number from -10 to +10
    (which is also ascending price, since step_factor > 0):

        {"step_number": -10..+10, "price": float, "type": "thick"|"thin"}

    Raises ValueError if day_open is not positive.
    """
    if day_open <= 0:
        raise ValueError(f"day_open must be positive, got {day_open!r}")

    factor = step_factor(day_open)
    return [
        {
            "step_number": n,
            "price": day_open * (1 + n * factor),
            "type": level_type(n),
        }
        for n in range(-N_STEPS, N_STEPS + 1)
    ]


if __name__ == "__main__":
    # Quick manual sanity print for the worked example.
    for lv in get_day_levels(4300):
        print(f"  step {lv['step_number']:+3d}  {lv['price']:10.3f}  {lv['type']}")
    print(f"\nstep size ≈ ${step_size(4300):.2f}")
