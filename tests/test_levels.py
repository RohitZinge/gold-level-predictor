"""Tests for the 21-level ladder formula (src/levels.py). No data needed."""

import pytest

from src.levels import get_day_levels, step_size


def test_open_4300_known_case():
    """open = 4300 -> step ≈ $5, lowest ≈ 4251, centre = 4300, highest ≈ 4349."""
    day_open = 4300
    levels = get_day_levels(day_open)

    # Exactly 21 levels, with step numbers -10..+10 in order.
    assert len(levels) == 21
    assert [lv["step_number"] for lv in levels] == list(range(-10, 11))

    prices = [lv["price"] for lv in levels]
    by_step = {lv["step_number"]: lv for lv in levels}

    # Step ≈ $5 (the spacing between adjacent levels).
    assert step_size(day_open) == pytest.approx(4.93, abs=0.1)
    # And that spacing should be roughly constant between neighbours.
    assert (prices[1] - prices[0]) == pytest.approx(step_size(day_open))

    # Centre = the open exactly.
    assert by_step[0]["price"] == pytest.approx(4300.0)

    # Lowest ≈ 4251, highest ≈ 4349 (prices ascend with step number).
    assert prices == sorted(prices)
    assert min(prices) == pytest.approx(4251, abs=1)   # 4250.7
    assert max(prices) == pytest.approx(4349, abs=1)   # 4349.3
    assert by_step[-10]["price"] is prices[0] or prices[0] == min(prices)

    # "open" at the centre, thick on even steps, thin on odd steps.
    for lv in levels:
        s = lv["step_number"]
        expected = "open" if s == 0 else ("thick" if s % 2 == 0 else "thin")
        assert lv["type"] == expected
    assert by_step[0]["type"] == "open"
    assert by_step[1]["type"] == "thin"
    assert by_step[2]["type"] == "thick"
    assert by_step[-1]["type"] == "thin"
    assert by_step[10]["type"] == "thick"


def test_rejects_non_positive_open():
    with pytest.raises(ValueError):
        get_day_levels(0)
    with pytest.raises(ValueError):
        get_day_levels(-100)
