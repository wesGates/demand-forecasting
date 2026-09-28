"""
Croston, SBA and TSB against the book's worked example and their own
definitions (FPP §13.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import base
from src.models.intermittent import ALPHA, INTERMITTENT, _croston_rate
from src.step4_models import MODELS, MODULE_OF


def _ctx(values, horizon=7) -> base.Context:
    values = np.asarray(values, dtype=float)
    dates = pd.date_range("2013-01-07", periods=len(values) + horizon, freq="D")
    return base.Context(
        history=pd.DataFrame({"date": dates[: len(values)], "sales": values}),
        targets=pd.DataFrame({"date": dates[len(values) :]}),
        origin=dates[len(values) - 1],
        horizon=horizon,
        series_id="S",
    )


# The book's soy-sauce example: sales of 1, 2, 1 four days apart, alpha 0.1,
# initialised at the first size and gap. Smoothed size 1.09, gap 4.00.
BOOK = [0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0]


def test_registered():
    from src.models import intermittent

    for name in INTERMITTENT:
        assert name in MODELS and MODULE_OF[name] is intermittent


def test_croston_matches_the_book():
    assert np.isclose(_croston_rate(np.array(BOOK, dtype=float)), 1.09 / 4.0)
    got = INTERMITTENT["croston"](_ctx(BOOK))
    assert got.shape == (7,)
    np.testing.assert_allclose(got, 1.09 / 4.0)


def test_sba_deflates_croston_by_half_alpha():
    ctx = _ctx(BOOK)
    np.testing.assert_allclose(
        INTERMITTENT["sba"](ctx), INTERMITTENT["croston"](ctx) * (1 - ALPHA / 2)
    )


def test_daily_seller_forecasts_its_level():
    ctx = _ctx(np.full(60, 10.0))
    np.testing.assert_allclose(INTERMITTENT["croston"](ctx), 10.0)
    np.testing.assert_allclose(INTERMITTENT["tsb"](ctx), 10.0)
    # SBA deflates whether or not the series is intermittent; on a daily
    # seller that is a known cost of the correction.
    np.testing.assert_allclose(INTERMITTENT["sba"](ctx), 10.0 * (1 - ALPHA / 2))


def test_tsb_decays_after_sales_stop_and_croston_does_not():
    alive = np.tile([0, 0, 3, 0], 30)
    dead = np.concatenate([alive, np.zeros(60)])
    croston_alive, croston_dead = (
        INTERMITTENT["croston"](_ctx(alive))[0],
        INTERMITTENT["croston"](_ctx(dead))[0],
    )
    tsb_alive, tsb_dead = (
        INTERMITTENT["tsb"](_ctx(alive))[0],
        INTERMITTENT["tsb"](_ctx(dead))[0],
    )
    assert croston_dead == croston_alive  # no sale, no update
    assert tsb_dead < 0.2 * tsb_alive  # 60 zero days at alpha 0.1 cut the chance to almost nothing


def test_under_two_sales_falls_back_and_says_so():
    base.fallbacks.clear()
    got = INTERMITTENT["croston"](_ctx([0] * 40 + [2] + [0] * 10))
    assert got.shape == (7,) and np.isfinite(got).all()
    assert base.fallbacks == ["croston: 28-day mean"]
    base.fallbacks.clear()
