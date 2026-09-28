"""
Step 4 - the six benchmarks compute what their names say (FPP Ch. 5).

Expected values are derived from the FPP definitions, not from the code:
mean, naive y_T, seasonal naive y_{T+h-m(k+1)}, drift y_T + h (y_T - y_1)/(T-1),
a 28-day moving average, and the same weekday 52 weeks ago.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.step4_models import ALL_FORECASTERS, BENCHMARKS, MODELS, Context


def _ctx(values, horizon=7, season=7) -> Context:
    values = np.asarray(values, dtype=float)
    dates = pd.date_range("2013-01-07", periods=len(values) + horizon, freq="D")
    history = pd.DataFrame({"date": dates[: len(values)], "sales": values})
    targets = pd.DataFrame({"date": dates[len(values) :]})
    return Context(
        history=history,
        targets=targets,
        origin=history["date"].iloc[-1],
        horizon=horizon,
        series_id="S",
        season=season,
    )


def test_registry_has_six_benchmarks_and_the_expected_models():
    from src.models.xgboost_quantile import TAUS, method_name

    assert set(BENCHMARKS) == {
        "mean",
        "naive",
        "seasonal_naive",
        "seasonal_naive_364",
        "drift",
        "moving_average_28",
    }
    quantile_names = {method_name(t) for t in TAUS}
    assert set(MODELS) == {
        "xgboost", "ets", "arima", "xgboost_rel",
        "arima_plain", "arma",  # the two weaker ARIMA relatives, baselines
        "xgboost_poisson", "xgboost_rel_recent",  # dev-suite experiments
        "xgboost_tweedie", "croston", "sba", "tsb",  # item 11, the intermittent set
    } | quantile_names
    assert set(ALL_FORECASTERS) == set(BENCHMARKS) | set(MODELS)
    assert not set(BENCHMARKS) & set(MODELS)


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_constant_series_gives_constant_forecast(name):
    got = BENCHMARKS[name](_ctx(np.full(400, 10.0)))
    assert got.shape == (7,)
    np.testing.assert_allclose(got, 10.0)


def test_benchmarks_on_a_ramp():
    n = 400
    y = np.arange(1, n + 1, dtype=float)  # y_t = t
    ctx = _ctx(y)
    last = float(n)
    np.testing.assert_allclose(BENCHMARKS["naive"](ctx), np.full(7, last))
    np.testing.assert_allclose(BENCHMARKS["drift"](ctx), last + np.arange(1, 8))
    np.testing.assert_allclose(BENCHMARKS["mean"](ctx), np.full(7, y.mean()))
    # h = 1..7 -> the matching weekday of the last complete week: t = n-6 .. n
    np.testing.assert_allclose(BENCHMARKS["seasonal_naive"](ctx), y[-7:])
    np.testing.assert_allclose(
        BENCHMARKS["moving_average_28"](ctx), np.full(7, y[-28:].mean())
    )
    # target day t -> value at t - 364
    np.testing.assert_allclose(
        BENCHMARKS["seasonal_naive_364"](ctx), np.arange(n + 1, n + 8) - 364.0
    )


def test_seasonal_naive_wraps_when_horizon_exceeds_the_season():
    y = np.arange(1, 29, dtype=float)
    got = BENCHMARKS["seasonal_naive"](_ctx(y, horizon=10))
    np.testing.assert_allclose(got, np.concatenate([y[-7:], y[-7:-4]]))


def test_seasonal_naive_364_falls_back_to_weekly_on_short_history():
    y = np.arange(1, 201, dtype=float)
    ctx = _ctx(y)
    np.testing.assert_allclose(
        BENCHMARKS["seasonal_naive_364"](ctx), BENCHMARKS["seasonal_naive"](ctx)
    )


def test_drift_is_the_line_through_first_and_last_observation():
    y = np.array([10.0, 0.0, 30.0, 40.0, 20.0, 60.0, 70.0])  # first 10, last 70
    slope = (70.0 - 10.0) / 6
    np.testing.assert_allclose(
        BENCHMARKS["drift"](_ctx(y, horizon=3)), 70 + slope * np.arange(1, 4)
    )


def test_context_carries_no_target_sales():
    ctx = _ctx(np.full(50, 1.0))
    assert "sales" not in ctx.targets.columns
