"""
`forecast_as_of` gives the same rows as the backtest for the same origin,
per store and pooled, because it runs the same code.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.forecast import forecast_as_of
from src.step1_problem import Config
from src.step5_evaluate import run_walk_forward
from tests.conftest import make_panel

KEYS = ["id", "method", "target_date"]


@pytest.mark.parametrize("pool_by", [None, "item_id"])
def test_matches_the_backtest_row_for_row(tmp_path, pool_by):
    panel = make_panel(stores={"S1": ("CA", 40.0), "S2": ("TX", 25.0)}, noise_sd=2.0)
    cfg = Config(
        n_folds=5, fold_step=1, min_train_days=200, pool_by=pool_by, cache_dir=tmp_path
    )
    methods = ["seasonal_naive", "ets", "croston", "xgboost", "xgboost_tweedie"]
    back = run_walk_forward(panel, cfg, methods=methods, progress=False, use_cache=False)
    as_of = cfg.fold_origins(panel["date"].max())[2]
    live = forecast_as_of(panel, cfg, methods, as_of)

    want = back[back["origin"] == as_of].sort_values(KEYS).reset_index(drop=True)
    got = live.sort_values(KEYS).reset_index(drop=True)
    assert len(got) == len(want) > 0
    assert_frame_equal(got[want.columns], want, check_dtype=False)


def test_refuses_a_day_that_is_not_an_origin(tmp_path):
    panel = make_panel(stores={"S1": ("CA", 40.0)})
    cfg = Config(n_folds=3, fold_step=7, min_train_days=200, cache_dir=tmp_path)
    origin = cfg.fold_origins(panel["date"].max())[0]
    with pytest.raises(ValueError):
        forecast_as_of(panel, cfg, ["ets"], origin + pd.Timedelta(days=1))
