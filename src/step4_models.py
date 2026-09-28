"""
Step 4, choosing and fitting models (FPP §1.6). The registry of everything
that produces a forecast, all behind one interface:

    forecaster(ctx: Context) -> np.ndarray of length ctx.horizon

`Context` holds what a forecaster may see and nothing from after the origin.
An earlier build handed a baseline the test frame and one of them quietly
indexed into it. That is the reason for the interface.

Two flat dicts. Adding a method is a function plus an entry.

Benchmarks (FPP §5.2, plus two of our own) are what a model has to beat.
See `benchmarks.py` for why there are six.

Models under test:

  xgboost       gradient-boosted trees on the supervised feature matrix
  xgboost_rel   the same trees on a level-relative target (item 5)
  ets           Holt-Winters exponential smoothing (FPP ch. 8)
  arima         seasonal ARIMA with holiday and SNAP regressors (FPP ch. 9-10)
  xgboost_q*    quantile-objective trees, kept as item 1's comparator
  xgboost_tweedie  the trees under a Tweedie loss, for items with many zero days
  croston, sba, tsb  the intermittent-demand benchmarks (FPP §13.2)

ETS and ARIMA are the two classical families FPP puts forward. the ETS
implementation takes no regressors, so it cannot be told a holiday is
coming, while ARIMA can. Keeping both shows what the calendar information is
worth to a classical model. A plain ARMA is left out on purpose. It is ARIMA
without differencing or a seasonal term, and on daily data with a weekly
cycle it would need very high orders to imitate what one seasonal term does.

Pooling is a training scope rather than a model, and it lives on
`Config.pool_by`. Per-store and pooled XGBoost are the same function with
the same features. The only difference is which rows step 5 puts into
`Context.train_pool` and whether the model is given the store identity.
"""

from __future__ import annotations

from src.models import (
    arima_model,
    arma_model,
    benchmarks,
    ets_model,
    intermittent,
    xgboost_model,
    xgboost_poisson,
    xgboost_quantile,
    xgboost_rel_recent,
    xgboost_relative,
    xgboost_tweedie,
)
from src.models.arima_model import (
    _arima_exog,
    _select_arima_order,
    arima_orders,
    fit_predict_arima,
)
from src.models.base import Context, Forecaster, _flat
from src.models.benchmarks import BENCHMARKS
from src.models.ets_model import fit_predict_ets
from src.models.xgboost_model import INNER_VAL_DAYS, XGB_PARAMS, fit_predict_xgboost

MODELS: dict[str, Forecaster] = {
    "xgboost": fit_predict_xgboost,
    "ets": fit_predict_ets,
    "arima": fit_predict_arima,
    "xgboost_rel": xgboost_relative.fit_predict_xgboost_relative,
    "xgboost_poisson": xgboost_poisson.fit_predict_xgboost_poisson,
    "xgboost_rel_recent": xgboost_rel_recent.fit_predict_xgboost_rel_recent,
    "arima_plain": arma_model.fit_predict_arima_plain,
    "arma": arma_model.fit_predict_arma,
    "xgboost_tweedie": xgboost_tweedie.fit_predict_xgboost_tweedie,
    **intermittent.INTERMITTENT,
    **xgboost_quantile.QUANTILE_MODELS,
}

# Everything that produces a forecast, benchmarks and models together.
ALL_FORECASTERS: dict[str, Forecaster] = {**BENCHMARKS, **MODELS}

# Which module each method lives in. The predictions cache hashes that
# module (plus the shared base, harness, features, loader and Config), so
# editing XGBoost keeps the cached ARIMA run. That one takes 10 min.
MODULE_OF = {
    **{name: benchmarks for name in BENCHMARKS},
    "xgboost": xgboost_model,
    "ets": ets_model,
    "arima": arima_model,
    "xgboost_rel": xgboost_relative,
    "xgboost_poisson": xgboost_poisson,
    "xgboost_rel_recent": xgboost_rel_recent,
    "arima_plain": arma_model,
    "arma": arma_model,
    "xgboost_tweedie": xgboost_tweedie,
    **{name: intermittent for name in intermittent.INTERMITTENT},
    **{name: xgboost_quantile for name in xgboost_quantile.QUANTILE_MODELS},
}


def reset_run_state() -> None:
    """
    Forget everything memoised within a run: ARIMA's chosen orders, the
    pooled fits and the quantile model's per-fold fits. The harness calls
    this at the start of every run so a second layout in the same process
    starts clean.
    """
    arima_orders.clear()
    xgboost_quantile.reset()
    xgboost_model.reset()
    xgboost_relative.reset()
    xgboost_rel_recent.reset()
    arma_model.reset()


__all__ = [
    "ALL_FORECASTERS",
    "BENCHMARKS",
    "Context",
    "Forecaster",
    "INNER_VAL_DAYS",
    "MODELS",
    "MODULE_OF",
    "XGB_PARAMS",
    "_arima_exog",
    "_flat",
    "_select_arima_order",
    "arima_orders",
    "reset_run_state",
    "fit_predict_arima",
    "fit_predict_ets",
    "fit_predict_xgboost",
]
