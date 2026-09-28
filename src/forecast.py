"""
One forecast from one day, made by the backtest's own code path.

The walk-forward harness builds a Context for every fold and calls each
method on it. A daily job has to do the same thing for one origin, and the
usual way that goes wrong is a second implementation that drifts from the
first, so this module does not have one. It builds the harness's state,
picks the fold whose origin is the day asked for, and calls the same
functions the backtest calls. A test holds the rows equal to the backtest's.

Replay only, for now: the origin has to be a day the panel has seven days of
sales after, since the harness builds the target rows from the panel. A
live run needs calendar rows without sales, which the loader does not make
yet. That is the next step, not this one.
"""

from __future__ import annotations

import pandas as pd

from src.step1_problem import Config
from src.step5_evaluate import _fold_context, _forecast_rows, _run_state


def forecast_as_of(
    df: pd.DataFrame, cfg: Config, methods: list[str], as_of
) -> pd.DataFrame:
    """
    Seven-day forecasts for every series in `df` from the origin `as_of`, one
    row per series, method and day, in the harness's own row format.

    `as_of` must be one of the config's fold origins. With `fold_step=1` that
    is any day of the test year. ARIMA is not supported here: the harness
    chooses its orders in a separate pass before the folds run.
    """
    if "arima" in methods:
        raise ValueError(
            "arima chooses its orders in the harness's own pass; not available here"
        )
    as_of = pd.Timestamp(as_of)
    origins = list(cfg.fold_origins(df["date"].max()))
    if as_of not in origins:
        raise ValueError(f"{as_of.date()} is not a fold origin of this config")
    state = _run_state(df, cfg, methods)
    fold = origins.index(as_of)
    rows = []
    for series_id in state["ids"]:
        built = _fold_context(series_id, fold, state)
        if built is None:
            continue
        ctx, meta = built
        rows.extend(_forecast_rows(ctx, meta, methods))
    return pd.DataFrame(rows)
