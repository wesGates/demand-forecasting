"""
Step 5 - Using and evaluating the forecasts (FPP §1.6, step 5).

This is the harness. It walks every series through every fold, asks every
forecaster for its forecast, and records what actually happened. Everything
reported afterwards - RMSSE tables, win rates, the plots - derives from the one
long table `run_walk_forward` returns, so there is exactly one place where a
forecast meets an actual.

**Walk-forward validation** (FPP §5.10, "time series cross-validation"). The
folds come from `Config.fold_origins`: each is a forecast origin T, and the
`test_window` days after it are scored. Nothing after T is visible to any
forecaster - `Context` carries no future sales, and the learned model's
training rows are filtered by `target_date <= T` in step 4. The folds tile the
held-out period end to end, so the scored days are exactly the ones step 3's
classification never saw.

**RMSSE** (FPP §5.8). Root mean squared *scaled* error: the forecast's RMSE
divided by the RMSE a naive forecast would have had on the training data.

    RMSSE = RMSE(forecast) / sqrt( mean( (y_t - y_{t-lag})^2 ) over training )

A value of 1.0 means "as good as naively repeating the value from `lag` days
ago"; 0.8 means 20% better. Because the denominator is a property of the series
and not of the forecast, RMSSE is comparable across stores of very different
size - a 5-unit error means something different at 100 units/day than at 15.

Two details are `Config` decisions, and the config comments say why:
`rmsse_scale_lag` (FPP says `season` for seasonal data; M5 used 1) and
`rmsse_scale_window` ("pre_holdout": one denominator per series over all
training data before the first scored day, shared by every fold and method).

A property worth having straight: within one series-fold every method shares
the same denominator, so RMSSE **never changes which method wins that fold** -
it is a rescaling of RMSE. What it changes is how you aggregate across stores.

RMSE, MAE and bias are kept alongside as the plain-units view. Bias is
`mean(forecast - actual)`: positive means over-forecasting, and it matters in
its own right for replenishment, where a consistent 5% under-forecast is worse
than noisy-but-centred.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features import FEATURE_VERSION, build_supervised
from src.step1_problem import Config
from src.step4_models import ALL_FORECASTERS, BENCHMARKS, Context

# --------------------------------------------------------------------------- #
# Supervised matrices, cached
# --------------------------------------------------------------------------- #


def _supervised_path(cfg: Config, series_id: str):
    key = repr((FEATURE_VERSION, series_id, cfg.horizon, cfg.use_price))
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return cfg.cache_dir / f"supervised_v{FEATURE_VERSION}_{digest}.parquet"


def supervised_matrices(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """
    One supervised matrix per series, built once and cached to parquet.

    Building takes ~8s per series; reading back takes well under a second. The
    cache key includes `FEATURE_VERSION`, so editing a feature and forgetting
    to rebuild cannot serve stale rows - the old file simply stops matching.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for series_id, series in df.groupby("id", sort=True, observed=True):
        path = _supervised_path(cfg, series_id)
        if path.exists():
            out[series_id] = pd.read_parquet(path)
        else:
            matrix = build_supervised(series, cfg.horizon, use_price=cfg.use_price)
            matrix.to_parquet(path, index=False)
            out[series_id] = matrix
    return out


# --------------------------------------------------------------------------- #
# The RMSSE denominator
# --------------------------------------------------------------------------- #


def naive_scale(y, lag: int) -> float:
    """
    RMSE of a lag-`lag` naive forecast over the given values - the RMSSE
    denominator of FPP §5.8, in its squared-error form.

    Returns NaN if there is nothing to difference. A series that never changes
    has a zero denominator; that is left as 0 so the resulting RMSSE is `inf`
    and visibly wrong, rather than silently replaced.
    """
    y = np.asarray(y, dtype=float)
    if len(y) <= lag:
        return np.nan
    diff = y[lag:] - y[:-lag]
    return float(np.sqrt(np.mean(diff**2)))


# --------------------------------------------------------------------------- #
# The walk-forward
# --------------------------------------------------------------------------- #


def run_walk_forward(
    df: pd.DataFrame, cfg: Config, methods: list[str] | None = None, progress: bool = True
) -> pd.DataFrame:
    """
    Every forecast from every method for every series-fold, with the actuals.

    Returns one row per (series, fold, method, horizon day) with columns
    `actual`, `forecast` and `scale` (the RMSSE denominator that applies to
    that series-fold). This long table is the single source every score and
    plot is computed from.

    `methods` defaults to every benchmark and every model. Pass a subset to
    iterate on one method quickly.
    """
    methods = list(methods or ALL_FORECASTERS)
    unknown = set(methods) - set(ALL_FORECASTERS)
    if unknown:
        raise ValueError(
            f"unknown method(s) {sorted(unknown)}; choose from {list(ALL_FORECASTERS)}"
        )

    last_date = df["date"].max()
    origins = cfg.fold_origins(last_date)
    holdout = cfg.holdout_start(last_date)
    matrices = supervised_matrices(df, cfg)

    rows = []
    groups = df.groupby("id", sort=True, observed=True)
    for series_id, series in tqdm(groups, desc="series", disable=not progress):
        series = series.sort_values("date").reset_index(drop=True)
        item_id, store_id = series["item_id"].iloc[0], series["store_id"].iloc[0]

        # The rows a learned model may train on: this series alone, or every
        # series that shares its pool key. Which is `Config.pool_by`'s decision.
        if cfg.pool_by is None:
            pool = matrices[series_id]
        else:
            key = series[cfg.pool_by].iloc[0]
            members = df.loc[df[cfg.pool_by] == key, "id"].unique()
            pool = pd.concat([matrices[m] for m in members], ignore_index=True)

        # "pre_holdout": one denominator for this series, from everything
        # before the first scored day. Shared by every fold and every method.
        fixed_scale = naive_scale(
            series.loc[series["date"] < holdout, "sales"], cfg.rmsse_scale_lag
        )

        for fold, origin in enumerate(origins):
            history = series[series["date"] <= origin]
            window_end = origin + pd.Timedelta(days=cfg.test_window)
            targets = series[(series["date"] > origin) & (series["date"] <= window_end)]
            if len(history) < cfg.min_train_days or len(targets) < cfg.test_window:
                continue

            if cfg.rmsse_scale_window == "per_fold":
                scale = naive_scale(history["sales"], cfg.rmsse_scale_lag)
            else:
                scale = fixed_scale

            ctx = Context(
                history=history,
                targets=targets.drop(columns=["sales"]),
                origin=origin,
                horizon=cfg.horizon,
                series_id=series_id,
                season=cfg.season,
                train_pool=pool,
                pool_by=cfg.pool_by,
                seed=cfg.seed,
            )
            actual = targets["sales"].to_numpy(dtype=float)
            dates = targets["date"].to_numpy()

            for name in methods:
                forecast = np.asarray(ALL_FORECASTERS[name](ctx), dtype=float)[
                    : len(targets)
                ]
                for h, (day, a, f) in enumerate(
                    zip(dates, actual, forecast, strict=True), 1
                ):
                    rows.append(
                        {
                            "id": series_id,
                            "item_id": item_id,
                            "store_id": store_id,
                            "fold": fold,
                            "origin": origin,
                            "method": name,
                            "kind": "benchmark" if name in BENCHMARKS else "model",
                            "horizon": h,
                            "target_date": day,
                            "actual": a,
                            "forecast": f,
                            "scale": scale,
                        }
                    )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

FOLD_KEYS = ["id", "item_id", "store_id", "fold", "origin", "method", "kind"]


def score_folds(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    One row per series-fold-method: RMSE, MAE, bias, and RMSSE.

    RMSSE here is `rmse / scale`, where `scale` came with the predictions. All
    methods in the same series-fold share that scale, which is why RMSSE
    cannot change who wins a fold - only how folds and stores are compared.
    """

    def one(g: pd.DataFrame) -> pd.Series:
        err = g["forecast"] - g["actual"]
        rmse = float(np.sqrt(np.mean(err**2)))
        scale = float(g["scale"].iloc[0])
        return pd.Series(
            {
                "rmse": rmse,
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "rmsse": rmse / scale if scale > 0 else np.inf,
                "n_days": len(g),
            }
        )

    return (
        predictions.groupby(FOLD_KEYS, sort=True, observed=True)
        .apply(one, include_groups=False)
        .reset_index()
    )


def rmsse_by_store(scores: pd.DataFrame) -> pd.DataFrame:
    """Stores down, methods across, mean RMSSE over folds. The headline table."""
    table = scores.pivot_table(
        index="store_id", columns="method", values="rmsse", aggfunc="mean"
    )
    # Order stores by volume (busiest first) and methods by overall RMSSE.
    store_order = (
        scores.groupby("store_id", observed=True)["rmse"].mean().sort_values().index
    )
    method_order = table.mean(axis=0).sort_values().index
    return table.reindex(index=store_order, columns=method_order)


def win_rates(scores: pd.DataFrame) -> pd.DataFrame:
    """
    For every method, the share of series-folds where it beat each benchmark.

    Rows are methods, columns are benchmarks, values are fractions. 0.5 means
    "no better than the benchmark"; the row for a benchmark against itself is
    left blank. Win rate is a blunt instrument - it says how *often*, not by
    how *much* - which is why it sits beside RMSSE rather than replacing it.
    """
    wide = scores.pivot_table(
        index=["id", "fold"], columns="method", values="rmsse", aggfunc="first"
    )
    out = {}
    for bench in BENCHMARKS:
        if bench not in wide:
            continue
        out[bench] = {
            m: float((wide[m] < wide[bench]).mean()) if m != bench else np.nan
            for m in wide.columns
        }
    return pd.DataFrame(out).sort_index()


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """
    One row per method: mean and median RMSSE across all series-folds, and
    mean bias. Sorted best first. The single table to quote.
    """
    out = (
        scores.groupby(["method", "kind"], observed=True)
        .agg(
            rmsse_mean=("rmsse", "mean"),
            rmsse_median=("rmsse", "median"),
            bias_mean=("bias", "mean"),
            n_folds=("rmsse", "size"),
        )
        .reset_index()
        .sort_values("rmsse_mean")
        .reset_index(drop=True)
    )
    return out


if __name__ == "__main__":
    from src.step1_problem import STUDY_ITEMS
    from src.step2_data import load_panel

    cfg = Config(item_ids=STUDY_ITEMS)
    print(cfg.describe(), "\n")

    panel = load_panel(cfg, verbose=False)
    predictions = run_walk_forward(panel, cfg)
    scores = score_folds(predictions)

    print(
        f"\n{len(predictions):,} forecast-days, {len(scores):,} series-fold-method scores\n"
    )
    print("=== summary: mean RMSSE per method, best first ===")
    print(summarise(scores).round(3).to_string(index=False))
    print("\n=== RMSSE by store (busiest first) x method (best first) ===")
    print(rmsse_by_store(scores).round(3).to_string())
    print("\n=== win rate: rows beat columns, share of series-folds ===")
    print(win_rates(scores).round(2).to_string())
