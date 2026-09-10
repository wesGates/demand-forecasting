"""
Run an experiment: train a model per series, compare it against baselines,
and report per-series and aggregate results.
 
The structure is deliberately plain. A `fit_predict` callable takes train and
test frames and returns predictions; swap in a different one and everything else
still works. That is the extension point - new models, new baselines, new
feature sets - without a plugin framework.
"""
 
from __future__ import annotations
 
from typing import Callable
 
import numpy as np
import pandas as pd
 
from src.evaluate import BASELINES, score, train_test_split_by_date
from src.features import feature_columns
 
 
# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
 
def fit_predict_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    **params,
) -> np.ndarray:
    """Train one XGBoost regressor on a single series and predict the test window."""
    from xgboost import XGBRegressor
 
    defaults = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=-1,
    )
    defaults.update(params)
 
    model = XGBRegressor(**defaults)
    model.fit(train[features], train["sales"])
    preds = model.predict(test[features])
    return np.clip(preds, 0, None)      # demand cannot be negative
 
 
MODELS: dict[str, Callable] = {
    "xgboost": fit_predict_xgboost,
}
 
 
# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
 
def run_experiment(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    baseline_name: str = "seasonal_naive",
    test_days: int = 28,
    min_train_days: int = 180,
    features: list[str] | None = None,
    **model_params,
) -> pd.DataFrame:
    """
    Fit one model per series and score it against a baseline.
 
    Returns one row per series with metrics for both, plus whether the model won.
    """
    features = features or feature_columns(df)
    model_fn = MODELS[model_name]
    baseline_fn = BASELINES[baseline_name]
 
    rows = []
    for series_id, g in df.groupby("id", sort=False):
        g = g.sort_values("date")
        train, test = train_test_split_by_date(g, test_days=test_days)
 
        # Rows with NaN lags (the start of each series) cannot be trained on.
        train_clean = train.dropna(subset=features)
        if len(train_clean) < min_train_days or test.empty:
            continue
 
        test_clean = test.dropna(subset=features)
        if test_clean.empty:
            continue
 
        y_true = test_clean["sales"].to_numpy(dtype=float)
        model_pred = model_fn(train_clean, test_clean, features, **model_params)
        base_pred = baseline_fn(train, test_clean)
 
        m = score(y_true, model_pred)
        b = score(y_true, np.asarray(base_pred, dtype=float))
 
        rows.append({
            "id": series_id,
            "n_train": len(train_clean),
            "n_test": len(test_clean),
            "mean_sales": float(train_clean["sales"].mean()),
            "zero_rate": float((train_clean["sales"] == 0).mean()),
            **{f"model_{k}": v for k, v in m.items()},
            **{f"base_{k}": v for k, v in b.items()},
        })
 
    res = pd.DataFrame(rows)
    if res.empty:
        return res
 
    res["won"] = res["model_rmse"] < res["base_rmse"]
    res["pct_improvement"] = (1 - res["model_rmse"] / res["base_rmse"].replace(0, np.nan)) * 100
    return res
 
 
def summarize(results: pd.DataFrame) -> dict[str, float]:
    """Headline numbers, in the shape you would put in a report."""
    if results.empty:
        return {}
    return {
        "series": len(results),
        "win_rate_pct": round(results["won"].mean() * 100, 1),
        "mean_improvement_pct": round(results["pct_improvement"].mean(), 1),
        "median_improvement_pct": round(results["pct_improvement"].median(), 1),
        "q1_improvement_pct": round(results["pct_improvement"].quantile(0.25), 1),
        "q3_improvement_pct": round(results["pct_improvement"].quantile(0.75), 1),
    }
 
 
# --------------------------------------------------------------------------- #
# Walk-forward runner
# --------------------------------------------------------------------------- #
 
def run_walk_forward(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    baseline_name: str = "seasonal_naive",
    horizon: int = 7,
    n_folds: int = 8,
    min_train_days: int = 180,
    features: list[str] | None = None,
    **model_params,
) -> pd.DataFrame:
    """
    Evaluate over several consecutive forecast windows instead of one.
 
    The test window is deliberately equal to the horizon. Features are lagged by
    `horizon` days, so a longer test window would let lags reach forward into the
    period being forecast - the model would be reading actuals it is supposed to
    predict. Keeping window == horizon makes every prediction a genuine
    `horizon`-day-ahead forecast.
 
    Multiple folds then recover the sample size a single long window would have
    given, without the leakage. This is the standard walk-forward setup: refit,
    forecast the next horizon, step forward, repeat.
    """
    features = features or feature_columns(df)
    model_fn = MODELS[model_name]
    baseline_fn = BASELINES[baseline_name]
 
    rows = []
    for series_id, g in df.groupby("id", sort=False):
        g = g.sort_values("date")
        last_date = g["date"].max()
 
        for fold in range(n_folds):
            # Fold 0 is the most recent window; each step moves one horizon back.
            test_end = last_date - pd.Timedelta(days=fold * horizon)
            test_start = test_end - pd.Timedelta(days=horizon - 1)
 
            train = g[g["date"] < test_start]
            test = g[(g["date"] >= test_start) & (g["date"] <= test_end)]
 
            train_clean = train.dropna(subset=features)
            test_clean = test.dropna(subset=features)
            if len(train_clean) < min_train_days or test_clean.empty:
                continue
 
            y_true = test_clean["sales"].to_numpy(dtype=float)
            model_pred = model_fn(train_clean, test_clean, features, **model_params)
            base_pred = np.asarray(baseline_fn(train, test_clean), dtype=float)
 
            m = score(y_true, model_pred)
            b = score(y_true, base_pred)
 
            rows.append({
                "id": series_id,
                "fold": fold,
                "test_start": test_start.date(),
                "n_train": len(train_clean),
                "mean_sales": float(train_clean["sales"].mean()),
                "zero_rate": float((train_clean["sales"] == 0).mean()),
                **{f"model_{k}": v for k, v in m.items()},
                **{f"base_{k}": v for k, v in b.items()},
            })
 
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["won"] = res["model_rmse"] < res["base_rmse"]
    res["pct_improvement"] = _pct_improvement(res)
    return res
 
 
def _pct_improvement(res: pd.DataFrame) -> pd.Series:
    """
    Percentage improvement over baseline, guarded against a zero baseline.
 
    On a sparse series the baseline occasionally predicts a 7-day window exactly
    (usually all zeros), giving base_rmse == 0. The naive formula then divides by
    zero and returns -inf, which poisons any mean taken over folds. Those folds
    carry no information about relative performance, so they are excluded.
    """
    base = res["base_rmse"].replace(0, np.nan)
    return (1 - res["model_rmse"] / base) * 100
 
 
def summarize_by_series(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse folds to one row per series - the view for spotting where it loses."""
    if results.empty:
        return results
    return (
        results.groupby("id")
        .agg(
            folds=("fold", "count"),
            mean_sales=("mean_sales", "mean"),
            zero_rate=("zero_rate", "mean"),
            model_rmse=("model_rmse", "mean"),
            base_rmse=("base_rmse", "mean"),
            win_rate=("won", "mean"),
            mean_improvement=("pct_improvement", "mean"),
            median_improvement=("pct_improvement", "median"),
        )
        .sort_values("zero_rate")
    )