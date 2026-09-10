"""
Baselines and metrics.

A model is only interesting relative to something dumber. These are the dumber
things. Every baseline takes the same arguments and returns predictions aligned
to `test`, so adding a new one is a single function plus a registry entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, season: int = 7) -> np.ndarray:
    """
    Predict each day with the value from `season` days earlier.

    For daily retail this is a genuinely strong baseline: it captures the weekly
    pattern for free, and it is what a person would do with no tools.
    """
    hist = pd.concat([train, test]).sort_values("date")["sales"].to_numpy()
    n_test = len(test)
    n_train = len(train)
    out = np.empty(n_test)
    for i in range(n_test):
        src = n_train + i - season
        out[i] = hist[src] if src >= 0 else np.nan
    return out


def moving_average(train: pd.DataFrame, test: pd.DataFrame, window: int = 28) -> np.ndarray:
    """Flat forecast at the mean of the last `window` training days."""
    value = train["sales"].tail(window).mean()
    return np.full(len(test), value)


def naive_last(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Flat forecast at the final training value."""
    return np.full(len(test), train["sales"].iloc[-1])


BASELINES = {
    "seasonal_naive": seasonal_naive,
    "moving_average": moving_average,
    "naive_last": naive_last,
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted absolute percentage error: total error divided by total volume.

    Preferred over MAPE for intermittent demand, because MAPE divides by actuals
    and actuals are frequently zero.
    """
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom else np.nan


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error. Positive means over-forecasting."""
    return float(np.mean(y_pred - y_true))


METRICS = {"rmse": rmse, "mae": mae, "wape": wape, "bias": bias}


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """All metrics at once, ignoring positions where either side is NaN."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {name: np.nan for name in METRICS}
    return {name: fn(y_true[mask], y_pred[mask]) for name, fn in METRICS.items()}


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def train_test_split_by_date(
    df: pd.DataFrame, test_days: int = 28
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically: the last `test_days` become the test set.

    Never split time series randomly. A random split lets the model see the
    future while predicting the past, which inflates validation scores and tells
    you nothing about real performance.
    """
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    return df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()


def walk_forward_folds(
    df: pd.DataFrame, n_folds: int = 3, test_days: int = 28
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Successive chronological folds, each testing on a later window.

    Fold 1 trains on the least data and tests earliest; the final fold tests on
    the most recent window. This is the time-series analogue of cross-validation.
    """
    folds = []
    last = df["date"].max()
    for k in reversed(range(n_folds)):
        test_end = last - pd.Timedelta(days=k * test_days)
        test_start = test_end - pd.Timedelta(days=test_days - 1)
        train = df[df["date"] < test_start].copy()
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        if not train.empty and not test.empty:
            folds.append((train, test))
    return folds
