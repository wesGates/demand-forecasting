"""
Diagnostic plots for demand forecasting.
 
These are the standard views practitioners actually use. Each one answers a
specific question, and each one catches a specific failure that a summary table
would hide.
 
    from src.plots import (
        plot_series, plot_forecast_vs_actual, plot_residuals,
        plot_win_scatter, plot_feature_importance, plot_split_diagram,
    )
"""
 
from __future__ import annotations
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
 
# --------------------------------------------------------------------------- #
# 1. Look at the data before modelling anything
# --------------------------------------------------------------------------- #
 
def plot_series(df: pd.DataFrame, series_id: str, ax=None):
    """
    Raw history for one series.
 
    Question it answers: what am I actually dealing with? Trend, seasonality,
    gaps, spikes, how often it sells zero. Skipping this step is how people end
    up modelling a series that stopped selling two years ago.
    """
    g = df[df["id"] == series_id].sort_values("date")
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
 
    ax.plot(g["date"], g["sales"], lw=0.7, color="#2E5F8A")
    zero_share = (g["sales"] == 0).mean()
    ax.set_title(f"{series_id}   mean={g['sales'].mean():.2f}   zeros={zero_share:.0%}")
    ax.set_ylabel("units")
    ax.grid(alpha=0.25)
    return ax
 
 
def plot_weekly_profile(df: pd.DataFrame, series_id: str, ax=None):
    """
    Average sales by day of week.
 
    Question: is there a weekly pattern worth capturing? If this is flat, the
    seasonal-naive baseline has nothing to work with and lag features on a
    7-day cycle will not help much either.
    """
    g = df[df["id"] == series_id].copy()
    g["dow"] = g["date"].dt.dayofweek
    prof = g.groupby("dow")["sales"].mean()
 
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(prof.index, prof.values, color="#2E5F8A")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_title("Average sales by weekday")
    ax.grid(alpha=0.25, axis="y")
    return ax
 
 
# --------------------------------------------------------------------------- #
# 2. The split - the plot that would have caught the leakage bug
# --------------------------------------------------------------------------- #
 
def plot_split_diagram(
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    horizon: int,
    ax=None,
):
    """
    Draw the train window, the test window, and where a lag feature points.
 
    Question: does any feature reach into the test window? Each red arrow shows
    where `lag_{horizon}` sources its value for a given test day. If an arrow
    lands inside the shaded test region, the model is reading data it is meant
    to be predicting - that is leakage, and it is visible at a glance.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2.6))
 
    ax.axvspan(train_end - pd.Timedelta(days=40), train_end,
               color="#2E5F8A", alpha=0.15, label="train")
    ax.axvspan(test_start, test_end, color="#B03A2E", alpha=0.18, label="test")
    ax.axvline(train_end, color="#333", lw=1.2, ls="--")
 
    test_days = pd.date_range(test_start, test_end)
    for d in test_days:
        src = d - pd.Timedelta(days=horizon)
        inside = src >= test_start
        ax.annotate(
            "", xy=(src, 0.5), xytext=(d, 0.5),
            arrowprops=dict(
                arrowstyle="->", lw=1.1,
                color="#B03A2E" if inside else "#2E7D5A",
                alpha=0.9,
            ),
        )
 
    n_bad = sum((d - pd.Timedelta(days=horizon)) >= test_start for d in test_days)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(
        f"lag_{horizon} sources   |   green = from train (safe), "
        f"red = from inside test (LEAK)   |   {n_bad} of {len(test_days)} leaking"
    )
    ax.legend(loc="upper left", fontsize=8)
    return ax
 
 
# --------------------------------------------------------------------------- #
# 3. Forecast vs actual - the single most-used plot in forecasting
# --------------------------------------------------------------------------- #
 
def plot_forecast_vs_actual(
    dates, y_true, y_pred, y_base=None, title="", ax=None
):
    """
    Actuals against predictions over the test window.
 
    Question: where does the model actually go wrong? A summary RMSE hides
    whether the error is spread evenly or driven by two catastrophic days.
    Plotting the baseline alongside shows *why* the model is better, not just
    that it is.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
 
    ax.plot(dates, y_true, "o-", color="#333", lw=1.6, ms=4, label="actual")
    ax.plot(dates, y_pred, "s--", color="#2E5F8A", lw=1.4, ms=3.5, label="model")
    if y_base is not None:
        ax.plot(dates, y_base, "^:", color="#B0803A", lw=1.2, ms=3.5, label="baseline")
 
    ax.set_title(title)
    ax.set_ylabel("units")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return ax
 
 
def plot_residuals(y_true, y_pred, ax=None):
    """
    Prediction minus actual, over time and as a distribution.
 
    Question: is the model biased? Residuals should scatter around zero with no
    pattern. If they sit mostly above zero the model over-forecasts, which in
    inventory terms means carrying stock you did not need. Structure in the
    residuals - drift, or a repeating shape - means there is signal the model
    has not captured.
    """
    resid = np.asarray(y_pred) - np.asarray(y_true)
    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    else:
        axes = ax
 
    axes[0].axhline(0, color="#B03A2E", lw=1)
    axes[0].plot(resid, "o", ms=3, color="#2E5F8A", alpha=0.7)
    axes[0].set_title(f"Residuals over time   (mean bias = {resid.mean():+.2f})")
    axes[0].grid(alpha=0.25)
 
    axes[1].hist(resid, bins=25, color="#2E5F8A", alpha=0.8)
    axes[1].axvline(0, color="#B03A2E", lw=1)
    axes[1].set_title("Residual distribution")
    axes[1].grid(alpha=0.25, axis="y")
    return axes
 
 
# --------------------------------------------------------------------------- #
# 4. Where does the model win and lose?
# --------------------------------------------------------------------------- #
 
def plot_win_scatter(results: pd.DataFrame, ax=None):
    """
    Improvement over baseline against how sparse each series is.
 
    Question: what *kind* of product does this model help with? The expected
    shape is a downward slope - dense, regular series are predictable and gain a
    lot; sparse intermittent ones gain little or lose. Being able to state that
    boundary is far more valuable than a single average win rate, and it is the
    honest way to report where an approach does not work.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
 
    colors = ["#2E7D5A" if w else "#B03A2E" for w in results["won"]]
    ax.scatter(results["zero_rate"], results["pct_improvement"],
               c=colors, s=45, alpha=0.8, edgecolor="white")
    ax.axhline(0, color="#333", lw=1, ls="--")
    ax.set_xlabel("share of days with zero sales  (sparser ->)")
    ax.set_ylabel("% improvement over baseline")
    ax.set_title("Where the model helps, and where it does not")
    ax.grid(alpha=0.25)
    return ax
 
 
def plot_error_by_volume(results: pd.DataFrame, ax=None):
    """
    Model vs baseline error against series volume.
 
    Question: is the model simply better on big sellers? Points below the
    diagonal are wins. Sizing by volume shows whether the wins are concentrated
    in the series that actually matter commercially.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
 
    lim = max(results["model_rmse"].max(), results["base_rmse"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color="#333", lw=1, ls="--")
    ax.scatter(results["base_rmse"], results["model_rmse"],
               s=results["mean_sales"] * 12 + 20,
               c="#2E5F8A", alpha=0.65, edgecolor="white")
    ax.set_xlabel("baseline RMSE")
    ax.set_ylabel("model RMSE")
    ax.set_title("Below the line = model wins\n(point size = sales volume)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.25)
    return ax
 
 
# --------------------------------------------------------------------------- #
# 5. What is the model using?
# --------------------------------------------------------------------------- #
 
def plot_feature_importance(model, features: list[str], top_n: int = 15, ax=None):
    """
    Which features the model relies on most.
 
    Question: is it using what I would expect? Recent lags and rolling means
    normally dominate. If something odd is at the top, that is worth
    investigating - unexpected importance is a common early symptom of leakage.
    """
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    imp = imp.tail(top_n)
 
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, max(3, 0.32 * len(imp))))
    ax.barh(imp.index, imp.values, color="#2E5F8A")
    ax.set_title("Feature importance")
    ax.grid(alpha=0.25, axis="x")
    return ax