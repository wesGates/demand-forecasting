"""
From forecast to order quantity (FPP §5.5, §5.9). The ordering prototype.

This module predates item 1. It is the calibrated-quantile code, and it is
still the only order-quantity code in the repository. Item 1 compared it
with an XGBoost trained on the quantile objective and found that the
calibrated quantiles match the fitted ones within 0.002 relative pinball at
a twenty-sixth of the fitting cost. The quantile work continues from here,
on the pooled, level-relative model's own residuals.

What it does, and why:

- The order is a weekly total. A Sunday order covers Monday to Sunday, so
  the error that reaches the shelf is the sum of seven forecasts minus the
  sum of seven actuals. Daily errors partly cancel inside a week and the
  daily RMSSE never sees it. `weekly_totals` builds the quantity the order
  actually is.
- Ordering the mean runs out about half the time. Demand lands above a
  point forecast as often as below it. The quantity to order is a quantile,
  the level that covers demand with a chosen probability τ. Which τ is an
  economic question, the critical fractile τ = Cu / (Cu + Co). Above 0.5
  when running out costs more than holding (ambient grocery), below 0.5 when
  holding costs more (fresh produce that ends up in the bin). M5 items are
  anonymised, so the study reports a range of τ and whether the method
  ranking changes across it.
- Where the quantiles come from. None of the methods produces a
  distribution. FPP §5.5's route works for all of them: a method's own past
  errors are its uncertainty. For each store and method, the weekly errors
  over a calibration year give an error distribution, and the τ-quantile
  forecast for a later week is the point forecast plus the τ-quantile of
  those errors. A method with tighter errors earns a smaller add-on, and a
  biased method gets corrected for free because its mean error is inside the
  distribution.
- Calibration weeks come before scored weeks. Estimating the quantile on
  the weeks it is judged on gives 90% coverage by construction. The run
  that feeds this module scores two years, the first calibrates and the
  second is judged, so every holiday appears in both.
- Pinball loss (FPP §5.9, the quantile score) is the metric, with the
  book's factor of two so that τ = 0.5 equals the absolute error:

      2 τ (y - q)        if y >= q   (under-forecast, a shortfall)
      2 (1 - τ) (q - y)  if y <  q   (over-forecast, a surplus)

  At τ = 0.9 a unit of shortfall costs nine times a unit of surplus. The
  asymmetry the decision cares about sits inside the metric.
- Pinball losses at different τ are on different scales and are never
  averaged across τ. Within one τ they are in units, so the report divides by
  the store's mean weekly sales, for the same reason RMSSE exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The service levels reported. Symmetric around 0.5 so the perishable and the
# ambient direction are both covered, plus the 0.9 an ambient item would use.
TAUS = (0.3, 0.5, 0.7, 0.9)

WEEK_KEYS = ["id", "item_id", "store_id", "fold", "origin", "week_kind", "method", "kind"]


def weekly_totals(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    One row per series-fold-method with the week's actual and forecast totals.

    Closure days are dropped first, as in `score_folds`, so both totals cover
    the same days. `error` is actual minus forecast. Positive is a shortfall,
    the sign a replenishment decision cares about.
    """
    p = predictions
    if "closure" in p:
        p = p[~p["closure"].astype(bool)]
    out = (
        p.groupby(WEEK_KEYS, observed=True)
        .agg(
            actual=("actual", "sum"),
            forecast=("forecast", "sum"),
            n_days=("actual", "size"),
            scale=("scale", "first"),
        )
        .reset_index()
    )
    out["error"] = out["actual"] - out["forecast"]
    return out


def calibrate(weekly: pd.DataFrame, scored_from: pd.Timestamp, taus=TAUS) -> pd.DataFrame:
    """
    Per (series, method, τ), the empirical τ-quantile of weekly errors over the
    calibration folds, meaning every fold whose origin is before `scored_from`.

    Returns `offset`, the add-on to a point forecast, and `n_calib`, how many
    weeks it rests on.
    """
    c = weekly[weekly["origin"] < scored_from]
    rows = []
    for (sid, method), g in c.groupby(["id", "method"], observed=True):
        for tau in taus:
            rows.append(
                {
                    "id": sid,
                    "method": method,
                    "tau": tau,
                    "offset": float(g["error"].quantile(tau)),
                    "n_calib": len(g),
                }
            )
    return pd.DataFrame(rows)


def calibrate_expanding(
    weekly: pd.DataFrame, scored_from: pd.Timestamp, taus=TAUS
) -> pd.DataFrame:
    """
    Like `calibrate`, but the calibration window grows. Each scored fold's
    offset uses every fold before it, earlier scored folds included. This is
    what a live system would do, recalibrate as errors accumulate. Returns one
    row per (series, method, fold, τ).
    """
    rows = []
    for (sid, method), g in weekly.groupby(["id", "method"], observed=True):
        g = g.sort_values("origin")
        errors = g["error"].to_numpy()
        origins = g["origin"].to_numpy()
        for i, origin in enumerate(origins):
            if origin < scored_from:
                continue
            past = errors[:i]
            for tau in taus:
                rows.append(
                    {
                        "id": sid,
                        "method": method,
                        "fold": int(g["fold"].iloc[i]),
                        "tau": tau,
                        "offset": float(np.quantile(past, tau)),
                        "n_calib": len(past),
                    }
                )
    return pd.DataFrame(rows)


def quantile_forecasts(
    weekly: pd.DataFrame, offsets: pd.DataFrame, scored_from: pd.Timestamp
) -> pd.DataFrame:
    """
    The scored weeks with a τ-quantile forecast `q` for every τ in `offsets`,
    the point forecast plus that series-method's offset. `offsets` may be
    fixed (from `calibrate`) or per fold (from `calibrate_expanding`).
    """
    s = weekly[weekly["origin"] >= scored_from]
    keys = ["id", "method"] + (["fold"] if "fold" in offsets else [])
    out = s.merge(offsets, on=keys, how="inner")
    out["q"] = out["forecast"] + out["offset"]
    return out


def pinball(actual, q, tau) -> np.ndarray:
    """FPP §5.9's quantile score, with the factor of two (τ = 0.5 gives |error|)."""
    actual, q, tau = (np.asarray(x, dtype=float) for x in (actual, q, tau))
    diff = actual - q
    return np.where(diff >= 0, 2 * tau * diff, 2 * (1 - tau) * (-diff))


def score_quantiles(qf: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-week quantile scores to the output of `quantile_forecasts`:
    `pinball` (units), `pinball_rel` (pinball over the store's mean weekly
    sales in the scored period), `covered` (demand fell at or below q),
    `shortfall` and `surplus` (units).
    """
    qf = qf.copy()
    qf["pinball"] = pinball(qf["actual"], qf["q"], qf["tau"])
    level = qf.groupby("id", observed=True)["actual"].transform("mean")
    qf["pinball_rel"] = qf["pinball"] / level
    qf["covered"] = qf["actual"] <= qf["q"]
    qf["shortfall"] = np.maximum(qf["actual"] - qf["q"], 0)
    qf["surplus"] = np.maximum(qf["q"] - qf["actual"], 0)
    return qf


def summarise_quantiles(
    scored: pd.DataFrame, week_kind: str | None = None
) -> pd.DataFrame:
    """
    Per method and τ, the mean relative pinball loss, achieved coverage, and
    the mean shortfall and surplus in units per week. Coverage should sit
    near τ. The gap between them is how well the calibration carried over from
    one year to the next.
    """
    if week_kind is not None:
        scored = scored[scored["week_kind"] == week_kind]
    return (
        scored.groupby(["method", "kind", "tau"], observed=True)
        .agg(
            pinball_rel=("pinball_rel", "mean"),
            coverage=("covered", "mean"),
            shortfall=("shortfall", "mean"),
            surplus=("surplus", "mean"),
            n_weeks=("covered", "size"),
        )
        .reset_index()
    )


def ranking_by_tau(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Methods down, τ across, relative pinball in the cells, best first at
    τ = 0.5. Answers whether the model ranking changes with the cost asymmetry.
    """
    table = summary.pivot_table(index="method", columns="tau", values="pinball_rel")
    order = (
        table[0.5].sort_values().index
        if 0.5 in table
        else table.mean(axis=1).sort_values().index
    )
    return table.reindex(order)


# --------------------------------------------------------------------------- #
# Daily quantiles: native (fitted) against calibrated (point + error quantile)
# --------------------------------------------------------------------------- #

# Methods whose forecasts are quantiles already, keyed by family. A family's
# entries map τ to the registered method name that produces that quantile.
NATIVE_QUANTILE_FAMILIES: dict[str, dict[float, str]] = {
    "xgboost_q": {
        0.3: "xgboost_q30",
        0.5: "xgboost_q50",
        0.7: "xgboost_q70",
        0.9: "xgboost_q90",
    }
}

DAY_KEYS = [
    "id",
    "item_id",
    "store_id",
    "fold",
    "origin",
    "week_kind",
    "horizon",
    "target_date",
]


def daily_calibrated_quantiles(
    predictions: pd.DataFrame,
    scored_from: pd.Timestamp,
    methods: tuple[str, ...] = (
        "xgboost",
        "ets",
        "arima",
        "seasonal_naive",
        "moving_average_28",
    ),
    taus=TAUS,
) -> pd.DataFrame:
    """
    Daily τ-quantile forecasts for the point-forecast methods, the same
    treatment as the fitted quantile model but applied after the fact.

    For each series, method and horizon, the τ-quantile of the daily errors
    (actual minus forecast) over the calibration folds is the add-on, and on
    the scored folds q = forecast + add-on. Per horizon, because the error
    spread grows with days ahead. One row per scored day, method and τ.
    """
    p = predictions[predictions["method"].isin(methods)]
    if "closure" in p:
        p = p[~p["closure"].astype(bool)]
    p = p.assign(error=p["actual"] - p["forecast"])
    calib = p[p["origin"] < scored_from]
    scored = p[p["origin"] >= scored_from]
    offsets = (
        calib.groupby(["id", "method", "horizon"], observed=True)["error"]
        .quantile(list(taus))
        .rename("offset")
        .reset_index()
        .rename(columns={"level_3": "tau"})
    )
    out = scored.merge(offsets, on=["id", "method", "horizon"], how="inner")
    out["q"] = out["forecast"] + out["offset"]
    out["source"] = "calibrated"
    return out[DAY_KEYS + ["method", "kind", "tau", "actual", "forecast", "q", "source"]]


def daily_native_quantiles(
    predictions: pd.DataFrame,
    scored_from: pd.Timestamp,
    families=NATIVE_QUANTILE_FAMILIES,
) -> pd.DataFrame:
    """
    Daily quantiles from methods that produce them directly. One row per
    scored day, family and τ; `method` is the family name so it lines up
    with the calibrated table.
    """
    if "closure" in predictions:
        predictions = predictions[~predictions["closure"].astype(bool)]
    frames = []
    for family, by_tau in families.items():
        for tau, name in by_tau.items():
            part = predictions[
                (predictions["method"] == name) & (predictions["origin"] >= scored_from)
            ]
            if part.empty:
                continue
            part = part.assign(
                method=family, tau=tau, q=part["forecast"], source="fitted"
            )
            frames.append(
                part[
                    DAY_KEYS
                    + ["method", "kind", "tau", "actual", "forecast", "q", "source"]
                ]
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score_daily_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Pinball (units and ÷ the store's mean daily sales), coverage, shortfall, surplus per day."""
    f = frame.copy()
    f["pinball"] = pinball(f["actual"], f["q"], f["tau"])
    level = f.groupby("id", observed=True)["actual"].transform("mean")
    f["pinball_rel"] = f["pinball"] / level
    f["covered"] = f["actual"] <= f["q"]
    f["shortfall"] = np.maximum(f["actual"] - f["q"], 0)
    f["surplus"] = np.maximum(f["q"] - f["actual"], 0)
    return f


def summarise_daily(scored: pd.DataFrame, week_kind: str | None = None) -> pd.DataFrame:
    """Per method, source and τ: mean relative pinball, coverage, shortfall and surplus per day."""
    if week_kind is not None:
        scored = scored[scored["week_kind"] == week_kind]
    return (
        scored.groupby(["method", "source", "tau"], observed=True)
        .agg(
            pinball_rel=("pinball_rel", "mean"),
            coverage=("covered", "mean"),
            shortfall=("shortfall", "mean"),
            surplus=("surplus", "mean"),
            n_days=("covered", "size"),
        )
        .reset_index()
    )


def daily_ranking_by_tau(summary: pd.DataFrame) -> pd.DataFrame:
    """Methods down, τ across, relative daily pinball in the cells, best at 0.5 first."""
    table = summary.pivot_table(
        index=["method", "source"], columns="tau", values="pinball_rel"
    )
    order = (
        table[0.5].sort_values().index
        if 0.5 in table
        else table.mean(axis=1).sort_values().index
    )
    return table.reindex(order)


def weekly_native_quantiles(
    predictions: pd.DataFrame,
    scored_from: pd.Timestamp,
    families=NATIVE_QUANTILE_FAMILIES,
) -> pd.DataFrame:
    """
    Weekly order-up-to levels from a fitted quantile model, the sum of its
    daily τ-quantiles over the fold. A sum of quantiles is not the quantile of
    a sum. For τ above 0.5 it over-covers and below 0.5 it under-covers, so
    this sits beside the calibrated weekly table and the coverage column says
    what the sum actually delivered.
    """
    daily = daily_native_quantiles(predictions, scored_from, families)
    if daily.empty:
        return daily
    keys = [
        "id",
        "item_id",
        "store_id",
        "fold",
        "origin",
        "week_kind",
        "method",
        "kind",
        "tau",
    ]
    w = (
        daily.groupby(keys, observed=True)
        .agg(actual=("actual", "sum"), q=("q", "sum"), n_days=("actual", "size"))
        .reset_index()
    )
    w["forecast"] = np.nan
    w["offset"] = np.nan
    return w


if __name__ == "__main__":
    from src.step1_problem import STUDY_ITEMS, Config
    from src.step2_data import load_panel
    from src.step4_models import ALL_FORECASTERS
    from src.step5_evaluate import run_walk_forward

    # Two years of folds. The first calibrates and the second is judged. The
    # scored year is the same layout the point-forecast tables use.
    scored = Config(item_ids=STUDY_ITEMS)
    both = Config(item_ids=STUDY_ITEMS, n_folds=2 * scored.n_folds)
    panel = load_panel(both, verbose=False)
    scored_from = scored.holdout_start(panel["date"].max()) - pd.Timedelta(days=1)

    # The two-year run is for the point-forecast methods only. The fitted
    # quantile model needs no calibration year, and fitting it on 104 folds
    # would cost an hour for nothing.
    fitted_names = [n for f in NATIVE_QUANTILE_FAMILIES.values() for n in f.values()]
    point_methods = [m for m in ALL_FORECASTERS if m not in fitted_names]
    predictions = run_walk_forward(panel, both, progress=False, methods=point_methods)
    weekly = weekly_totals(predictions)
    print(
        f"{len(weekly):,} series-fold-method weeks; scored from origin {scored_from.date()}"
    )

    print(
        "\n=== weekly-total bias (forecast - actual, units/week) and share of weeks under-forecast ==="
    )
    w = weekly[weekly["origin"] >= scored_from].copy()
    w["under"] = w["error"] > 0
    print(
        w.groupby("method", observed=True)
        .agg(
            bias=("error", lambda e: -e.mean()),
            sd=("error", "std"),
            under=("under", "mean"),
        )
        .sort_values("sd")
        .round(2)
        .to_string()
    )

    offsets = calibrate(weekly, scored_from)
    scored_q = score_quantiles(quantile_forecasts(weekly, offsets, scored_from))
    summary = summarise_quantiles(scored_q)
    print(
        "\n=== relative pinball loss by method x tau (fixed calibration year), best at 0.5 first ==="
    )
    print(ranking_by_tau(summary).round(3).to_string())
    print("\n=== achieved coverage by method x tau (target = tau) ===")
    print(
        summary.pivot_table(index="method", columns="tau", values="coverage")
        .round(2)
        .to_string()
    )
    for week in ("normal", "holiday"):
        print(f"\n=== relative pinball, {week} weeks ===")
        print(ranking_by_tau(summarise_quantiles(scored_q, week)).round(3).to_string())

    expanding = calibrate_expanding(weekly, scored_from)
    scored_x = score_quantiles(quantile_forecasts(weekly, expanding, scored_from))
    print("\n=== relative pinball, expanding calibration window ===")
    print(ranking_by_tau(summarise_quantiles(scored_x)).round(3).to_string())

    # Fitted quantiles against calibrated ones, at the daily level where the
    # fitted model's outputs are quantiles by construction. The fitted methods
    # run on the scored year only. The calibrated ones need year 1 as well.
    fitted = run_walk_forward(panel, scored, progress=False, methods=fitted_names)
    daily = pd.concat(
        [
            daily_calibrated_quantiles(predictions, scored_from),
            daily_native_quantiles(fitted, scored_from),
        ],
        ignore_index=True,
    )
    scored_d = score_daily_quantiles(daily)
    print(
        "\n=== DAILY relative pinball by method x tau: fitted quantiles vs calibrated ==="
    )
    print(daily_ranking_by_tau(summarise_daily(scored_d)).round(3).to_string())
    print("\n=== DAILY coverage by method x tau (target = tau) ===")
    print(
        summarise_daily(scored_d)
        .pivot_table(index=["method", "source"], columns="tau", values="coverage")
        .round(2)
        .to_string()
    )
    for week in ("normal", "holiday"):
        print(f"\n=== DAILY relative pinball, {week} weeks ===")
        print(daily_ranking_by_tau(summarise_daily(scored_d, week)).round(3).to_string())
    wn = score_quantiles(weekly_native_quantiles(fitted, scored_from))
    print(
        "\n=== WEEKLY: fitted quantiles summed over the week (see docstring caveat) ==="
    )
    print(
        summarise_quantiles(wn)
        .pivot_table(index="method", columns="tau", values=["pinball_rel", "coverage"])
        .round(3)
        .to_string()
    )
