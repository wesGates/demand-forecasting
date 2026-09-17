"""
Feature engineering.

One rule governs this entire module:

    **Every feature for a target day must be computable from data available at
    the forecast origin T.**

We forecast days T+1 .. T+h from a single origin T, in FPP's notation
`ŷ_{T+h|T}`. So a row for target day T+h may use:

  - sales up to and including day T, and nothing after it;
  - anything that is known in advance for T+h - the calendar, SNAP dates, the
    horizon h itself.

Getting this wrong is *leakage*, and it produces validation scores that look
excellent and collapse in production. It has already happened once on this
project, so `assert_no_leakage` checks it mechanically rather than by reading.

**Why origin-based rather than target-based lags.** A simpler-looking scheme is
to shift every feature by the horizon and let each target day carry its own
lags. That is leakage-free too, but it means each target day is forecast from a
*different* origin, and the first day of the window throws away six days of data
you would actually have. Here every row in a fold shares one origin, exactly as
a replenishment run does: stand on Sunday, order for Monday through Sunday.

The consequence is that the lag columns are identical across the h rows of a
fold, and `horizon` is what distinguishes them - so `horizon` must be a feature.

Lag choices come from the autocorrelation in step 3 rather than convention: the
weekly harmonics (7, 14, 21, 28) dominate and lag 1 is independently strong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lags measured back from the forecast origin T. Justified by the ACF in step 3.
LAGS = (1, 2, 3, 7, 14, 21, 28)

# Rolling windows ending at T - recent level and volatility.
ROLL_WINDOWS = (7, 28, 56)

# Same-weekday history: the mean of the last k occurrences of the target's
# weekday, at or before T. The learned version of a seasonal naive forecast,
# and usually the strongest single feature in daily retail.
DOW_WINDOWS = (4, 8)


# --------------------------------------------------------------------------- #
# Calendar - known in advance, so safe for any target date
# --------------------------------------------------------------------------- #


def calendar_features(rows: pd.DataFrame, use_price: bool = False) -> pd.DataFrame:
    """
    Features that are known in advance for the target date.

    Every one of these is knowable before the day arrives, so none of them can
    leak: a calendar does not depend on what was sold, and in this dataset the
    shelf price is set a week ahead.

    Price is opt-in via `Config.use_price` - see that docstring for why it is
    off for the current study item.
    """
    out = pd.DataFrame(index=rows.index)
    date = rows["date"]

    out["dow"] = date.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["day_of_month"] = date.dt.day
    out["month"] = date.dt.month
    out["day_of_year"] = date.dt.dayofyear
    out["snap"] = rows["snap"].astype("int8")
    out["is_event"] = rows["event_name_1"].notna().astype("int8")
    if use_price:
        out["sell_price"] = rows["sell_price"].astype("float32")
    return out


# --------------------------------------------------------------------------- #
# History - everything here must end at the origin
# --------------------------------------------------------------------------- #


def history_features(history: pd.DataFrame, target_dates: pd.Series) -> pd.DataFrame:
    """
    Summaries of one series' sales, all ending at the forecast origin.

    Parameters
    ----------
    history
        Rows for a single series, sorted by date, containing **only** dates at
        or before the origin T. The caller is responsible for that cut; this
        function trusts it and `assert_no_leakage` verifies it.
    target_dates
        The days being forecast. Used only for the same-weekday features, which
        depend on which weekday is being predicted.

    Returns
    -------
    One row per target date. Lag and rolling columns are constant across those
    rows by construction - they describe the origin, not the target - which is
    why `horizon` must also be a feature.
    """
    sales = history["sales"].to_numpy(dtype=float)
    n = len(sales)
    out = pd.DataFrame(index=target_dates.index)

    # Lags counted back from the origin: lag_1 is the origin day itself.
    for k in LAGS:
        out[f"lag_{k}"] = sales[-k] if n >= k else np.nan

    # Level and volatility over windows ending at the origin.
    for w in ROLL_WINDOWS:
        window = sales[-w:] if n >= 1 else np.array([])
        out[f"roll_mean_{w}"] = window.mean() if len(window) else np.nan
        out[f"roll_std_{w}"] = window.std(ddof=1) if len(window) > 1 else np.nan

    # Trend: is the recent level above or below the longer-run level?
    out["trend_7_56"] = out["roll_mean_7"] / out["roll_mean_56"].replace(0, np.nan) - 1

    # Same-weekday history - the learned seasonal naive. This one genuinely
    # varies per target day, because each target day has its own weekday.
    hist_dow = history["date"].dt.dayofweek.to_numpy()
    target_dow = target_dates.dt.dayofweek.to_numpy()
    for k in DOW_WINDOWS:
        values = [
            sales[hist_dow == d][-k:].mean() if (hist_dow == d).sum() else np.nan
            for d in target_dow
        ]
        out[f"dow_mean_{k}"] = values

    return out


def build_fold_features(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    origin: pd.Timestamp,
    use_price: bool = False,
) -> pd.DataFrame:
    """
    Assemble the feature matrix for one fold of one series.

    `history` must already be cut at `origin`; `targets` are the days being
    forecast. The returned frame carries `horizon` (1..h), which is what
    separates otherwise identical rows.
    """
    if len(history) and history["date"].max() > origin:
        raise ValueError(
            f"history extends to {history['date'].max().date()}, past the "
            f"origin {origin.date()}. Cut it before calling."
        )

    feats = pd.concat(
        [
            calendar_features(targets, use_price=use_price),
            history_features(history, targets["date"]),
        ],
        axis=1,
    )
    feats["horizon"] = (targets["date"] - origin).dt.days.astype("int16")
    return feats


# --------------------------------------------------------------------------- #
# The safeguard
# --------------------------------------------------------------------------- #


def assert_no_leakage(
    history: pd.DataFrame, targets: pd.DataFrame, origin: pd.Timestamp
) -> None:
    """
    Refuse to build features if anything reaches past the forecast origin.

    This is the structural check the project's safeguards call for: it proves
    the boundary by comparing dates, rather than inferring it statistically from
    a model's error. A statistical test can only say "this looks suspicious";
    this says "this row used data from after T", with the date.

    Raises
    ------
    ValueError
        If the history extends past the origin, or any target is not after it.
    """
    if len(history):
        latest = history["date"].max()
        if latest > origin:
            raise ValueError(
                f"LEAK: history reaches {latest.date()}, past origin {origin.date()}."
            )
    if len(targets):
        earliest = targets["date"].min()
        if earliest <= origin:
            raise ValueError(
                f"LEAK: target day {earliest.date()} is not after origin "
                f"{origin.date()} - it is inside the training period."
            )


# --------------------------------------------------------------------------- #
# The supervised training matrix
# --------------------------------------------------------------------------- #

# Shortest history that can fill every feature. Derived from the window
# constants rather than typed in, so lengthening a window above cannot leave
# this silently too small - which would put NaN features in the first rows
# with no error to say so.
MIN_HISTORY = max(max(LAGS), max(ROLL_WINDOWS), 7 * max(DOW_WINDOWS))

# Bump whenever the feature definitions above change. Step 5 caches each
# series' supervised matrix to parquet keyed on this, so an edited feature
# cannot silently be served from a file built by the old definition.
FEATURE_VERSION = 1


def build_supervised(
    series: pd.DataFrame, horizon: int, use_price: bool = False
) -> pd.DataFrame:
    """
    Every (origin, horizon) pair for one series, as a supervised learning table.

    Walks the series day by day. At each origin it builds the same features a
    real forecast would have, and attaches the target that actually occurred.
    Built once per series and then sliced per fold, so the cost is paid once.

    The returned frame carries `origin_date` and `target_date` alongside the
    features. Those two columns are what make leak-free training mechanical:

      - a row may be **trained on** only if its `target_date` is at or before
        the fold's origin, because only then was the answer observable;
      - a row is **predicted** when its `origin_date` equals the fold's origin.

    Both filters are pure date comparisons, which is the point - there is no
    step where a human has to reason about which rows are safe.
    """
    series = series.sort_values("date").reset_index(drop=True)
    frames = []

    for i in range(MIN_HISTORY, len(series) - horizon):
        origin = series["date"].iloc[i]
        targets = series.iloc[i + 1 : i + 1 + horizon]

        feats = build_fold_features(
            series.iloc[: i + 1], targets, origin, use_price=use_price
        )
        feats["origin_date"] = origin
        feats["target_date"] = targets["date"].to_numpy()
        feats["target"] = targets["sales"].to_numpy(dtype=float)
        frames.append(feats)

    out = pd.concat(frames, ignore_index=True)
    out["id"] = series["id"].iloc[0]
    for col in POOL_ID_COLS:
        out[col] = series[col].iloc[0]
    return out


# Identity columns. Bookkeeping for a per-series model; features for a pooled
# one, where the model needs them to tell the series in its pool apart.
POOL_ID_COLS = ("store_id", "item_id")

NON_FEATURE_COLS = (
    "id",
    *POOL_ID_COLS,
    "date",
    "sales",
    "origin_date",
    "target_date",
    "target",
)


def supervised_feature_columns(
    pool: pd.DataFrame, pool_by: str | None = None
) -> list[str]:
    """
    Model input columns from a supervised matrix.

    `pool_by` is `Config.pool_by`. When it is None the model sees one series and
    identity columns would be constant, so they are left out. When it is set
    the pool spans several series and identity becomes a feature - handed over
    as native categoricals by the model code.
    """
    cols = [c for c in pool.columns if c not in NON_FEATURE_COLS]
    return [*cols, *POOL_ID_COLS] if pool_by is not None else cols
