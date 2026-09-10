"""
Feature engineering for daily demand forecasting.

The one rule that matters here: every feature must be computable at prediction
time. If you are standing on day T forecasting day T+7, you do not yet know what
happened on day T+3. So every lag and rolling window is shifted by at least the
forecast horizon. Getting this wrong is called *leakage*, and it produces
validation scores that look excellent and collapse in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lags are expressed as an offset *on top of* the horizon, so the same config
# stays leakage-safe whatever horizon you forecast at.
DEFAULT_LAG_OFFSETS = (0, 1, 7, 14, 21, 28)
DEFAULT_ROLL_WINDOWS = (7, 28)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, month, and event flags. All known in advance, so no shift."""
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek           # 0 = Monday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # An event on the day, plus proximity - demand often moves before a holiday,
    # not on it.
    df["is_event"] = df["event_name_1"].notna().astype(int)
    df["days_to_event"] = _days_to_next_event(df)
    return df


def _days_to_next_event(df: pd.DataFrame) -> pd.Series:
    """Days until the next calendar event, capped at 28."""
    event_dates = df.loc[df["event_name_1"].notna(), "date"].drop_duplicates().sort_values()
    if event_dates.empty:
        return pd.Series(28, index=df.index)
    idx = np.searchsorted(event_dates.values, df["date"].values, side="left")
    idx = np.clip(idx, 0, len(event_dates) - 1)
    delta = (event_dates.values[idx] - df["date"].values).astype("timedelta64[D]").astype(int)
    return pd.Series(np.clip(delta, 0, 28), index=df.index)


def add_lag_features(
    df: pd.DataFrame,
    horizon: int,
    lag_offsets: tuple[int, ...] = DEFAULT_LAG_OFFSETS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
) -> pd.DataFrame:
    """
    Add lagged sales and rolling statistics, shifted to respect the horizon.

    With horizon=7 and offset=0 you get lag_7 - the value one week before the day
    being predicted, which is the most recent figure you would actually have.
    """
    df = df.sort_values(["id", "date"]).copy()
    g = df.groupby("id")["sales"]

    for off in lag_offsets:
        lag = horizon + off
        df[f"lag_{lag}"] = g.shift(lag)

    # Rolling stats are computed on already-shifted data so the window never
    # includes anything from inside the forecast gap.
    shifted = g.shift(horizon)
    for w in roll_windows:
        df[f"roll_mean_{w}"] = shifted.groupby(df["id"]).transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
        df[f"roll_std_{w}"] = shifted.groupby(df["id"]).transform(
            lambda s, w=w: s.rolling(w, min_periods=2).std()
        )

    # Share of recent days with no sale - a direct measure of intermittency,
    # which is the thing that decides whether ML or a simple rule wins.
    df["zero_rate_28"] = shifted.groupby(df["id"]).transform(
        lambda s: s.eq(0).rolling(28, min_periods=1).mean()
    )
    return df


def add_price_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Price level and recent change. Prices are known ahead, so no shift needed."""
    df = df.copy()
    g = df.groupby("id")["sell_price"]
    df["price_change_7"] = df["sell_price"] / g.shift(7) - 1
    df["price_vs_avg"] = df["sell_price"] / g.transform("mean") - 1
    return df


def make_features(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Full feature pipeline. Returns the panel with feature columns added."""
    df = add_calendar_features(df)
    df = add_lag_features(df, horizon=horizon)
    df = add_price_features(df, horizon=horizon)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The columns to actually feed the model - everything engineered, nothing raw."""
    prefixes = ("lag_", "roll_mean_", "roll_std_")
    named = [
        "dow", "is_weekend", "day_of_month", "week_of_year",
        "is_event", "days_to_event", "snap",
        "sell_price", "price_change_7", "price_vs_avg",
        "zero_rate_28",
    ]
    lagged = [c for c in df.columns if c.startswith(prefixes)]
    return [c for c in named + lagged if c in df.columns]
