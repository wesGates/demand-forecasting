"""
Load the M5 competition data and reshape it into a tidy daily panel.

M5 ships sales in *wide* format: one row per store-item, one column per day
(d_1 ... d_1941). Almost everything downstream wants *long* format: one row per
store-item-date. This module does that reshape and joins on the calendar and
price tables.

Usage
-----
    from src.data import load_m5

    df = load_m5("data", store_id="CA_1", cat_id="FOODS")
    print(df.head())
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_m5(
    data_dir: str | Path = "data",
    store_id: str | None = "CA_1",
    cat_id: str | None = "FOODS",
    dept_id: str | None = None,
) -> pd.DataFrame:
    """
    Return a long-format daily panel of unit sales with calendar and price columns.

    Parameters
    ----------
    data_dir
        Directory holding the three M5 csv files.
    store_id, cat_id, dept_id
        Optional filters applied *before* the reshape. Filtering first matters:
        the full dataset is 30,490 series x 1,941 days = ~59M rows once melted,
        which is slow and memory-hungry. Subsetting to one store and category
        keeps iteration fast without changing the methodology.
        Pass None to skip a filter.

    Returns
    -------
    DataFrame with one row per (id, date), sorted by id then date.
    """
    data_dir = Path(data_dir)

    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(data_dir / "sell_prices.csv")

    # ---- filter before melting -------------------------------------------
    if store_id is not None:
        sales = sales[sales["store_id"] == store_id]
    if cat_id is not None:
        sales = sales[sales["cat_id"] == cat_id]
    if dept_id is not None:
        sales = sales[sales["dept_id"] == dept_id]

    if sales.empty:
        raise ValueError("No rows left after filtering - check store_id/cat_id.")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    # ---- wide -> long ----------------------------------------------------
    df = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    # ---- attach real dates and calendar features -------------------------
    # SNAP flags are per-state; pick the column matching each row's state.
    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    df = df.merge(calendar[cal_cols], on="d", how="left")

    snap_lookup = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    df["snap"] = 0
    for state, col in snap_lookup.items():
        mask = df["state_id"] == state
        df.loc[mask, "snap"] = df.loc[mask, col]
    df = df.drop(columns=list(snap_lookup.values()))

    # ---- attach prices ---------------------------------------------------
    # Prices are weekly (keyed on wm_yr_wk), so this fans out to daily.
    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    return df


def trim_leading_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the run of zeros before each item's first ever sale.

    In M5 an item's row exists for the whole history even if the product had not
    launched yet, so early zeros mean "not stocked" rather than "no demand". Left
    in, they teach the model that the item sells nothing and drag every rolling
    feature down. A missing price is the reliable signal for this: no price on
    file means the item was not being sold that week.
    """
    df = df.copy()
    df["_has_price"] = df["sell_price"].notna()
    first_sold = (
        df[df["_has_price"]]
        .groupby("id")["date"]
        .min()
        .rename("first_sold_date")
    )
    df = df.merge(first_sold, on="id", how="left")
    df = df[df["date"] >= df["first_sold_date"]]
    return df.drop(columns=["_has_price", "first_sold_date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = load_m5("data", store_id="CA_1", cat_id="FOODS")
    print(f"rows: {len(df):,}")
    print(f"series: {df['id'].nunique():,}")
    print(f"dates: {df['date'].min().date()} -> {df['date'].max().date()}")
    print()
    print(df.head())
    print()

    trimmed = trim_leading_zeros(df)
    dropped = len(df) - len(trimmed)
    print(f"pre-launch rows dropped: {dropped:,} ({dropped / len(df):.1%})")
    print(f"zero-sales share before: {(df['sales'] == 0).mean():.1%}")
    print(f"zero-sales share after:  {(trimmed['sales'] == 0).mean():.1%}")