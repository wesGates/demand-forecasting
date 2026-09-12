"""
Step 2 - Gathering information (FPP §1.6, step 2).

Turns the three raw M5 files into one tidy daily panel: one row per
store-item-date, with the calendar and price information already attached.

Three things in here are not obvious and all three matter:

1. **Filter before melting.** The full panel is 30,490 series x 1,941 days -
   about 59 million rows once reshaped. Subsetting first keeps a notebook
   session responsive without changing the method at all.

2. **SNAP flags are per state.** The calendar carries `snap_CA`, `snap_TX` and
   `snap_WI` as separate columns. Each row must read the column for its own
   store's state, not all three.

3. **Pre-launch zeros are not zero demand.** An item's row spans the whole
   history even before the product was stocked. Those leading zeros mean "not
   for sale", not "nobody bought it". Left in, they teach a model the item sells
   nothing and drag every rolling average down. A missing price is the reliable
   signal: no price on file that week means it was not being sold.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pandas as pd

from src.step1_problem import Config

# Bump when load_m5 or trim_prelaunch_zeros changes what they produce. The
# cache key includes this, so old parquet files stop being served instead of
# silently returning a panel built by superseded logic.
CACHE_VERSION = 1

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAL_COLS = [
    "d",
    "date",
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]
SNAP_BY_STATE = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_m5(
    data_dir: str | Path = "data",
    item_ids: tuple[str, ...] = (),
    store_ids: tuple[str, ...] = (),
    dept_id: str | None = None,
    cat_id: str | None = None,
) -> pd.DataFrame:
    """
    Return a long-format daily panel: one row per (id, date).

    Every filter is optional and all are applied *before* the reshape. Passing
    nothing loads the whole panel, which is slow and large - always pass
    something in interactive work.
    """
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(data_dir / "sell_prices.csv")

    # ---- filter before melting ------------------------------------------
    if item_ids:
        sales = sales[sales["item_id"].isin(item_ids)]
    if store_ids:
        sales = sales[sales["store_id"].isin(store_ids)]
    if dept_id is not None:
        sales = sales[sales["dept_id"] == dept_id]
    if cat_id is not None:
        sales = sales[sales["cat_id"] == cat_id]
    if sales.empty:
        raise ValueError("No rows left after filtering - check the subset in Config.")

    day_cols = [c for c in sales.columns if c.startswith("d_")]

    # ---- wide -> long ----------------------------------------------------
    df = sales.melt(
        id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales"
    )

    # ---- calendar --------------------------------------------------------
    df = df.merge(calendar[CAL_COLS], on="d", how="left", validate="m:1")

    # Each row takes the SNAP flag for its own state. An unmapped state would
    # otherwise keep the 0 default - a plausible-looking flag that is simply
    # wrong - so refuse rather than guess.
    unmapped = set(df["state_id"].unique()) - set(SNAP_BY_STATE)
    if unmapped:
        raise ValueError(
            f"No SNAP column mapped for state(s) {sorted(unmapped)}. "
            f"Add them to SNAP_BY_STATE rather than defaulting to 0."
        )
    df["snap"] = 0
    for state, col in SNAP_BY_STATE.items():
        mask = df["state_id"] == state
        df.loc[mask, "snap"] = df.loc[mask, col]
    df = df.drop(columns=list(SNAP_BY_STATE.values()))

    # ---- prices (weekly, fanned out to daily) ----------------------------
    # validate="m:1" matters more than it looks. A duplicate (store, item, week)
    # in the price file would give a store-item two rows for one date, and
    # groupby().shift(7) shifts by seven *rows*, not seven days - so every lag
    # feature would silently point at the wrong date.
    df = df.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="m:1"
    )

    df["sales"] = df["sales"].astype("float32")
    return df.sort_values(["id", "date"]).reset_index(drop=True)


def trim_prelaunch_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop each series' rows from before the item was first offered for sale.

    Uses the first date on which a price exists; everything earlier is removed
    regardless of what the sales column says. How much this removes depends
    entirely on the subset, so `load_panel` reports the actual figure per run
    rather than quoting one here.

    Known limitation, worth stating in the write-up: this trims only the
    *leading* run of unpriced days. An item withdrawn and later relisted keeps
    its mid-history gap, which will read as genuine zero demand. (Verified not
    to occur in the current study set - all missing prices are leading runs.)
    """
    first_priced = (
        df[df["sell_price"].notna()]
        .groupby("id", observed=True)["date"]
        .min()
        .rename("launch_date")
    )

    # A series that was never priced has no launch date, and every date
    # comparison against NaT is False - so it would vanish without a trace.
    never_priced = sorted(set(df["id"].unique()) - set(first_priced.index))
    if never_priced:
        shown = never_priced[:5]
        more = " ..." if len(never_priced) > 5 else ""
        warnings.warn(
            f"{len(never_priced)} series have no price on record and are being "
            f"dropped entirely: {shown}{more}",
            stacklevel=2,
        )

    out = df.merge(first_priced, on="id", how="left")
    out = out[out["date"] >= out["launch_date"]]
    return out.drop(columns=["launch_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cached entry point - this is what notebooks call
# --------------------------------------------------------------------------- #


def _cache_key(cfg: Config) -> str:
    """Stable hash of the subset *and* the build logic version."""
    payload = repr((CACHE_VERSION, sorted(cfg.subset.items())))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def load_panel(cfg: Config, use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Load, join and trim the panel described by `cfg`, caching the result.

    The CSV melt takes tens of seconds; the parquet round-trip takes under one.
    Cached files live in `cfg.cache_dir`, which is gitignored. Bumping
    CACHE_VERSION or deleting that directory forces a rebuild.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.cache_dir / f"panel_v{CACHE_VERSION}_{_cache_key(cfg)}.parquet"

    if use_cache and path.exists():
        df = pd.read_parquet(path)
        if verbose:
            print(f"loaded cached panel: {path.name}  ({len(df):,} rows)")
        return df

    raw = load_m5(cfg.data_dir, **cfg.subset)
    df = trim_prelaunch_zeros(raw)

    if verbose:
        dropped = len(raw) - len(df)
        print(
            f"built panel: {df['id'].nunique():,} series, {len(df):,} rows, "
            f"{df['date'].min().date()} -> {df['date'].max().date()}\n"
            f"  pre-launch rows dropped : {dropped:,} ({dropped / len(raw):.1%})\n"
            f"  zero-sales share before : {(raw['sales'] == 0).mean():.1%}\n"
            f"  zero-sales share after  : {(df['sales'] == 0).mean():.1%}"
        )

    df.to_parquet(path, index=False)
    return df


if __name__ == "__main__":
    cfg = Config(item_ids=("FOODS_3_120", "FOODS_3_681", "FOODS_3_282"))
    print(cfg.describe(), "\n")
    df = load_panel(cfg, use_cache=False)
    print("\nseries per item/store:")
    print(df.groupby(["item_id", "store_id"], observed=True).size().unstack().to_string())
