"""
Step 1 - Problem definition (FPP §1.6, step 1).

Before any data is touched, FPP §1.3 says the forecasting task must be pinned
down explicitly: what is being forecast, at what grain, how far ahead, and how
success is judged. This module is that decision, written down once, in one
object, so that nothing downstream has to guess and nothing downstream can
quietly disagree.

Changing the horizon, the fold count, or the data subset should mean editing one
line here - never hunting through the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repo root, derived from this file's location. Relative paths in a Config are
# resolved against it, so the same Config works from the project root, from
# inside notebooks/, or from anywhere else - no "../data" bookkeeping.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The study set. One fast-moving staple as the spine, plus two low-volume items
# whose demand class genuinely varies by store. Chosen by the availability screen
# in step 3: every one of these is continuously stocked at all ten stores (longest
# zero run <= 11 days over ~1,885 days), so their demand classes reflect customer
# behaviour rather than stocking gaps.
STUDY_ITEMS = ("FOODS_3_586", "FOODS_3_232", "FOODS_3_156")

# Columns a model is allowed to pool over. Anything else is a typo.
POOL_SCOPES = (None, "item_id", "dept_id", "cat_id", "store_id", "state_id")

# Leading rows lost per series to feature warm-up: the longest lag is
# horizon + 28, and the 28-day rolling windows sit on top of already-shifted
# data. Used only to explain `min_train_days` - the real drop happens in the
# feature module.
WARMUP_DAYS = 63


@dataclass(frozen=True)
class Config:
    """
    The complete definition of one forecasting experiment.

    Frozen on purpose. A config that changes halfway through a notebook session
    is a config you cannot trust when reporting the result.

    Parameters
    ----------
    horizon
        How many days ahead we forecast. We produce a forecast for every day
        h = 1 .. horizon from a single forecast origin T, which in FPP notation
        is `ŷ_{T+h|T}`. Daily, not weekly: inventory is consumed day by day, and
        a weekly total destroys the weekday pattern that decides *when* a store
        runs out.
    test_window
        Days scored per fold. Defaults to `horizon` and may never exceed it -
        see `__post_init__`.
    n_folds
        Number of consecutive walk-forward origins. Each fold steps one
        `test_window` further back in time. More folds recover sample size
        without ever lengthening the test window.
    min_train_days
        Minimum *usable* training days, counted after feature warm-up has
        dropped the leading rows with incomplete lags (about WARMUP_DAYS rows
        per series). A series with fewer than this in a given fold is skipped
        for that fold. 365 keeps at least one full annual cycle, so in raw
        calendar terms a series needs roughly 365 + WARMUP_DAYS days.
    season
        Length of the dominant seasonal cycle, in days. 7 for daily retail.
    pool_by
        Which rows a model may learn from - the cross-learning scope.
        None       -> one model per store-item (each series trained alone)
        "item_id"  -> one model per item, pooled across its stores
        "dept_id"  -> one model per department, pooled across items and stores
        "cat_id" / "state_id" / "store_id" -> wider scopes, same machinery
        Widening this is the documented path from per-series training to full
        cross-learning; it is a parameter, not a rewrite.
    seed
        Fixed so that repeat runs on identical inputs give identical output.
    """

    # --- what we forecast -------------------------------------------------
    horizon: int = 7
    test_window: int | None = None
    season: int = 7

    # --- how we validate --------------------------------------------------
    n_folds: int = 8
    min_train_days: int = 365

    # --- which data -------------------------------------------------------
    item_ids: tuple[str, ...] = ()
    store_ids: tuple[str, ...] = ()
    dept_id: str | None = None
    cat_id: str | None = None

    # --- how we model -----------------------------------------------------
    pool_by: str | None = None
    seed: int = 0

    # --- where things live ------------------------------------------------
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")  # both resolved against PROJECT_ROOT

    def __post_init__(self) -> None:
        if self.test_window is None:
            object.__setattr__(self, "test_window", self.horizon)

        # --- the non-negotiable safeguard ---------------------------------
        # Features are lagged by the horizon. If the scored window were longer
        # than the horizon, the lags for its later days would point at dates
        # *inside* that window, and the model would be reading the actuals it
        # is supposed to be predicting. This produced a fabricated result once
        # already, so it is asserted rather than trusted to convention.
        if self.test_window > self.horizon:
            raise ValueError(
                f"test_window ({self.test_window}) must not exceed horizon "
                f"({self.horizon}). A longer scored window lets lag features "
                f"reach into the period being forecast. Recover sample size "
                f"with more folds (n_folds), never with a longer window."
            )
        if self.horizon < 1 or self.n_folds < 1:
            raise ValueError("horizon and n_folds must both be >= 1.")

        # Catch a mistyped pooling scope here rather than deep inside a model.
        if self.pool_by not in POOL_SCOPES:
            raise ValueError(
                f"pool_by={self.pool_by!r} is not a poolable column. "
                f"Use one of: {POOL_SCOPES}"
            )

        # Anchor relative paths to the repo root so the working directory
        # never changes what a Config means. Absolute paths pass through.
        for p in ("data_dir", "cache_dir"):
            value = Path(getattr(self, p))
            if not value.is_absolute():
                value = PROJECT_ROOT / value
            object.__setattr__(self, p, value)

    # ---------------------------------------------------------------------- #

    @property
    def subset(self) -> dict[str, object]:
        """The data filter, as a plain dict - used for cache keys and captions."""
        return {
            "item_ids": tuple(self.item_ids),
            "store_ids": tuple(self.store_ids),
            "dept_id": self.dept_id,
            "cat_id": self.cat_id,
        }

    @property
    def test_days_total(self) -> int:
        """Total days scored per series across all folds."""
        return self.n_folds * self.test_window

    def describe(self) -> str:
        """Human-readable summary, for the top of a notebook."""
        if self.pool_by is None:
            scope = "per store-item"
        else:
            scope = f"pooled by {self.pool_by}"

        filters = [f"{k}={v}" for k, v in self.subset.items() if v not in ((), None)]
        where = ", ".join(filters) if filters else "whole panel"

        return (
            f"Forecast daily unit sales, h=1..{self.horizon} from origin T.\n"
            f"  subset      : {where}\n"
            f"  validation  : {self.n_folds} walk-forward folds "
            f"x {self.test_window}-day window ({self.test_days_total} days scored)\n"
            f"  training    : {scope}, min {self.min_train_days} usable days\n"
            f"  seed        : {self.seed}"
        )


if __name__ == "__main__":
    print(Config(item_ids=("FOODS_3_120",), dept_id="FOODS_3").describe(), "\n")
    for bad in (dict(test_window=28), dict(pool_by="item")):
        try:
            Config(**bad)
        except ValueError as e:
            print(f"rejected {bad}:\n  {e}\n")
