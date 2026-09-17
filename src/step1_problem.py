"""
Step 1 - Problem definition (FPP §1.6, step 1).

Before any data is touched, FPP §1.3 says the forecasting task must be pinned
down explicitly: what is being forecast, at what grain, how far ahead, and how
success is judged. This module is that decision, written down once, in one
object, so that nothing downstream has to guess and nothing downstream can
quietly disagree.

Changing the horizon, the fold count, or the data subset should mean editing one
line here - never hunting through the codebase. Every field below carries a
comment saying what it does and, where the book has an opinion, which FPP
section that opinion lives in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Repo root, derived from this file's location. Relative paths in a Config are
# resolved against it, so the same Config works from the project root, from
# inside notebooks/, or from anywhere else - no "../data" bookkeeping.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The study set: one fast-moving staple across all ten stores. Chosen by the
# availability screen in step 3 - continuously stocked everywhere (longest zero
# run 10 days over 1,885 days), so its statistics reflect customer behaviour
# rather than stocking gaps. Volumes run 15-101 units/day across the ten stores,
# which is the gradient the model comparison is run against.
STUDY_ITEMS = ("FOODS_3_586",)

# Columns a model is allowed to pool over. Anything else is a typo.
POOL_SCOPES = (None, "item_id", "dept_id", "cat_id", "store_id", "state_id")

# How the RMSSE denominator is computed across the walk-forward folds.
#   "pre_holdout" - one denominator per series, from the whole training period
#                   before the first scored day. Every fold and every method
#                   share it, so fold-to-fold numbers are comparable.
#   "per_fold"    - recomputed from each fold's own training history. Closer to
#                   a literal reading of a single train/test split, but the
#                   scale drifts between folds.
RMSSE_SCALE_WINDOWS = ("pre_holdout", "per_fold")


@dataclass(frozen=True)
class Config:
    """
    The complete definition of one forecasting experiment.

    Frozen on purpose. A config that changes halfway through a notebook session
    is a config you cannot trust when reporting the result. Field-by-field
    explanations are in the comments beside each field.
    """

    # ======================================================================
    # What we forecast
    # ======================================================================

    # How many days ahead we forecast. One forecast for each of h = 1..horizon
    # from a single origin T - FPP writes this ŷ_{T+h|T} (§1.7). Daily rather
    # than weekly because inventory is consumed day by day, and a weekly total
    # destroys the weekday pattern that decides *when* a store runs out.
    horizon: int = 7

    # Days scored per fold. Defaults to `horizon` and may never exceed it - see
    # the safeguard in __post_init__. Not from FPP; from a bug this project has
    # already had.
    test_window: int | None = None

    # Length of the dominant seasonal cycle, in days. 7 for daily retail. Used
    # by the seasonal naive benchmark (FPP §5.2), by ETS as its seasonal period
    # (§8.3), and as the default RMSSE scaling lag (§5.8).
    season: int = 7

    # ======================================================================
    # How we validate
    # ======================================================================

    # Number of consecutive walk-forward folds - FPP calls this "time series
    # cross-validation" with a rolling forecasting origin (§5.10). Each fold
    # steps one `test_window` further back. More folds recover sample size
    # without ever lengthening the scored window.
    n_folds: int = 8

    # A series must have at least this many days of history at a fold's origin
    # or that fold is skipped for it. 365 keeps one full annual cycle in view.
    # Project decision, not FPP. (The feature module separately needs
    # `features.MIN_HISTORY` days to fill every lag; that is a much smaller
    # number and is handled there.)
    min_train_days: int = 365

    # ======================================================================
    # How we score
    # ======================================================================

    # RMSSE divides forecast error by the error of a naive forecast on the
    # training data, so it is scale-free and comparable across stores (FPP
    # §5.8). This is the lag of that naive forecast. FPP says: lag 1 for
    # non-seasonal data, lag m (= `season`) for seasonal data - and this data is
    # weekly-seasonal, so the default follows the book. None means "use
    # `season`". The M5 competition used lag 1; set this to 1 for tables that
    # compare directly against published M5 numbers.
    rmsse_scale_lag: int | None = None

    # Which training data the RMSSE denominator is computed from, across folds.
    # FPP §5.8 says "the training set" but its cross-validation section (§5.10)
    # does not address rolling origins, so this is a project decision. See
    # RMSSE_SCALE_WINDOWS above for the two options and why "pre_holdout" is
    # the default.
    rmsse_scale_window: str = "pre_holdout"

    # ======================================================================
    # Which data
    # ======================================================================

    # Filters applied before the wide-to-long reshape. Empty tuple / None means
    # "no filter on this column". Always pass something in interactive work -
    # the unfiltered panel is ~59 million rows.
    item_ids: tuple[str, ...] = ()
    store_ids: tuple[str, ...] = ()
    dept_id: str | None = None
    cat_id: str | None = None

    # ======================================================================
    # How we model
    # ======================================================================

    # Which rows a learned model may train on - the cross-learning scope.
    #   None       -> one model per store-item, trained on that series alone
    #   "item_id"  -> one model per item, trained on all of its stores at once
    # Wider scopes are accepted and use the same machinery: the model sees every
    # series in the pool and is handed `store_id` and `item_id` as categorical
    # features so it can tell them apart. Not an FPP concept - it comes from the
    # M5 competition, where every winning method pooled. Only None and
    # "item_id" have been exercised so far.
    pool_by: str | None = None

    # Whether `sell_price` is offered to the model. Price is known a week ahead
    # in this dataset, so it is a legitimate predictor in FPP's sense (§7 -
    # regression with predictors known in advance). Off by default because for
    # the current study item it takes three values in five years, on the same
    # two dates at every store, and is constant inside any forecast window - a
    # clock, not a price. An item whose price actually moves may want it on.
    use_price: bool = False

    # Fixed so that repeat runs on identical inputs give identical output. The
    # validator checks this.
    seed: int = 0

    # ======================================================================
    # Where things live (resolved against PROJECT_ROOT)
    # ======================================================================

    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")

    def __post_init__(self) -> None:
        if self.test_window is None:
            object.__setattr__(self, "test_window", self.horizon)
        if self.rmsse_scale_lag is None:
            object.__setattr__(self, "rmsse_scale_lag", self.season)

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
        if self.rmsse_scale_lag < 1:
            raise ValueError("rmsse_scale_lag must be >= 1.")

        # Catch a mistyped option here rather than deep inside a model.
        if self.pool_by not in POOL_SCOPES:
            raise ValueError(
                f"pool_by={self.pool_by!r} is not a poolable column. "
                f"Use one of: {POOL_SCOPES}"
            )
        if self.rmsse_scale_window not in RMSSE_SCALE_WINDOWS:
            raise ValueError(
                f"rmsse_scale_window={self.rmsse_scale_window!r} is not an option. "
                f"Use one of: {RMSSE_SCALE_WINDOWS}"
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

    def holdout_start(self, last_date):
        """
        The first date that will ever be scored, given this fold layout.

        Walk-forward folds step backwards from the end of the data in
        `test_window` chunks, so the earliest scored day sits
        `n_folds * test_window` days from the end.

        **This is the single source of truth for that boundary.** Step 3 uses it
        to make sure no class label is computed from a scored day, and step 5
        uses it to lay out the folds. If the two ever computed it separately
        they could drift apart, and a label would silently absorb test data.
        """
        return last_date - pd.Timedelta(days=self.test_days_total - 1)

    def fold_origins(self, last_date) -> list:
        """
        The forecast origin for every walk-forward fold, earliest first.

        Fold k scores the `test_window` days beginning `k` windows after
        `holdout_start`, and its origin is the day before the first of those.
        Together the folds tile the held-out period exactly - no gaps, no
        overlap, and the last fold ends on the final day of data.

        Shares `holdout_start` with step 3's classification cutoff, so the two
        cannot drift apart.
        """
        start = self.holdout_start(last_date)
        return [
            start + pd.Timedelta(days=k * self.test_window - 1)
            for k in range(self.n_folds)
        ]

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
            f"  training    : {scope}, min {self.min_train_days} days, "
            f"price {'on' if self.use_price else 'off'}\n"
            f"  scoring     : RMSSE scaled by lag-{self.rmsse_scale_lag} naive error, "
            f"{self.rmsse_scale_window} window\n"
            f"  seed        : {self.seed}"
        )


if __name__ == "__main__":
    print(Config(item_ids=STUDY_ITEMS).describe(), "\n")
    for bad in (
        dict(test_window=28),
        dict(pool_by="item"),
        dict(rmsse_scale_window="all"),
    ):
        try:
            Config(**bad)
        except ValueError as e:
            print(f"rejected {bad}:\n  {e}\n")
