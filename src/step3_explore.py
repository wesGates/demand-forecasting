"""
Step 3 - Preliminary exploratory analysis (FPP §1.6, step 3).

FPP is blunt about this: *always start by graphing the data*. Nothing here fits
a model. It answers the questions §1.6 says to ask before modelling - is there a
pattern, a trend, seasonality, outliers, relationships between variables - and
adds the one measurement this project is built around.

**Classifying demand.** "Sporadic and inconsistent" is a judgement until you
measure it. The standard measurement (Syntetos & Boylan) is two numbers:

  ADI  - average demand interval: days divided by days with a sale.
         "How *often* does it sell?"  High ADI = sporadic.
  CV²  - squared coefficient of variation of the non-zero sale sizes.
         "When it does sell, how *consistent* is the amount?"  High CV² =
         inconsistent.

Cut each at its conventional threshold and you get four quadrants:

                 CV² < 0.49          CV² >= 0.49
    ADI <  1.32   smooth              erratic
    ADI >= 1.32   intermittent        lumpy

A caution worth carrying into any discussion of this: 1.32 and 0.49 are
conventions derived from one comparison of forecasting methods, not laws of
nature. A series at ADI 1.31 is not meaningfully different from one at 1.33.
Treat the numbers as continuous and the quadrants as labels of convenience -
which is why `plot_demand_class_map` shows the cloud, not just the buckets.

**Availability comes before intermittency.** A long unbroken block of zero
sales is not sporadic demand - it usually means the item was not on the shelf.
M5 has no inventory data, so the two are indistinguishable from sales alone, and
the pre-launch price trim cannot catch them (these days *have* a price on file).
Left in, they inflate ADI exactly like genuine intermittency and the
classification ends up measuring stocking gaps instead of customer behaviour.

`max_zero_run` measures this so series can be *screened* rather than silently
repaired. That is deliberate: a screen states a selection criterion you can
defend, where a trim would need an arbitrary threshold baked into the pipeline
and would punch gaps into series that lag features would then reach across.

**These are computed on training data only.** The class label is part of the
reported result, so it must not be informed by any day the model is scored on.
`classification_cutoff` returns the first scored date across all folds, and
everything at or after it is excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.step1_problem import Config

# Syntetos-Boylan cut points. Conventions, not laws - see the module docstring.
ADI_CUT = 1.32
CV2_CUT = 0.49

DEMAND_CLASSES = ("smooth", "erratic", "intermittent", "lumpy")


def classification_cutoff(df: pd.DataFrame, cfg: Config) -> pd.Timestamp:
    """
    First date that will ever be scored, across every walk-forward fold.

    Folds walk backwards from the end of the data in `test_window` steps, so the
    earliest scored day sits `n_folds * test_window` days from the end. Anything
    on or after this date is off-limits for computing a class label.
    """
    return df["date"].max() - pd.Timedelta(days=cfg.test_days_total - 1)


def demand_class(adi: float, cv2: float) -> str:
    """Bucket one series from its ADI and CV². See the quadrant table above."""
    if not np.isfinite(adi) or not np.isfinite(cv2):
        return "unclassifiable"
    if adi < ADI_CUT:
        return "smooth" if cv2 < CV2_CUT else "erratic"
    return "intermittent" if cv2 < CV2_CUT else "lumpy"


def max_zero_run(sales: np.ndarray) -> int:
    """
    Longest unbroken stretch of zero-sales days.

    The availability screen. A fast-moving grocery item that genuinely sells
    every day should never post a two-week zero run; if it does, it was almost
    certainly unavailable rather than unwanted.
    """
    best = current = 0
    for is_zero in sales == 0:
        current = current + 1 if is_zero else 0
        best = max(best, current)
    return best


def _series_row(sales: np.ndarray) -> dict[str, float]:
    """ADI and CV² for one series' sales vector."""
    n = len(sales)
    nonzero = sales[sales > 0]

    # CV² needs at least two sales to have any spread to measure. A series with
    # zero or one sale in the window gets no label rather than a fabricated one.
    if len(nonzero) < 2:
        return {
            "adi": np.inf if len(nonzero) == 0 else n / len(nonzero),
            "cv2": np.nan,
            "max_zero_run": max_zero_run(sales),
        }

    adi = n / len(nonzero)
    cv2 = float((nonzero.std(ddof=1) / nonzero.mean()) ** 2)
    return {"adi": float(adi), "cv2": cv2, "max_zero_run": max_zero_run(sales)}


def series_stats(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    One row per series: size, sparsity, ADI, CV², and demand class.

    Computed strictly on data before `classification_cutoff`, so no scored day
    contributes to a label that will later appear in the results table.
    """
    cutoff = classification_cutoff(df, cfg)
    train = df[df["date"] < cutoff]

    rows = []
    for series_id, g in train.groupby("id", sort=False, observed=True):
        sales = g["sales"].to_numpy(dtype=float)
        stats = _series_row(sales)
        rows.append(
            {
                "id": series_id,
                "item_id": g["item_id"].iloc[0],
                "store_id": g["store_id"].iloc[0],
                "state_id": g["state_id"].iloc[0],
                "n_days": len(sales),
                "mean_sales": float(sales.mean()),
                "zero_rate": float((sales == 0).mean()),
                **stats,
            }
        )

    out = pd.DataFrame(rows)
    out["demand_class"] = [
        demand_class(a, c) for a, c in zip(out["adi"], out["cv2"], strict=True)
    ]
    return out.sort_values(["item_id", "mean_sales"], ascending=[True, False])


def class_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Item x store grid of demand classes - the study design at a glance."""
    return stats.pivot(index="item_id", columns="store_id", values="demand_class")


def summarise(stats: pd.DataFrame, cutoff: pd.Timestamp) -> str:
    """A short text summary, for the top of a notebook section."""
    counts = stats["demand_class"].value_counts()
    spread = (
        stats.groupby("item_id", observed=True)["demand_class"]
        .nunique()
        .rename("classes")
    )
    lines = [
        f"classification window ends {cutoff.date()} "
        f"({stats['n_days'].min():,}-{stats['n_days'].max():,} days per series)",
        f"series: {len(stats)}",
        f"longest zero run: {stats['max_zero_run'].min()}-{stats['max_zero_run'].max()} "
        f"days (availability screen)",
        "class counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
        "classes spanned per item: " + ", ".join(f"{k}={v}" for k, v in spread.items()),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from src.step1_problem import STUDY_ITEMS
    from src.step2_data import load_panel

    cfg = Config(item_ids=STUDY_ITEMS)
    df = load_panel(cfg, verbose=False)

    stats = series_stats(df, cfg)
    print(summarise(stats, classification_cutoff(df, cfg)), "\n")
    print(class_table(stats).to_string(), "\n")
    cols = [
        "item_id",
        "store_id",
        "mean_sales",
        "zero_rate",
        "max_zero_run",
        "adi",
        "cv2",
        "demand_class",
    ]
    print(stats[cols].round(3).to_string(index=False))
