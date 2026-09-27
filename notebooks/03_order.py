# %% [markdown]
# # Step 5, continued — From forecast to order
#
# `02_evaluate` scored every method as a point forecast with a symmetric
# error. This notebook takes the forecast the rest of the way, to the weekly
# quantity a store would order at a chosen service level, and scores that
# with the metric the decision cares about (FPP §5.5, §5.9).
#
# This is the ordering prototype from before item 1. Item 1 found that
# calibrated error quantiles match an XGBoost trained on the quantile
# objective, at a twenty-sixth of the cost, so this calibrated route is the
# one the quantile work continues from. Like the other notebooks it
# is the procedure. It runs unchanged for any item in `STUDY_ITEMS` and
# prints its findings, which go in a dated write-up.
#
# It needs a two-year walk-forward. The first year calibrates each method's
# error quantiles and the second is judged. The run is cached, so after the
# first time this loads in seconds.
#
# Run a cell with Shift+Enter.

# %%
import sys
from pathlib import Path

# Put the repo root on sys.path so `from src import ...` resolves however this
# file is run, cell-by-cell in the Interactive Window or as a plain script.
if not any((Path(p) / "src").is_dir() for p in sys.path):
    sys.path.insert(
        0,
        str(
            next(
                d
                for start in [
                    Path(globals().get("__file__", ".")).resolve().parent,
                    Path.cwd(),
                ]
                for d in (start, *start.parents)
                if (d / "src").is_dir()
            )
        ),
    )

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # noqa: F821
    get_ipython().run_line_magic("autoreload", "2")  # noqa: F821
except NameError:
    pass

import matplotlib.pyplot as plt
import pandas as pd

from src import plots
from src.order import (
    calibrate,
    calibrate_expanding,
    quantile_forecasts,
    ranking_by_tau,
    score_quantiles,
    summarise_quantiles,
    weekly_totals,
)
from src.step1_problem import STUDY_ITEMS, Config
from src.step2_data import load_panel
from src.step3_explore import series_stats
from src.step5_evaluate import run_walk_forward

plots.use_style()
ITEM = STUDY_ITEMS[0]

# %%
# Where figures appear - see 01_explore for the two settings.
FIGURE_BACKEND = "tk"
try:
    try:
        get_ipython().run_line_magic("matplotlib", FIGURE_BACKEND)  # noqa: F821
    except Exception:
        get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
except NameError:
    pass  # not running under IPython - leave the backend alone

# %% [markdown]
# ## The problem definition, and the data
#
# The same `Config` as the other notebooks defines the scored year. The
# calibration year is the 52 folds before it, so the run below is asked for
# twice the folds.

# %%
cfg = Config(item_ids=STUDY_ITEMS)
df = load_panel(cfg)
stats = series_stats(df, cfg)
top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]

# %% [markdown]
# ## From forecast to order (FPP §5.5 and §5.9)
#
# Everything above scores a point forecast with a symmetric error. A
# replenishment decision needs two things it does not have.
#
# The order is a weekly total. A Sunday order covers Monday to Sunday, so the
# error that reaches the shelf is the week's forecast total minus the week's
# actual total. Daily errors partly cancel inside a week and the daily RMSSE
# never sees it.
#
# Ordering the mean runs out about half the time. A point forecast is the
# centre of what might happen. The quantity to order is a quantile, the level
# that covers demand with a chosen probability τ, and which τ is an economic
# question. τ = Cu / (Cu + Co), the understock cost over the sum of both.
# Above 0.5 when running out is the expensive mistake (ambient grocery),
# below 0.5 when holding is (fresh produce that goes in the bin). This item
# is anonymised, so its τ is unknown and a range is reported.
#
# Where the quantiles come from. Each method's own weekly errors over a
# calibration year, the year before the scored one, give it an error
# distribution. Its τ-quantile forecast is the point forecast plus the
# τ-quantile of those errors. Every method is treated the same way, so a
# benchmark and a model compete on the same footing, and a biased method is
# corrected for free. Calibrating on the same weeks that are then judged
# would give 90% coverage by construction, which is why the run below has
# two years of folds.
#
# Pinball loss (the quantile score, FPP §5.9) is the metric. A unit of
# shortfall costs 2τ, a unit of surplus 2(1 − τ). At τ = 0.9 running out is
# nine times as expensive as over-ordering. Losses at different τ are on
# different scales, so read across methods within a column and never along
# a row.
#
# What to look for is whether the ranking changes with τ. A method that is
# best at 0.9 and not at 0.3 is the right model for an ambient item and the
# wrong one for a perishable, and RMSSE cannot tell you which is which. Then
# coverage. A calibrated method's 0.9 order covers about 90% of weeks, and
# the gap is how well one year's errors described the next.

# %%
both = Config(**{**cfg.__dict__, "n_folds": 2 * cfg.n_folds})
scored_from = cfg.holdout_start(df["date"].max()) - pd.Timedelta(days=1)
weekly = weekly_totals(run_walk_forward(df, both))
print(
    f"{len(weekly):,} series-fold-method weeks; calibrate before {scored_from.date()}, score after"
)

# %% [markdown]
# Weekly-total bias, forecast minus actual in units per week, its spread,
# and the share of weeks each method under-forecast. The last column is the
# "order the mean and run out half the time" claim, measured.

# %%
_w = weekly[weekly["origin"] >= scored_from].assign(under=lambda w: w["error"] > 0)
(
    _w.groupby("method", observed=True)
    .agg(
        bias=("error", lambda e: -e.mean()), sd=("error", "std"), under=("under", "mean")
    )
    .sort_values("sd")
    .round(2)
)

# %% [markdown]
# Relative pinball loss by method and τ, calibrated on the first year and
# scored on the second. Best at τ = 0.5 first. Then the same on holiday weeks
# only, and with a calibration window that grows as the scored year proceeds,
# which is what a live system would do.

# %%
offsets = calibrate(weekly, scored_from)
scored_q = score_quantiles(quantile_forecasts(weekly, offsets, scored_from))
summary = summarise_quantiles(scored_q)
ranking_by_tau(summary).round(3)

# %%
summary.pivot_table(index="method", columns="tau", values="coverage").round(2)

# %%
ranking_by_tau(summarise_quantiles(scored_q, "holiday")).round(3)

# %%
expanding = calibrate_expanding(weekly, scored_from)
ranking_by_tau(
    summarise_quantiles(
        score_quantiles(quantile_forecasts(weekly, expanding, scored_from))
    )
).round(3)

# %% [markdown]
# The picture. Left, quantile score against service level, where lines that
# cross mean the ranking depends on the cost asymmetry. Right, achieved
# coverage against target, where the diagonal is perfect calibration. Below,
# what the order would have been each week at the busiest store with the
# actual over it. Weeks where black rises above the dashed line are the
# stockouts a 90% service level accepts.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
plots.plot_pinball_by_tau(summary, ax=axes[0])
plots.plot_coverage_by_tau(summary, ax=axes[1])
plots.merge_legends(fig)
plots.show()

# %%
fig, ax = plt.subplots(figsize=(12.5, 3.8))
plots.plot_weekly_order_band(scored_q, top_id, "arima", ax=ax)
plots.show()

# %% [markdown]
# ## What this step should have established
#
# The answers go in a dated write-up, alongside the others:
#
# 1. How often ordering the mean would have run out. The share of weeks
#    under-forecast, per method.
# 2. Whether the model ranking survives the move from a symmetric error to
#    an asymmetric cost. Relative pinball by τ, and where the lines cross.
# 3. Whether the quantiles are trustworthy. Coverage against target, and how
#    much it slips on holiday weeks.
# 4. What the order would have been. The weekly band at the busiest store,
#    and which weeks a given service level would have run short.
