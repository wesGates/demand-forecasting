# %% [markdown]
# # Step 5 — Using and evaluating the forecasts
#
# FPP §1.6's last step. Every method from step 4 is run through every
# walk-forward fold, and what it forecast is set against what actually sold.
#
# Like `01_explore`, this notebook is the procedure. It runs unchanged for
# any item in `STUDY_ITEMS` and prints its findings. What those findings mean
# for a particular product goes in a separate dated write-up.
#
# Two things to hold onto while reading:
#
# - RMSSE (FPP §5.8) is the forecast's error divided by the error a seasonal
#   naive forecast would have made on the training data. 1.0 means no better
#   than repeating last week's same weekday, 0.8 means 20% better. It is
#   scale-free, so a 100-unit store and a 15-unit store are comparable.
# - Every method in a series-fold shares the same denominator, so RMSSE never
#   changes which method wins a fold. It changes how stores and folds compare
#   to each other.
#
# Run a cell with Shift+Enter.

# %%
import sys
from pathlib import Path

# Put the repo root on sys.path so `from src import ...` resolves however this
# file is run, cell-by-cell in the Interactive Window or as a plain script from
# any directory. Changed from `sys.path.append("..")` to avoid the
# ModuleNotFoundError that `python notebooks/02_evaluate.py` raised from the repo root.
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

from src import plots
from src.step1_problem import STUDY_ITEMS, Config
from src.step2_data import load_panel
from src.step3_explore import classification_cutoff, series_stats
from src.step4_models import MODELS
from src.step5_evaluate import (
    rmsse_by_store,
    run_walk_forward,
    score_folds,
    summarise,
    win_rates,
)

plots.use_style()
ITEM = STUDY_ITEMS[0]

# %%
# Where figures appear. "tk" pops every figure out into its own resizable
# window (zoom, pan, hover labels) AND keeps an inline copy in the Interactive
# Window, because each plotting cell ends with `plots.show()`. "inline" keeps
# only the inline copy. Change it and re-run this cell; it only affects
# figures made after it runs. On a machine with no display, "tk" falls back
# to "inline" by itself.
FIGURE_BACKEND = "tk"
try:
    try:
        get_ipython().run_line_magic("matplotlib", FIGURE_BACKEND)  # noqa: F821
    except Exception:
        get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
except NameError:
    pass  # not running under IPython - leave the backend alone

# %% [markdown]
# ## The problem definition
#
# The same object step 3 used. Note the two scoring fields (the RMSSE
# scaling lag and window) and the fold layout. Changing `n_folds` here
# changes how many weeks get scored and moves the classification cutoff in
# step 3 earlier, since both read from `Config.holdout_start`.

# %%
cfg = Config(item_ids=STUDY_ITEMS)
print(cfg.describe())

# %% [markdown]
# ## Run the walk-forward
#
# For every store and every fold, cut the history at the origin, hand each
# method only what it is allowed to see, collect its forecast, and record the
# actual. The result is one long table, one row per store, fold, method and
# day ahead, that every number and plot below is computed from. There is one
# place where a forecast meets an actual.
#
# The result is cached to parquet under a key that includes the config and
# the source of the model code, so editing a model invalidates the cache by
# itself. `run_walk_forward(..., use_cache=False)` forces a rerun. A full run
# takes a few minutes with the parallel harness, ARIMA being the slow one.
# The validator (`python -m src.validate`) should be green before any of
# these numbers are quoted.
#
# Two things happen to the scored days before any number is computed. The
# closure day the loader flagged (Christmas) is dropped, because forecasting a
# shut store is not a demand question. And every fold is tagged `normal` or
# `holiday`. A fold is a holiday fold if any scored day falls from two days
# before a major event to one day after it.

# %%
df = load_panel(cfg)
stats = series_stats(df, cfg)
cutoff = classification_cutoff(df, cfg)
origins = cfg.fold_origins(df["date"].max())
top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]

predictions = run_walk_forward(df, cfg)
scores = score_folds(predictions)
# Printed text stays ASCII. A Windows console that is not UTF-8 chokes on an
# arrow or a middle dot.
print(f"{len(predictions):,} forecast-days | {len(scores):,} series-fold-method scores")
print(
    f"scored window: {predictions['target_date'].min().date()} -> "
    f"{predictions['target_date'].max().date()}"
)

# %% [markdown]
# ## The headline table
#
# One row per method, best first. FPP §5.8 lays its accuracy tables out this
# way, methods as rows and measures as columns.
#
# What to look for:
#
# - How far the models sit below the benchmarks, and whether the three
#   models are meaningfully apart from each other. Medians within a few
#   hundredths are a tie. Say so.
# - Bias. Positive means over-forecasting. For replenishment this matters in
#   its own right. A consistent under-forecast turns into stockouts.
# - Whether `naive` and `drift` sit near or above 1.0. With a seasonal-naive
#   denominator, "repeat yesterday" should score worse than 1.0 on a series
#   with a weekly cycle.
# - The two week columns. `rmsse_normal` and `rmsse_holiday` are the same
#   score on the ordinary folds and on the holiday folds. Quote both. A pooled
#   number cannot say whether a method's advantage comes from the ordinary
#   weeks or from the handful where the calendar does the work. ETS is the
#   one model that cannot be told a holiday is coming, and the holiday column
#   is where that shows.
# - `n_unscored` and `n_fallback`. Folds with no score, and folds where a
#   method gave up and returned the 28-day mean. Both should be zero here.

# %%
summarise(scores).round(3)

# %% [markdown]
# ## By store
#
# Stores down (busiest first), methods across (best first). This is the
# study's question. Does which method wins depend on the store?
#
# Read down the volume gradient. If the models' advantage over
# `moving_average_28` narrows or reverses at the quietest stores, that is the
# pattern to report. Then compare the holiday-week table, since the store
# ranking can change when the calendar is doing the work.

# %%
rmsse_by_store(scores).round(3)

# %%
rmsse_by_store(scores, "normal").round(3)

# %%
rmsse_by_store(scores, "holiday").round(3)

# %% [markdown]
# ## Win rates
#
# Rows beat columns. The share of series-folds where the row's RMSSE was
# lower than the column's. 0.5 is "no better than". A blunter instrument than
# RMSSE, it says how often and not by how much, which is why it sits beside
# the table.
#
# Two pairs to look at. `moving_average_28` against `seasonal_naive`: a
# mean-based method beating a single-past-day method more often than not is
# the √2 effect from step 4's notes, and it is why this project reports six
# benchmarks. And `seasonal_naive_364` ("this day last year", what an orderer
# checks before a holiday) against `seasonal_naive` ("this day last week"):
# on an item whose level drifts, last year's lookup carries the old level
# with it, and the holiday-week table says whether it helps at all.

# %%
win_rates(scores).round(2)

# %%
win_rates(scores, "holiday").round(2)

# %% [markdown]
# ## Forecasts against actuals (FPP §5.8)
#
# The book's verdict on its version of this figure is "it is obvious from the
# graph". If a method is better, this should show it before any table does.
# Drawn for the busiest store. Change `top_id` to look at another.
#
# Black is what sold. Coloured lines are the three models, grey the two
# strongest benchmarks. On the weekly layout the faint vertical lines are
# fold origins, and a kink there is the origin moving. On the every-day
# layout the one-step forecast is drawn. The heavier vertical line is where
# the held-out window begins. Everything left of it is lead-in context.
#
# Does the model follow the weekly shape (weekend peaks) or just the level?
# Does it lag behind level shifts? Are its misses one-sided?

# %%
fig, ax = plt.subplots(figsize=(12.5, 3.6))
plots.plot_forecast_folds(
    predictions, df, top_id, holdout_start=cutoff, origins=origins, ax=ax
)
plots.show()

# %% [markdown]
# ## Error by horizon (FPP §5.10)
#
# The book's figure 5.24. Error as a function of how many days ahead the
# forecast was made. A day-ahead forecast knows more than a week-ahead one,
# so the lines should rise with h.
#
# Read how steeply each line rises. A flat line means the method is not
# using the recent past at all, which is how a long-run mean behaves. A steep
# line means the method's edge is concentrated at short horizons. Where the
# model lines cross the benchmark lines, if they do, is the horizon beyond
# which the model stops earning its complexity.
#
# On the weekly layout the origins step by exactly 7 days, so every origin is
# the same weekday and h = 7 is always Sunday, the busiest day. The plot then
# mixes "how far ahead" with "which weekday". The every-day layout removes
# this (item 2) and is the reported one.

# %%
fig, ax = plt.subplots(figsize=(7, 3.4))
plots.plot_rmsse_by_horizon(predictions, ax=ax)
plots.show()

# %% [markdown]
# ## RMSSE by store, the study's own view
#
# Not one of FPP's figures. It is the picture of the by-store table above.
# Busiest store on the left. The line at 1.0 is "as good as seasonal naive on
# the training data".
#
# Watch whether the coloured markers pull away from the grey pack at the busy
# stores and sink back into it at the quiet ones.

# %%
fig, ax = plt.subplots(figsize=(10, 3.8))
plots.plot_rmsse_by_store(scores, stats, item_id=ITEM, ax=ax)
plots.show()

# %% [markdown]
# ## Residual diagnostics (FPP §5.4)
#
# The book says a good method's residuals are uncorrelated and centred on
# zero, and ideally of constant variance and roughly normal. One panel per
# question, for each model at the busiest store.
#
# - Time plot. Is the mean near zero (printed in the title), and is the
#   spread steady across the window?
# - ACF. Is anything left that the method should have caught? Bars outside
#   the grey band are structure the forecast missed. The Ljung-Box p-value in
#   the title tests this. Above 0.05 means the residuals look like white
#   noise.
# - Histogram. Roughly symmetric and bell-shaped, or skewed and heavy-tailed?
#
# FPP's remedy for a non-zero mean is worth knowing: "if the residuals have
# mean m, simply add m to all forecasts". Correlation is harder, and the book
# defers it to Chapter 10.
#
# On the every-day layout the one-step residuals are used, which is what FPP
# states the test for, so the verdicts are real ones. On the weekly layout
# these are 1- to 7-step-ahead errors from a shared origin, and correlation
# out to lag 6 is expected even for a perfect model. There, a spike at lag 7
# or beyond, or a mean far from zero, is what would count against a method.
#
# The loop reads the model registry, so a model added in step 4 appears here
# without editing the notebook.

# %%
for model in MODELS:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    plots.plot_residual_diagnostics(
        predictions, top_id, model, season=cfg.season, axes=axes
    )
    fig.suptitle(f"{model} at {top_store} — residual diagnostics", fontsize=11)
plots.show()

# %% [markdown]
# ## What this step should have established
#
# The answers go in a dated write-up, alongside step 3's:
#
# 1. Which methods beat the benchmarks, and by how much. RMSSE means and
#    medians, and whether the models are separable from each other.
# 2. Whether the ranking depends on the store (the by-store table and figure,
#    read along the volume gradient) and on the kind of week. A method that
#    wins the ordinary weeks and loses the holiday weeks is a different
#    finding from one that wins both.
# 3. How error grows with horizon, and whether the models' edge is at short
#    lead times only.
# 4. Whether the residuals are clean. Zero mean, no leftover structure. A
#    non-zero mean is a bias worth correcting, leftover ACF structure is a
#    feature the model is missing.
# 5. Where the models lose, named. The stores or folds where a benchmark wins
#    are the honest edge of what the method can do.
#
# The forecast is not yet an order. `03_order` takes it the rest of the way.
