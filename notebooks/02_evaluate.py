# %% [markdown]
# # Step 5 — Using and evaluating the forecasts
#
# FPP §1.6's last step. Every method from step 4 is run through every
# walk-forward fold, and what it forecast is set against what actually sold.
#
# Like `01_explore`, this notebook is the *procedure* — it runs unchanged for
# any item in `STUDY_ITEMS`, and prints its findings rather than stating them.
# What those findings mean for a particular product goes in a dated write-up
# under `findings/`.
#
# Two things to hold onto while reading:
#
# - **RMSSE** (FPP §5.8) is the forecast's error divided by the error a naive
#   forecast would have made on the training data. 1.0 means "no better than
#   repeating last week's same weekday"; 0.8 means 20% better. It is
#   scale-free, so a 100-unit store and a 15-unit store are comparable.
# - **Every method in a series-fold shares the same denominator**, so RMSSE
#   never changes *which* method wins a fold. It changes how stores and folds
#   are compared to each other.
#
# Run a cell with **Shift+Enter**.

# %%
import sys
from pathlib import Path

# Put the repo root on sys.path so `from src import ...` resolves however this
# file is run: cell-by-cell in the Interactive Window, which starts in
# notebooks/, or as a plain script from any directory at all. This was
# `sys.path.append("..")`, which assumed the working directory was always
# notebooks/ - true for the Interactive Window, but not for
# `python notebooks/02_evaluate.py`, which raised ModuleNotFoundError instead.
# Searching upward for the directory that actually holds src/ works from either.
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
try:
    get_ipython().run_line_magic("matplotlib", "tk")  # noqa: F821  ("inline" to revert)
except NameError:
    pass

# %% [markdown]
# ## The problem definition
#
# The same object step 3 used. Note the two scoring fields — the RMSSE scaling
# lag and window — and the fold layout. Changing `n_folds` here changes how
# many weeks get scored *and* moves the classification cutoff in step 3
# earlier, since both read from `Config.holdout_start`.

# %%
cfg = Config(item_ids=STUDY_ITEMS)
print(cfg.describe())

# %% [markdown]
# ## Run the walk-forward
#
# For every store and every fold: cut the history at the origin, hand each
# method only what it is allowed to see, collect its forecast, and record the
# actual. The result is one long table — one row per store, fold, method, and
# day ahead — that every number and plot below is computed from. There is
# exactly one place where a forecast meets an actual.
#
# The first run builds each store's feature matrix (~8s per store) and caches
# it; after that a full run takes under a minute. The validator
# (`python -m src.validate`) should be green before any of these numbers are
# quoted.

# %%
df = load_panel(cfg)
stats = series_stats(df, cfg)
cutoff = classification_cutoff(df, cfg)
origins = cfg.fold_origins(df["date"].max())
top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]

predictions = run_walk_forward(df, cfg)
scores = score_folds(predictions)
# Printed text stays ASCII: a Windows console that is not UTF-8 will choke on
# an arrow or a middle dot, and a notebook that only runs in one terminal is
# not a notebook that runs.
print(f"{len(predictions):,} forecast-days | {len(scores):,} series-fold-method scores")
print(
    f"scored window: {predictions['target_date'].min().date()} -> "
    f"{predictions['target_date'].max().date()}"
)

# %% [markdown]
# ## The headline table
#
# One row per method, best first. FPP §5.8 lays its accuracy tables out this
# way — methods as rows, measures as columns.
#
# **What to look for:**
#
# - How far the models sit below the benchmarks, and **whether the two models
#   are meaningfully apart from each other.** Medians within a few hundredths
#   are a tie; say so rather than crowning one.
# - **Bias.** Positive means over-forecasting. For replenishment this matters
#   in its own right — a consistent under-forecast turns into stockouts.
# - Whether `naive` and `drift` sit near or above 1.0. With a seasonal-naive
#   denominator, "repeat yesterday" *should* score worse than 1.0 on a series
#   with a weekly cycle.

# %%
summarise(scores).round(3)

# %% [markdown]
# ## By store
#
# Stores down (busiest first), methods across (best first). This is the
# study's actual question: **does which-method-wins depend on the store?**
#
# **What to look for:** read down the volume gradient. If the models' advantage
# over `moving_average_28` narrows — or reverses — at the quietest stores, that
# is the pattern to report. It says the models are earning their keep where
# there is signal to learn, and not where there isn't.

# %%
rmsse_by_store(scores).round(3)

# %% [markdown]
# ## Win rates
#
# Rows beat columns: the share of series-folds where the row's RMSSE was lower
# than the column's. 0.5 is "no better than". This is a blunter instrument than
# RMSSE — it says how *often*, not by how *much* — which is why it sits beside
# the table rather than replacing it.
#
# **What to look for:** `moving_average_28` against `seasonal_naive`. A mean-based
# method beating a single-past-day method more often than not is the √2 effect
# from step 4's notes, and it is why this project reports five benchmarks
# instead of one.

# %%
win_rates(scores).round(2)

# %% [markdown]
# ## Forecasts against actuals — FPP §5.8
#
# The book's own verdict on its version of this figure is *"it is obvious from
# the graph."* That is the bar: if a method is better, this should show it
# before any table does. Drawn for the busiest store; change `top_id` to look
# at another.
#
# **Reading it:** black is what sold. Coloured lines are the two models, grey
# the two strongest benchmarks. The faint vertical lines are fold origins —
# every seventh day the forecaster was re-run from a fresh standing point, so a
# kink there is the origin moving, not a property of the method. The heavier
# vertical line is where the held-out window begins; everything left of it is
# lead-in context only.
#
# **What to look for:** does the model follow the *weekly shape* (weekend
# peaks) or just the level? Does it lag behind level shifts? Are its misses
# one-sided?

# %%
fig, ax = plt.subplots(figsize=(12.5, 3.6))
plots.plot_forecast_folds(
    predictions, df, top_id, holdout_start=cutoff, origins=origins, ax=ax
)
fig.tight_layout()
fig

# %% [markdown]
# ## Error by horizon — FPP §5.10
#
# The book's figure 5.24: error as a function of how many days ahead the
# forecast was made. A day-ahead forecast knows more than a week-ahead one, so
# the lines should rise with h.
#
# **What to look for:** *how steeply* each line rises. A flat line means the
# method is not using the recent past at all — a long-run mean behaves this
# way. A steep line means the method's edge is concentrated at short horizons.
# Where the model lines cross the benchmark lines, if they do, is the horizon
# beyond which the model stops earning its complexity.
#
# **Caveat.** Our fold origins step by exactly 7 days, so every origin is the
# same weekday — h = 1 is always Monday, h = 7 always Sunday. This plot
# therefore mixes "how far ahead" with "which weekday", and Sunday is the
# busiest, noisiest day. If the curves are not a clean monotone rise, that is
# why. `OPEN_QUESTIONS.md` has the options for separating the two.

# %%
fig, ax = plt.subplots(figsize=(7, 3.4))
plots.plot_rmsse_by_horizon(predictions, ax=ax)
fig.tight_layout()
fig

# %% [markdown]
# ## RMSSE by store — the study's own view
#
# Not one of FPP's figures; it is the picture of the by-store table above.
# Busiest store on the left. The line at 1.0 is "as good as seasonal naive on
# the training data".
#
# **What to look for:** whether the coloured markers pull away from the grey
# pack at the busy stores and sink back into it at the quiet ones.

# %%
fig, ax = plt.subplots(figsize=(10, 3.8))
plots.plot_rmsse_by_store(scores, stats, item_id=ITEM, ax=ax)
fig.tight_layout()
fig

# %% [markdown]
# ## Residual diagnostics — FPP §5.4
#
# The book says a good method's residuals should be, essentially,
# **uncorrelated** and **centred on zero**; and usefully, of **constant
# variance** and roughly **normal**. One panel per question, for each model at
# the busiest store.
#
# **Reading it:**
#
# - **Time plot** — is the mean near zero (printed in the title), and is the
#   spread steady across the window rather than growing or shrinking?
# - **ACF** — is anything left that the method should have caught? Bars
#   outside the grey band are structure the forecast missed. The Ljung-Box
#   p-value in the title tests this formally; above 0.05 means the residuals
#   are indistinguishable from white noise.
# - **Histogram** — roughly symmetric and bell-shaped, or skewed / heavy-tailed?
#
# FPP's remedy for a non-zero mean is blunt and worth knowing: *"if the
# residuals have mean m, simply add m to all forecasts."* Correlation is
# harder, and the book defers it to Chapter 10.
#
# These are held-out residuals over a short window — 56 days per store — so
# treat the verdicts as a check, not a proof.

# %%
for model in ("xgboost", "ets"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    plots.plot_residual_diagnostics(
        predictions, top_id, model, season=cfg.season, axes=axes
    )
    fig.suptitle(f"{model} at {top_store} — residual diagnostics", y=1.04, fontsize=11)
    fig.tight_layout()
fig

# %% [markdown]
# ## What this step should have established
#
# The answers go in a dated file under `findings/`, alongside step 3's:
#
# 1. **Which methods beat the benchmarks, and by how much** — RMSSE means and
#    medians, and whether the models are separable from each other.
# 2. **Whether the ranking depends on the store** — the by-store table and
#    figure, read along the volume gradient.
# 3. **How error grows with horizon** — and whether the models' edge is at
#    short lead times only.
# 4. **Whether the residuals are clean** — zero mean, no leftover structure.
#    A non-zero mean is a bias worth correcting; leftover ACF structure is a
#    feature the model is missing.
# 5. **Where the models lose** — named, not hidden. The stores or folds where
#    a benchmark wins are the honest edge of what the method can do.
#
# Only now is it reasonable to write anything up.
