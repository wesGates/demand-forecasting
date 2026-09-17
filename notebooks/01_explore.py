# %% [markdown]
# # Step 3 — Preliminary exploratory analysis
#
# FPP §1.6 lists five steps for a forecasting task. Step 3 is exploratory
# analysis, and the book is blunt about it: **always start by graphing the
# data.** Nothing here fits a model.
#
# This notebook is the *procedure*. It runs unchanged for any item in
# `STUDY_ITEMS`. What it finds for a particular item — the level shifts, the
# SNAP lift, whether price is worth using — belongs in a dated write-up under
# `findings/`, not here, so that this file stays true no matter which product it
# is pointed at.
#
# Each section answers one of the questions FPP §1.6 says to ask before
# modelling, plus one measurement specific to this project: how *sporadic* and
# how *inconsistent* each series is, which decides whether a machine-learning
# model is the right tool at all.
#
# Run a cell with **Shift+Enter**. Variables persist between cells, exactly
# like MATLAB sections.

# %%
import sys

sys.path.append("..")  # so `src` imports work from inside notebooks/

# Pick up edits to anything under src/ without restarting the kernel.
# Python caches imported modules, so without this an edited src/plots.py
# stays invisible to a running session and you get AttributeError on a
# function you can see in the file.
try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # noqa: F821
    get_ipython().run_line_magic("autoreload", "2")  # noqa: F821
except NameError:
    pass  # not running under IPython

import matplotlib.pyplot as plt

from src import plots
from src.step1_problem import STUDY_ITEMS, Config
from src.step2_data import load_panel
from src.step3_explore import class_table, classification_cutoff, series_stats, summarise

plots.use_style()

ITEM = STUDY_ITEMS[0]  # this notebook explores one item at a time

# %% [markdown]
# ### Optional — pop figures out into their own windows
#
# By default plots appear inline in the Interactive Window, which is fine for
# scrolling but poor for inspecting. Run the next cell and every figure from
# then on opens in its **own resizable window** with a zoom / pan / save
# toolbar — MATLAB figure windows, essentially.
#
# Swap `"tk"` for `"inline"` and re-run to go back. Changing it only affects
# figures created *after* the cell runs, so run it before the plotting cells.

# %%
try:
    get_ipython().run_line_magic("matplotlib", "tk")  # noqa: F821  ("inline" to revert)
except NameError:
    pass  # not running under IPython - leave the backend alone

# %% [markdown]
# ## How to read these charts
#
# Two conventions recur throughout, so they are worth learning once.
#
# **Grey means actual daily sales. Blue means a rolling average.**
# The grey is *not* a shaded band or a confidence interval — it is the real
# daily line. With ~1,900 daily points squeezed into one panel it zig-zags
# faster than the eye can follow and reads as a cloud. That is informative:
# the **vertical thickness of the grey** at any date is how much day-to-day
# variation there was around then. Thick grey = volatile; thin grey = steady.
# Follow the blue line for level and trend.
#
# **Volume ordering.** Wherever stores appear side by side they are ordered by
# average daily sales, busiest first. So position carries information too.
#
# **Why some cells end with a bare `fig`.** The Interactive Window displays
# whatever the last line of a cell evaluates to. A cell ending in
# `fig.tight_layout()` evaluates to `None`, so that figure appears *only* in its
# pop-out window; ending the cell with `fig` makes it render inline as well.
# Every plotting cell below ends that way, so all figures appear in both places.

# %% [markdown]
# ## The problem definition
#
# Everything downstream reads from this one object. Changing the horizon or the
# subset means editing this cell, not hunting through the code.

# %%
cfg = Config(item_ids=STUDY_ITEMS)
print(cfg.describe())

# %% [markdown]
# ## Load the data
#
# The panel is wide-to-long reshaped, joined to the calendar and price tables,
# with each row taking the SNAP flag for its own state.
#
# Watch the pre-launch trim: M5 keeps a row for every item on every date, even
# before the product existed. Those leading zeros mean "not stocked", not "no
# demand", and a missing price is the signal. **How much gets trimmed is itself
# a finding** — a few percent means a long-established item; a fifth of the rows
# means it launched well into the history.

# %%
df = load_panel(cfg)

# %% [markdown]
# ## Are these series even usable? — the availability screen
#
# Before asking whether demand is *sporadic*, ask whether the item was *on the
# shelf*. A long unbroken run of zero sales almost always means unavailability,
# not indifference — and it inflates every sparsity statistic exactly like
# genuine intermittent demand would.
#
# M5 has no inventory data, so the two cannot be distinguished from sales alone.
# The screen below reports the longest zero run per series and **warns** if any
# exceeds a month. If it warns, stop: the demand classes that follow are
# measuring stocking gaps, and the item probably needs replacing or its history
# needs cutting to the last continuously-stocked stretch.

# %%
stats = series_stats(df, cfg)
cutoff = classification_cutoff(df, cfg)
top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]  # busiest store
print(summarise(stats, cutoff))

# %% [markdown]
# ## How sporadic, how inconsistent? — classifying demand
#
# Two numbers per series, both computed **only on data before the first scored
# day**, so no label is informed by a day the model will later be judged on.
#
# - **ADI** — days divided by days-with-a-sale. *How often does it sell?*
#   Note that this is just `1 / (1 - zero rate)`; it is a restatement of
#   sparsity, not new information. Its value is in being paired with:
# - **CV²** — squared coefficient of variation of the non-zero sale sizes.
#   *When it does sell, how consistent is the amount?*
#
# Cut both at their conventional thresholds and four classes fall out.
#
# **What to look for:** does the class vary across stores for this item? It
# only can if volume is low enough for some stores to post zeros — at 50 units
# a day, ADI is pinned near 1.0 by arithmetic and every store lands in
# `smooth`. So a uniform class table is not a failure of the method; it is a
# fact about the item, and it decides which research question the study can
# actually answer.

# %%
cols = ["item_id", "store_id", "mean_sales", "max_zero_run", "adi", "cv2", "demand_class"]
stats[cols].round(3)

# %%
class_table(stats)

# %% [markdown]
# ### The class map
#
# **Reading it:** every circle is one store-item series. The two grey crosshair
# lines are the cut points, chopping the plot into four quadrants; the italic
# words in the corners name the quadrant they sit in (they label regions, not
# points). Circles are tagged with their store.
#
# The cut points (ADI 1.32, CV² 0.49) are conventions from one comparison of
# forecasting methods — not laws. A series at 1.31 is not meaningfully different
# from one at 1.33. Showing the cloud makes that visible in a way a
# count-by-class table cannot — and if every point sits in one corner, the
# *spread within* that corner is still the gradient a model comparison runs
# against.

# %%
fig, ax = plt.subplots(figsize=(6.4, 4.6))
plots.plot_demand_class_map(stats, highlight_item=ITEM, item_id=ITEM, ax=ax)
fig.tight_layout()
fig

# %% [markdown]
# ## Is there a pattern? A trend? Outliers?
#
# The central exhibit: one product, every store. The product is held constant,
# so anything differing between panels is a property of that store's demand.
#
# **Reading it:** one panel per store, busiest first, left to right then down.
# Each panel title is `store · demand class · mean units/day`. Grey is daily
# sales, blue the 28-day average. **The orange vertical line** marks where the
# held-out window begins — everything to its right is reserved for scoring and
# took no part in any statistic on this page. **Orange circles** flag outliers
# by a robust control-limit rule: residual from a centred 28-day rolling
# *median*, scaled by the median absolute deviation, flagged beyond 4 robust
# sigma. Median and MAD rather than mean and standard deviation for the usual
# control-chart reason — a few large spikes inflate a standard deviation and
# end up concealing themselves.
#
# Y-axes are deliberately **not** shared. Store volumes can differ by an order
# of magnitude, and a shared axis would flatten the small stores into a
# featureless line. The cost is that panel heights are not comparable between
# stores, which is why the mean is printed in every title.
#
# **What to look for, and be ready to explain:**
#
# - **Level shifts** — a store that steps up or down and stays there. A trend
#   model chases it; a mean is wrong on both sides.
# - **Dips and spikes** — sudden collapses or one-day surges. These are FPP's
#   "outliers requiring expert explanation." M5 rarely carries the explanation,
#   and saying so is more honest than averaging over them.
# - **Dead blocks** — flat zero for weeks. If you see any, the availability
#   screen above should already have warned.
# - **Which outliers sit at zero.** On a fast mover a zero-sales day is not
#   demand; it is a stockout, a closed store, or a recording gap. The count is
#   printed below the figure.

# %%
plots.plot_store_grid(df, ITEM, stats, holdout_start=cutoff, flag_outliers=True)

# %%
# How many days the outlier rule flags, and how many of those are zeros.
_n_out = _n_zero = 0
for _sid, _g in df[df["item_id"] == ITEM].groupby("id", observed=True):
    _g = _g.sort_values("date")
    _odd = plots._outliers(_g["sales"])
    _n_out += int(_odd.sum())
    _n_zero += int((_odd & (_g["sales"] == 0)).sum())
print(
    f"{ITEM}: {_n_out:,} of {len(df[df['item_id'] == ITEM]):,} days flagged "
    f"({_n_out / len(df[df['item_id'] == ITEM]):.2%}) - "
    f"{_n_out - _n_zero} high spikes, {_n_zero} zero-sales days"
)

# %% [markdown]
# ## Is seasonality important?
#
# Weekday seasonality is what a 7-day forecast lives or dies on. The month panel
# says whether an annual pattern exists worth giving the model a feature for.
#
# **What to look for:** how much taller the weekend bars are than midweek, and
# whether the month bars are flat or humped.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
plots.plot_seasonality(df, top_id, axes=axes)
fig.suptitle(f"{ITEM} at {top_store}", y=1.05, fontsize=11)
fig.tight_layout()
fig

# %% [markdown]
# ### The same shape, across every store
#
# One store tells you about one store. Here each store's profile is divided by
# that store's own mean, so a 100-unit store and a 15-unit store both read as
# "how far above or below my usual level is this weekday?" — without that step
# the busiest store sets the shape and the small ones vanish along the bottom.
#
# **Reading it:** thin grey lines are the individual stores; bold blue is their
# mean. **What to look for:** if the greys bunch tightly around the blue, the
# seasonal shape is a property of the *product* and one feature serves every
# store. If they fan out, seasonality differs by store, and a pooled model
# would need store identity to capture it.
#
# In the pop-out window, hovering a grey line shows which store it is.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
plots.plot_seasonality_by_store(df, ITEM, axes=axes)
fig.suptitle(f"{ITEM} — seasonal shape across stores", y=1.05, fontsize=11)
fig.tight_layout()
fig

# %% [markdown]
# ## Which lags actually carry signal? — autocorrelation
#
# FPP §2.8. This is the plot that decides the lag features in step 4. Choosing
# lags 1, 7, 14, 28 because everyone does is convention; choosing them because
# the autocorrelation says so is evidence.
#
# **Reading it:** each bar is `r_k`, how strongly a day resembles the day `k`
# days earlier. The grey band is the ±1.96/√T white-noise interval — bars inside
# it are indistinguishable from random. With ~1,900 days that band is very
# narrow, so almost everything is significant; what matters is the *shape*.
#
# **What to look for:** spikes at 7, 14, 21, 28 that decay slowly mean a
# persistent weekly cycle. A strong bar at lag 1 means the recent *level*
# matters in its own right, not only through the weekly pattern. Bars at 6 and
# 8 nearly as tall as 7 mean the weekly peak is broad. Each of those is an
# argument for a specific feature.
#
# The right panel overlays each calendar year on a shared day-of-year axis
# (FPP §2.4) to answer "are there business cycles?". **If the years trace the
# same shape** it is seasonal, and an annual feature is worth having. **If they
# wander independently** it was drift, and a month feature would fit noise.
# Also look at whether the years sit at different *levels* — a multi-year drift
# on top of the seasonal shape.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
plots.plot_acf(df, top_id, ax=axes[0], title=f"{ITEM} at {top_store} — autocorrelation")
plots.plot_year_overlay(df, top_id, ax=axes[1])
fig.tight_layout()
fig

# %% [markdown]
# ## Relationships between variables
#
# Price, calendar events, and sparsity. Whether each one is worth a feature is
# an *item-specific* answer, so the numbers behind the plots are printed rather
# than described.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
plots.plot_price_relationship(df, ITEM, ax=axes[0])
plots.plot_event_effect(df, ITEM, ax=axes[1])
plots.plot_zero_rate(stats, item_id=ITEM, ax=axes[2])
fig.tight_layout()
fig

# %% [markdown]
# **Is price a real variable, or a clock?** Price is known in advance, so it is
# a legitimate feature — *if it actually varies*. The check below prints how
# many distinct prices exist and when each was in force.
#
# - Many prices, changing at different times in different stores: a genuine
#   economic signal. Consider `Config(use_price=True)`.
# - A handful of prices that step up on the same dates everywhere: that is a
#   calendar in disguise. It is constant inside any forecast window, so it
#   carries no within-window information, and offering it to the model invites
#   memorising a *period* rather than learning a *relationship*. Leave
#   `use_price` off, and say why in the write-up.

# %%
_p = df[df["item_id"] == ITEM]
print(
    f"distinct prices across all stores and weeks: {sorted(_p['sell_price'].dropna().unique())}"
)
print(_p.groupby("sell_price", observed=True)["date"].agg(["min", "max"]).to_string())
print(
    f"\nstores that all follow the same price schedule: "
    f"{_p.groupby('store_id', observed=True)['sell_price'].nunique().eq(_p['sell_price'].nunique()).all()}"
)

# %% [markdown]
# **Calendar events** — holidays are the other calendar feature known years in
# advance. The event plot shows each event type's mean as a percentage
# difference from an ordinary day, with the sample count beside it. An effect
# resting on a few dozen observations is a hint; one resting on several hundred
# is a feature.
#
# **SNAP benefit days** are the most promising calendar feature in this dataset:
# the dates are fixed by state and known years ahead, so any effect here costs
# nothing to obtain. The lift is printed below; a consistent few percent across
# stores is usable signal.

# %%
fig, ax = plt.subplots(figsize=(9, 3))
plots.plot_snap_effect(df, ITEM, stats, ax=ax)
fig.tight_layout()
fig

# %%
_m = _p.groupby("snap", observed=True)["sales"].mean()
print(
    f"{ITEM}: ordinary day {_m.get(0, float('nan')):.2f}  SNAP day {_m.get(1, float('nan')):.2f}  "
    f"lift {(_m.get(1, float('nan')) / _m.get(0, float('nan')) - 1) * 100:+.1f}%"
)

# %% [markdown]
# ## What this step should have established
#
# Before moving to step 4, you should be able to answer each of these for the
# item in hand — and the answers go in a dated file under `findings/`:
#
# 1. **Is the working set clean?** Longest zero run, pre-launch trim share,
#    zero-rate per store. Did the availability screen warn?
# 2. **What is the dominant structure?** Weekly? Annual? Multi-year drift?
#    Which lags did the autocorrelation justify?
# 3. **Which known-in-advance variables carry signal?** SNAP lift, event
#    effects by type, and whether price is a variable or a clock.
# 4. **Does the demand class vary across stores?** If yes, the study can ask
#    "does the best model change with class." If every store is `smooth`, the
#    question becomes "which model wins on this kind of item, and does pooling
#    across stores help" — a different, still legitimate, question.
#
# Only now is it reasonable to fit anything.
