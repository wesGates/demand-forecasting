# %% [markdown]
# # Step 3 — Preliminary exploratory analysis
#
# FPP §1.6 lists five steps for a forecasting task. Step 3 is exploratory
# analysis, and the book is blunt about it: **always start by graphing the
# data.** Nothing here fits a model.
#
# Each section below answers one of the questions FPP says to ask, plus one
# measurement specific to this project: how *sporadic* and how *inconsistent*
# each series is, which is what decides whether a machine-learning model is the
# right tool at all.
#
# Run a cell with **Shift+Enter**. Variables persist between cells, exactly
# like MATLAB sections.

# %%
import sys

sys.path.append("..")  # so `src` imports work from inside notebooks/

import matplotlib.pyplot as plt

from src import plots
from src.step1_problem import STUDY_ITEMS, Config
from src.step2_data import load_panel
from src.step3_explore import class_table, classification_cutoff, series_stats, summarise

plots.use_style()

SPINE = "FOODS_3_586"  # the fast-moving staple
VARIES = "FOODS_3_232"  # the item whose demand class changes by store

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
# daily line. With ~1,885 daily points squeezed into one panel it zig-zags
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
# Every plotting cell below ends that way, so all five figures appear in both
# places.

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
# Three items across all ten stores. The panel is wide-to-long reshaped, joined
# to the calendar and price tables, with each row taking the SNAP flag for its
# own state.
#
# Watch the pre-launch trim: M5 keeps a row for every item on every date, even
# before the product existed. Those leading zeros mean "not stocked", not "no
# demand", and a missing price is the signal.

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
# These three items were chosen precisely because they don't have the problem:
# across ~1,885 days, the longest zero run at any store is 11 days.
#
# (Most `FOODS_3` items are *not* like this — the median longest zero run across
# all 8,230 store-item series is 83 days. That is a limitation of the data
# worth stating, not a property of demand.)

# %%
stats = series_stats(df, cfg)
cutoff = classification_cutoff(df, cfg)
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

# %%
cols = ["item_id", "store_id", "mean_sales", "max_zero_run", "adi", "cv2", "demand_class"]
stats[cols].round(3)

# %% [markdown]
# The same product lands in different classes at different stores — and that is
# the premise of this whole study. Note it happens for the two low-volume items
# and *not* for the fast mover: at 50 units a day a store never posts a zero, so
# ADI is pinned near 1.0 by arithmetic. **Class variation requires low volume.**

# %%
class_table(stats)

# %% [markdown]
# ### The class map
#
# **Reading it:** every circle is one store-item series, 30 in total. The two
# grey crosshair lines are the cut points, chopping the plot into four
# quadrants; the italic words in the corners name the quadrant they sit in
# (they label regions, not points). Solid blue circles are the highlighted
# item, tagged with their store; hollow grey circles are the other two items,
# shown for context.
#
# The cut points (ADI 1.32, CV² 0.49) are conventions from one comparison of
# forecasting methods — not laws. A series at 1.31 is not meaningfully different
# from one at 1.33. Showing the cloud makes that visible in a way a
# count-by-class table cannot.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
plots.plot_demand_class_map(stats, highlight_item=SPINE, ax=axes[0])
plots.plot_demand_class_map(stats, highlight_item=VARIES, ax=axes[1])
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
# sales, blue the 28-day average, x-axis is 2011–2016.
#
# Y-axes are deliberately **not** shared — TX_2 sells 101/day and WI_2 sells
# 15/day, so a shared axis would flatten the small stores into a featureless
# line. The cost is that panel heights are not comparable between stores,
# which is why the mean is printed in every title.

# %%
plots.plot_store_grid(df, SPINE, stats)

# %% [markdown]
# Things to look for and be ready to explain:
#
# - **Level shifts** — WI_3 steps down through 2014 and stays there. A trend
#   model would chase it; a mean would be wrong on both sides.
# - **Dips and spikes** — CA_2 collapses in early 2015 then recovers; WI_1 has a
#   sharp mid-2014 spike. These are the "outliers requiring expert explanation"
#   FPP asks about. We cannot explain them from M5 alone, which is itself worth
#   saying.
# - **No dead blocks** — which is why these items were selected.

# %%
plots.plot_store_grid(df, VARIES, stats)

# %% [markdown]
# The same ten stores for a low-volume item look completely different — this is
# where "lumpy" stops being a label and becomes something you can see.

# %% [markdown]
# ## Is seasonality important?
#
# Weekday seasonality is what a 7-day forecast lives or dies on. The month panel
# says whether an annual pattern exists worth giving the model a feature for.

# %%
spine_top = stats[stats["item_id"] == SPINE].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
plots.plot_seasonality(df, spine_top["id"], axes=axes)
fig.suptitle(f"{SPINE} at {spine_top['store_id']}", y=1.05, fontsize=11)
fig.tight_layout()
fig

# %% [markdown]
# ## How strong are the relationships between variables?
#
# SNAP benefit days are the most promising calendar feature in this dataset:
# the dates are fixed by state and known years in advance, so any effect here is
# forecasting signal that costs nothing to obtain.
#
# **Reading it:** two bars per store — blue is the mean on ordinary days,
# orange the mean on SNAP days. Measured across the whole history the lift is
# +6.5% for FOODS_3_586, +4.3% for FOODS_3_232, +5.7% for FOODS_3_156. Small,
# but consistent across nearly every store, which is what makes it usable.

# %%
fig, ax = plt.subplots(figsize=(9, 3))
plots.plot_snap_effect(df, SPINE, stats, ax=ax)
fig.tight_layout()
fig

# %% [markdown]
# ## What this step establishes
#
# 1. The three study items are continuously stocked, so their demand classes
#    measure customer behaviour rather than stocking gaps.
# 2. Demand class varies by store for the low-volume items and not for the fast
#    mover — so the study has both a "which model for a fast mover" question and
#    a "does the answer change with demand class" question.
# 3. Weekday seasonality and SNAP effects are visible, which is what the feature
#    set in step 4 needs to capture.
#
# Only now is it reasonable to fit anything.
