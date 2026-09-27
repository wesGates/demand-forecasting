# %% [markdown]
# # Step 3 — Preliminary exploratory analysis
#
# FPP §1.6 lists five steps for a forecasting task. Step 3 is exploratory
# analysis, and the book says to always start by graphing the data. Nothing
# here fits a model.
#
# This notebook is the procedure. It runs unchanged for any item in
# `STUDY_ITEMS`. What it finds for a particular item (the level shifts, the
# SNAP lift, whether price is worth using) goes in a separate dated
# write-up, so this file stays true whichever product it is pointed at.
#
# Each section answers one of the questions FPP §1.6 says to ask before
# modelling, plus one measurement of our own. How sporadic and how
# inconsistent each series is, which decides whether a learned model is the
# right tool at all.
#
# Run a cell with Shift+Enter. Variables persist between cells, like MATLAB
# sections.

# %%
import sys
from pathlib import Path

# Put the repo root on sys.path so `from src import ...` resolves however this
# file is run, cell-by-cell in the Interactive Window or as a plain script from
# any directory. Changed from `sys.path.append("..")` to avoid the
# ModuleNotFoundError that `python notebooks/01_explore.py` raised from the repo root.
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
import pandas as pd

from src import plots
from src.step1_problem import STUDY_ITEMS, Config
from src.step2_data import MAJOR_EVENTS, load_panel
from src.step3_explore import (
    class_table,
    classification_cutoff,
    event_effects,
    series_stats,
    summarise,
)

plots.use_style()

ITEM = STUDY_ITEMS[0]  # this notebook explores one item at a time

# %% [markdown]
# ### Where figures appear
#
# By default every figure opens in its own resizable window with a zoom, pan
# and save toolbar (MATLAB figure windows, essentially) and an inline copy is
# kept in the Interactive Window, so there is a scrollable record of
# everything that was drawn. Set `FIGURE_BACKEND = "inline"` in the next cell
# and re-run it to keep only the inline copies. It only affects figures
# created after the cell runs, so run it before the plotting cells.

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
# ## How to read these charts
#
# Two conventions recur throughout.
#
# Grey is actual daily sales and blue is a rolling average. The grey is the
# real daily line. About 1,900 daily points in one panel zig-zag faster than
# the eye can follow and read as a cloud. The vertical thickness of the grey
# at any date is how much day-to-day variation there was around then. Follow
# the blue line for level and trend.
#
# Wherever stores appear side by side they are ordered by average daily
# sales, busiest first, so position carries information too.
#
# Every plotting cell ends with `plots.show()`. It shows each figure made in
# that cell once. With the pop-out backend, a window and an inline copy; with
# the inline backend, the inline copy alone. Ending a cell with a bare `fig`
# would display it a second time.
#
# `plots.use_style()` switches on constrained layout, which makes room for
# the legends placed below each axes. Do not add `tight_layout()`.

# %% [markdown]
# ## The problem definition
#
# Everything downstream reads from this one object. Changing the horizon or
# the subset means editing this cell.

# %%
cfg = Config(item_ids=STUDY_ITEMS)
print(cfg.describe())

# %% [markdown]
# ## Load the data
#
# The panel is wide-to-long reshaped, joined to the calendar and price tables,
# with each row taking the SNAP flag for its own state.
#
# Watch the pre-launch trim. M5 keeps a row for every item on every date,
# even before the product existed. Those leading zeros mean "not stocked",
# and a missing price is the signal. How much gets trimmed is a finding in
# itself. A few percent means a long-established item, a fifth of the rows
# means it launched well into the history.
#
# Two more things the loader does, both reported in the printout. Every
# store records zero on Christmas Day because the stores were shut. That is a
# missing observation (FPP §13.7), so the day is filled with the same-weekday
# mean of the four preceding weeks and flagged `closure`. Preceding weeks
# only, so nothing from later in the year leaks into the fill. And three
# holiday-proximity columns are attached, `is_holiday`, `days_to_holiday`
# and `days_since_holiday`, for the events that measurably move this item.
# The event-effect figure further down is the evidence for that list.

# %%
df = load_panel(cfg)

# %% [markdown]
# ## Are these series even usable? The availability screen
#
# Before asking whether demand is sporadic, ask whether the item was on the
# shelf. A long unbroken run of zero sales almost always means the item was
# unavailable, and it inflates every sparsity statistic the same way genuine
# intermittent demand would.
#
# M5 has no inventory data, so the two cannot be told apart from sales alone.
# The screen below reports the longest zero run per series and warns if any
# exceeds a month. If it warns, stop. The demand classes that follow would
# be measuring stocking gaps, and the item probably needs replacing or its
# history cutting to the last continuously stocked stretch.

# %%
stats = series_stats(df, cfg)
cutoff = classification_cutoff(df, cfg)
top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]  # busiest store
print(summarise(stats, cutoff))

# %% [markdown]
# ## How sporadic, how inconsistent? Classifying demand
#
# Two numbers per series, both computed only on data before the first scored
# day, so no label is informed by a day the model is later judged on.
#
# - ADI, days divided by days with a sale. How often does it sell? This is
#   just `1 / (1 - zero rate)`, a restatement of sparsity. Its value is in
#   being paired with the next one.
# - CV², the squared coefficient of variation of the non-zero sale sizes.
#   When it does sell, how consistent is the amount?
#
# Cut both at their conventional thresholds and four classes fall out.
#
# What to look for is whether the class varies across stores for this item.
# It only can if volume is low enough for some stores to post zeros. At 50
# units a day ADI is pinned near 1.0 by arithmetic and every store lands in
# `smooth`. A uniform class table is a fact about the item, and it decides
# which question the study can answer.

# %%
cols = ["item_id", "store_id", "mean_sales", "max_zero_run", "adi", "cv2", "demand_class"]
stats[cols].round(3)

# %%
class_table(stats)

# %% [markdown]
# ### The class map
#
# Every circle is one store-item series. The two grey crosshair lines are
# the cut points, and the italic words in the corners name the quadrants.
# Circles are tagged with their store.
#
# The cut points (ADI 1.32, CV² 0.49) are conventions from one comparison of
# forecasting methods. A series at 1.31 is no different from one at 1.33.
# Showing the cloud makes that visible in a way a count-by-class table
# cannot, and if every point sits in one corner, the spread within that
# corner is still the gradient a model comparison runs along.

# %%
fig, ax = plt.subplots(figsize=(6.4, 4.6))
plots.plot_demand_class_map(stats, highlight_item=ITEM, item_id=ITEM, ax=ax)
plots.show()

# %% [markdown]
# ## Is there a pattern? A trend? Outliers?
#
# The central exhibit. One product, every store. The product is held
# constant, so anything differing between panels is a property of that
# store's demand.
#
# One panel per store, busiest first, left to right then down. Each panel
# title is `store · demand class · mean units/day`. Grey is daily sales,
# blue the 28-day average. The orange vertical line marks where the held-out
# window begins. Everything to its right is reserved for scoring and took no
# part in any statistic on this page. Orange circles flag outliers by a
# robust control-limit rule, the residual from a centred 28-day rolling
# median scaled by the median absolute deviation, flagged beyond 4 robust
# sigma. Median and MAD because a few large spikes inflate a standard
# deviation and end up hiding themselves.
#
# The y-axes are not shared. Store volumes differ by an order of magnitude
# and a shared axis would flatten the small stores into a featureless line.
# Panel heights are then not comparable, which is why the mean is printed in
# every title.
#
# What to look for, and be ready to explain:
#
# - Level shifts. A store that steps up or down and stays there. A trend
#   model chases it, a mean is wrong on both sides.
# - Dips and spikes. Sudden collapses or one-day surges. FPP calls these
#   "outliers requiring expert explanation". M5 rarely carries the
#   explanation, and saying so is more honest than averaging over them.
# - Dead blocks, flat zero for weeks. The availability screen above should
#   already have warned.
# - Which outliers sit at zero. On a fast mover a zero-sales day is a
#   stockout, a closed store, or a recording gap. Christmas closures were
#   already filled by the loader, so any zero that remains is one of the
#   other two. The count is printed below the figure.

# %%
_ = plots.plot_store_grid(df, ITEM, stats, holdout_start=cutoff, flag_outliers=True)
plots.show()

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
# Weekday seasonality is what a 7-day forecast lives or dies on. The month
# panel says whether an annual pattern exists that is worth a feature.
#
# Look at how much taller the weekend bars are than midweek, and whether the
# month bars are flat or humped.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
plots.plot_seasonality(df, top_id, axes=axes)
fig.suptitle(f"{ITEM} at {top_store}", fontsize=11)
plots.show()

# %% [markdown]
# ### The same shape, across every store
#
# One store tells you about one store. Here each store's profile is divided
# by that store's own mean, so a 100-unit store and a 15-unit store both read
# as "how far above or below my usual level is this weekday". Without that
# step the busiest store sets the shape and the small ones vanish along the
# bottom.
#
# Thin grey lines are the stores and bold blue is their mean. Greys bunched
# around the blue mean the seasonal shape belongs to the product and one
# feature serves every store. Greys that fan out mean seasonality differs by
# store, and a pooled model needs store identity to capture it.
#
# In the pop-out window, hovering a grey line shows which store it is.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
plots.plot_seasonality_by_store(df, ITEM, axes=axes)
fig.suptitle(f"{ITEM} — seasonal shape across stores", fontsize=11)
plots.show()

# %% [markdown]
# ## Which lags actually carry signal? Autocorrelation
#
# FPP §2.8. This plot decides the lag features in step 4. Lags chosen
# because the autocorrelation says so are evidence, lags chosen because
# everyone uses them are convention.
#
# Each bar is `r_k`, how strongly a day resembles the day `k` days earlier.
# The grey band is the ±1.96/√T white-noise interval and bars inside it are
# indistinguishable from random. With about 1,900 days that band is very
# narrow, so almost everything is significant and the shape is what matters.
#
# Spikes at 7, 14, 21, 28 that decay slowly mean a persistent weekly cycle. A
# strong bar at lag 1 means the recent level matters in its own right. Bars
# at 6 and 8 nearly as tall as 7 mean the weekly peak is broad. Each of those
# is an argument for a specific feature.
#
# The right panel overlays each calendar year on a shared day-of-year axis
# (FPP §2.4). Years that trace the same shape mean it is seasonal and an
# annual feature is worth having. Years that wander independently mean it was
# drift, and a month feature would fit noise. Also look at whether the years
# sit at different levels, a multi-year drift on top of the seasonal shape.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
plots.plot_acf(df, top_id, ax=axes[0], title=f"{ITEM} at {top_store} — autocorrelation")
plots.plot_year_overlay(df, top_id, ax=axes[1])
plots.show()

# %% [markdown]
# ## Relationships between variables
#
# Price, sparsity, and calendar events. Whether each one is worth a feature
# is an item-specific answer, so the numbers behind the plots are printed.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
plots.plot_price_relationship(df, ITEM, ax=axes[0])
plots.plot_zero_rate(stats, item_id=ITEM, ax=axes[1])
plots.show()

# %% [markdown]
# Which calendar events move this item, measured. Each event's sales are
# divided by a same-weekday baseline from the surrounding weeks, on the day
# and on the two days before it, using only the data before the
# classification cutoff. The events that clear a 15% bar in either direction
# are the ones `step2_data.MAJOR_EVENTS` names, and they drive the
# holiday-proximity features and the normal/holiday split in step 5.
# Christmas has no day marker because the stores were shut, but its run-up is
# the largest in the calendar, which is why the features carry distance to
# the event and an on/off flag alone would miss it.

# %%
calendar = pd.read_csv(cfg.data_dir / "calendar.csv", parse_dates=["date"])
effects = event_effects(df, calendar)
print(effects.round(2).to_string())

fig, ax = plt.subplots(figsize=(7.5, 6.2))
plots.plot_event_effects(effects, major=MAJOR_EVENTS, ax=ax)
plots.show()

# %% [markdown]
# Is price a real variable or a clock? Price is known in advance, so it is a
# legitimate feature if it actually varies. The check below prints how many
# distinct prices exist and when each was in force.
#
# - Many prices, changing at different times in different stores. A genuine
#   economic signal. Consider `Config(use_price=True)`.
# - A handful of prices that step up on the same dates everywhere. That is a
#   calendar in disguise. It is constant inside any forecast window, so it
#   carries no within-window information, and offering it to the model
#   invites memorising a period. Leave `use_price` off and say why in the
#   write-up.

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
# SNAP benefit days are the other calendar feature known years in advance.
# The dates are fixed by state, so any effect here costs nothing to obtain.
# The lift is printed below. A consistent few percent across stores is
# usable signal.

# %%
fig, ax = plt.subplots(figsize=(9, 3))
plots.plot_snap_effect(df, ITEM, stats, ax=ax)
plots.show()

# %%
_m = _p.groupby("snap", observed=True)["sales"].mean()
print(
    f"{ITEM}: ordinary day {_m.get(0, float('nan')):.2f}  SNAP day {_m.get(1, float('nan')):.2f}  "
    f"lift {(_m.get(1, float('nan')) / _m.get(0, float('nan')) - 1) * 100:+.1f}%"
)

# %% [markdown]
# ## What this step should have established
#
# Before moving to step 4, answer each of these for the item in hand. The
# answers go in a dated write-up.
#
# 1. Is the working set clean? Longest zero run, pre-launch trim share,
#    zero rate per store. Did the availability screen warn?
# 2. What is the dominant structure? Weekly, annual, multi-year drift? Which
#    lags did the autocorrelation justify?
# 3. Which known-in-advance variables carry signal? SNAP lift, which calendar
#    events move the item (and by how much on the days before), and whether
#    price is a variable or a clock.
# 4. Does the demand class vary across stores? If yes, the study can ask
#    whether the best model changes with class. If every store is `smooth`,
#    the question becomes which model wins on this kind of item and whether
#    pooling across stores helps.
