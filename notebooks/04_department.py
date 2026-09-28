# %% [markdown]
# # A whole department: classify, screen, choose
#
# The two-item study picked its items by hand. This notebook does the first
# part of that for a whole department. It classifies every series by demand
# pattern (step 3), screens out the ones a forecast cannot be judged on, and
# prints the survivors by class. Which items go into the sweep is still a
# decision made by looking, so nothing here chooses a model or an item.
#
# Run a cell with Shift+Enter. Variables persist between cells.

# %%
import sys
from pathlib import Path

# Put the repo root on sys.path so `from src import ...` resolves however this
# file is run, cell-by-cell in the Interactive Window or as a plain script from
# any directory.
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
    pass  # not running under IPython

import warnings

import matplotlib.pyplot as plt
import pandas as pd

from src import plots
from src.step1_problem import Config
from src.step2_data import load_panel
from src.step3_explore import classification_cutoff, series_stats

plots.use_style()

DEPT = "FOODS_3"
STORES = ("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3")

# The screen. A series is only worth forecasting if it was on the shelf.
# The history floor is the harness's own minimum training window, so it is
# read from Config rather than chosen here. The zero-run cap is a judgment:
# there is no published cut-off, and a chance-based rule flags nearly every
# series in M5, since stock-out gaps are everywhere in it. 90 days keeps a
# usable sample; the findings rerun the key comparisons at 60 days as a check.
MIN_HISTORY_DAYS = Config().min_train_days
MAX_ZERO_RUN = 90  # longest run of zero-sales days allowed at any store

# %% [markdown]
# ## Classify every series
#
# One store at a time. The whole department in one panel is 16 million rows
# and about 8 GB, which this machine cannot hold beside everything else, so
# the loop loads 823 series per store and stacks the results. The
# classification window ends on the same date for every store, so the labels
# are what a single load would give. About 70 seconds; the result is cached
# beside the panels.

# %%
cache = Path("cache") / f"dept_stats_{DEPT}.parquet"
if cache.exists():
    stats = pd.read_parquet(cache)
else:
    parts = []
    for store in STORES:
        cfg = Config(dept_id=DEPT, store_ids=(store,), fold_step=1, n_folds=358)
        df = load_panel(cfg, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # the zero-run warning fires per store; the screen below handles it
            parts.append(series_stats(df, cfg))
        print(store, "done", flush=True)
    cutoff = classification_cutoff(df, cfg)
    stats = pd.concat(parts, ignore_index=True)
    stats.to_parquet(cache)
    print("classification window ends", cutoff.date())

print(f"{len(stats):,} series, {stats['item_id'].nunique()} items")
print(stats["demand_class"].value_counts().to_string())

# %% [markdown]
# ## The screen
#
# M5 has no inventory data, so a long run of zero sales cannot be told from
# an item that was not on the shelf. The two-item study checked this by
# hand. Here the numbers do it: per item, the shortest history at any store
# and the longest zero run at any store. The table shows how many items pass
# as the two limits move. The count rises steadily with the cap, with no
# natural break, which is why the cap is stated as a choice and checked
# rather than derived.

# %%
item = stats.groupby("item_id").agg(
    stores=("id", "size"),
    min_history=("n_days", "min"),
    max_zero_run=("max_zero_run", "max"),
    mean_sales=("mean_sales", "mean"),
    majority_class=("demand_class", lambda s: s.value_counts().idxmax()),
    classes=("demand_class", "nunique"),
)
print("items passing, all ten stores present:")
runs = (30, 60, 90, 180, 365)
print(f"{'history ≥':>10} | " + " | ".join(f"run ≤ {z:>3}" for z in runs))
for d in (365, 730):
    n = [
        ((item.stores == 10) & (item.min_history >= d) & (item.max_zero_run <= z)).sum()
        for z in runs
    ]
    print(f"{d:>8} d | " + " | ".join(f"{x:>9}" for x in n))

passing = item[
    (item.stores == 10)
    & (item.min_history >= MIN_HISTORY_DAYS)
    & (item.max_zero_run <= MAX_ZERO_RUN)
]
print(
    f"\nchosen screen: history ≥ {MIN_HISTORY_DAYS} d, zero run ≤ {MAX_ZERO_RUN} d → {len(passing)} items"
)
print(passing["majority_class"].value_counts().to_string())

# %% [markdown]
# ## The class map
#
# Every series is a dot. Ink is a series whose item passed the screen, grey
# is one that did not. The cut points are the two lines. Most of the
# department sits right of the ADI cut: items that do not sell every day at
# a given store.

# %%
fig, ax = plt.subplots(figsize=(7.4, 5.2))
plots.plot_demand_class_cloud(stats, keep=stats["item_id"].isin(passing.index), ax=ax)
plots.show()

# %% [markdown]
# ## Items to choose from
#
# The survivors, by majority class across the ten stores, busiest first.
# `classes` is how many classes the item's ten stores span; 1 means every
# store agrees. The sweep takes a list of item ids. Pick from here, write the
# list down with the reason, and pass it to `tools/sweep.py`.

# %%
pd.set_option("display.max_rows", 400)
for cls in ("smooth", "erratic", "intermittent", "lumpy"):
    g = passing[passing["majority_class"] == cls].sort_values(
        "mean_sales", ascending=False
    )
    print(f"\n{cls}: {len(g)} items")
    print(
        g[["mean_sales", "min_history", "max_zero_run", "classes"]].round(2).to_string()
    )

passing.to_csv(Path("cache") / f"dept_screen_{DEPT}.csv")

# %% [markdown]
# ## What this step should have established
#
# 1. How the department splits by demand class, and how much of it the
#    screen removes.
# 2. Which items are candidates in each class, with their volume.
# 3. The chosen items and why, written down before the sweep runs.
