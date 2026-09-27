"""Report figure 5: the first study's scaled error by store. Reads
tools/report/first_study_scores.csv (made by first_study_scores.py in the
first-study checkout). Run from the repo root with PYTHONPATH=. ; argv[1] is
the output png."""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from src import plots

t = pd.read_csv("tools/report/first_study_scores.csv", index_col="store_id")
MODELS = {"xgboost": "XGBoost", "arima": "ARIMA", "ets": "Exponential smoothing (ETS)"}
BENCH = {"seasonal_naive": "This day last week", "moving_average_28": "28-day moving average",
         "seasonal_naive_364": "This day last year", "mean": "Long-run average", "naive": "Yesterday's sales"}
# "Yesterday plus trend" is left out. It sits on top of "yesterday's sales" at every store and is not in Table 5.
plots.use_style(); plt.rcParams["savefig.dpi"] = 300
fig, ax = plt.subplots(figsize=(10, 4.0))
x = range(len(t))
for key, name in BENCH.items():  # benchmarks: grey markers, joined by one faint grey so each can be followed across stores
    ax.plot(x, t[key], color="#b3b3b3", lw=0.9, zorder=1)
    ax.scatter(x, t[key], marker=plots.BENCH_MARKERS[key], color=plots.INK_SOFT, s=26, zorder=2, label=name)
for key, name in MODELS.items():
    ax.plot(x, t[key], color=plots.MODEL_COLOURS[key], lw=2.2, marker="o", ms=5.5, zorder=3, label=name)
ax.axhline(1.0, color=plots.AXIS, lw=1, zorder=0)
ax.set_xticks(list(x), t.index); ax.set_ylabel("scaled error (RMSSE)")
ax.set_title("The first study: scaled error by store, busiest store on the left (lower is better)", fontsize=10.5)
handles, labels = ax.get_legend_handles_labels()
order = list(range(len(BENCH), len(labels))) + list(range(len(BENCH)))  # models first in the legend
plots.legend_below(ax, ncols=4, handles=[handles[i] for i in order], labels=[labels[i] for i in order])
fig.savefig(sys.argv[1], bbox_inches="tight"); plt.close(fig); print("wrote", sys.argv[1])
