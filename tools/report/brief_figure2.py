"""Brief figure 2: average error on the week's total order, in units, by store,
for last week's number, ARMA, ARIMA and the final model. Fast mover, every-day layout.
Run from the repo root with PYTHONPATH=. ; argv[1] is the output png."""
import sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import plots
from src.step1_problem import Config
from src.step5_evaluate import _read_cached
from src.step2_data import load_panel
from src.step3_explore import series_stats
L = dict(fold_step=1, n_folds=358); item = "FOODS_3_586"
per = Config(item_ids=(item,), **L)
runs = {"This day last week": _read_cached(per, "seasonal_naive"), "ARMA": _read_cached(per, "arma"), "ARIMA": _read_cached(per, "arima"),
        "XGBoost, pooled, level-relative (final)": _read_cached(Config(item_ids=(item,), pool_by="item_id", **L), "xgboost_rel")}
def weekly_mae(f):
    g = f[~f["closure"].astype(bool)].groupby(["store_id", "origin"]).agg(fc=("forecast", "sum"), ac=("actual", "sum"))
    return (g.fc - g.ac).abs().groupby(level=0).mean()
tab = pd.DataFrame({k: weekly_mae(v) for k, v in runs.items()})
df = load_panel(per, verbose=False); stats = series_stats(df, per)
order = stats.sort_values("mean_sales", ascending=False)["store_id"].tolist()
tab = tab.reindex(order); print(tab.round(1).to_string()); print("all stores:", tab.mean().round(1).to_dict())
plots.use_style(); plt.rcParams["savefig.dpi"] = 300  # crisp in the PDF
fig, ax = plt.subplots(figsize=(7.4, 3.2))
x = np.arange(len(order)); w = 0.2
for i, (name, col) in enumerate(zip(tab.columns, (plots.INK_MUTED, plots.MODEL_COLOURS["arma"], plots.MODEL_COLOURS["arima"], plots.MODEL_COLOURS["xgboost_rel@pooled"]))):
    ax.bar(x + (i - 1.5) * w, tab[name], width=w, color=col, label=name)
ax.set_xticks(x, order); ax.set_ylabel("average error on the week's total, units")
ax.set_title("Weekly error by store, fast mover, busiest store on the left", fontsize=10.5); ax.grid(axis="x", visible=False)
plots.legend_below(ax, ncols=4)
fig.savefig(sys.argv[1], bbox_inches="tight"); plt.close(fig); print("wrote", sys.argv[1])
