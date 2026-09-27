"""README and report figure 1: average weekly error per store for last week's number, ARMA,
ARIMA and the final model. Fast mover, every-day layout, from the cache.
Run from the repo root with PYTHONPATH=. ; argv[1] is the output png."""
import sys
import matplotlib.pyplot as plt
from src import plots
from src.step1_problem import Config
from src.step5_evaluate import _read_cached

L = dict(fold_step=1, n_folds=358); item = "FOODS_3_586"
per = Config(item_ids=(item,), **L)
pooled = Config(item_ids=(item,), pool_by="item_id", **L)
runs = [("This day last week", _read_cached(per, "seasonal_naive"), plots.INK_MUTED),
        ("ARMA", _read_cached(per, "arma"), plots.MODEL_COLOURS["arma"]),
        ("ARIMA", _read_cached(per, "arima"), plots.MODEL_COLOURS["arima"]),
        ("XGBoost (final)", _read_cached(pooled, "xgboost_rel"), plots.MODEL_COLOURS["xgboost_rel@pooled"])]

def weekly_error(f):
    # absolute error on each store-week's total, averaged; closure days left out
    g = f[~f["closure"].astype(bool)].groupby(["store_id", "origin"]).agg(fc=("forecast", "sum"), ac=("actual", "sum"))
    return float((g.fc - g.ac).abs().mean())

err = [weekly_error(f) for _, f, _ in runs]
final = err[-1]
for (name, _, _), e in zip(runs, err):
    print(f"{name:20s} {e:.1f}  final lower by {1 - final / e:.1%}")

plots.use_style(); plt.rcParams["savefig.dpi"] = 300
fig, ax = plt.subplots(figsize=(8, 2.6))
y = range(len(runs))[::-1]  # final model at the bottom
for yi, (name, _, col), e in zip(y, runs, err):
    ax.barh(yi, e, color=col, height=0.6)
    ax.text(e + 0.4, yi, f"{e:.1f}", va="center", fontsize=9,
            fontweight="bold" if name.startswith("XGBoost") else "normal")
    if not name.startswith("XGBoost"):
        ax.text(1.02, yi, f"XGBoost {1 - final / e:.0%} lower", va="center", fontsize=9.5,
                color=plots.INK_SOFT, transform=ax.get_yaxis_transform())
ax.set_yticks(list(y), [n for n, _, _ in runs])
ax.set_xlim(0, 40)
ax.set_xlabel("average weekly error per store, units")
ax.set_title("Fast mover, ten stores, one year of daily forecasts (lower is better)", fontsize=10.5)
ax.grid(axis="y", visible=False)
fig.savefig(sys.argv[1], bbox_inches="tight"); plt.close(fig); print("wrote", sys.argv[1])
