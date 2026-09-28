"""Two figures from the department sweep: scaled error by demand class for
every method, and the order step's cost of service by class. Reads the
tables tools/sweep_report.py and tools/order_by_class.py wrote.

    PYTHONPATH=. python tools/sweep_figures.py [out_dir]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import plots

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures/item11"); OUT.mkdir(parents=True, exist_ok=True)
plots.use_style(); plt.rcParams["savefig.dpi"] = 300
NAME = {"seasonal_naive": "This day last week", "moving_average_28": "28-day average", "ets": "ETS", "croston": "Croston",
        "sba": "SBA", "tsb": "TSB", "xgboost_rel@item_id": "XGBoost, pooled per item, level-relative",
        "xgboost_tweedie@store_id": "XGBoost, pooled per store, Tweedie"}
COLOUR = {"seasonal_naive": plots.INK_MUTED, "moving_average_28": plots.INK_MUTED, "ets": plots.MODEL_COLOURS["ets"],
          "croston": plots.NAVY, "sba": "#3F6BB5", "tsb": "#7B52AB",
          "xgboost_rel@item_id": plots.MODEL_COLOURS["xgboost_rel@pooled"], "xgboost_tweedie@store_id": plots.MODEL_COLOURS["xgboost@pooled"]}
CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]

# --- figure 1: RMSSE by class, one dot per method -----------------------------
t = pd.read_csv("cache/sweep_by_class.csv")
# One method order for every panel, worst at the bottom, so the labels on the left apply across.
order = t.groupby("method")["rmsse"].mean().sort_values(ascending=False).index
fig, axes = plt.subplots(1, len(CLASSES), figsize=(7.4, 3.6), sharex=True)
for ax, cls in zip(axes, CLASSES, strict=True):
    g = t[t["demand_class"] == cls].set_index("method").reindex(order).reset_index().dropna(subset=["rmsse"])
    ax.set_yticks(range(len(order)), [NAME[m] for m in order] if ax is axes[0] else [""] * len(order), fontsize=7.5)
    if g.empty:
        ax.set_title(f"{cls}: no items", fontsize=9); continue
    y = np.array([list(order).index(m) for m in g["method"]])
    ax.hlines(y, 0, g["rmsse"], color=[COLOUR[m] for m in g["method"]], lw=1.2, alpha=0.6)
    ax.scatter(g["rmsse"], y, color=[COLOUR[m] for m in g["method"]], s=36, zorder=3)
    for yi, (_, r) in zip(y, g.iterrows(), strict=True):
        ax.text(r["rmsse"] + 0.01, yi, f"{r['rmsse']:.2f}", va="center", fontsize=7.5, color=plots.INK_SOFT)
    ax.set_title(f"{cls}, {int(g['items'].max())} items", fontsize=9); ax.grid(axis="y", visible=False)
    ax.axvline(1.0, color=plots.INK_MUTED, lw=1)
axes[0].set_xlim(0.3, 1.15)
fig.supxlabel("scaled error (RMSSE; 1.0 = 'this day last week' on the training history)", fontsize=9)
fig.suptitle("Every method by demand class, one department, 52 weekly origins, ten stores", fontsize=10.5)
fig.savefig(OUT / "1_rmsse_by_class.png", bbox_inches="tight"); plt.close(fig)

# --- figure 2: the order step, units over per week for the same service --------
o = pd.read_csv("cache/order_by_class.csv")
o = o[np.isclose(o["tau"], 0.95)]
o["rel"] = o["surplus"] / o["weekly_sales"]
order = o.groupby("method")["rel"].mean().sort_values(ascending=False).index
fig, axes = plt.subplots(1, len(CLASSES), figsize=(7.4, 3.6))
for ax, cls in zip(axes, CLASSES, strict=True):
    g = o[o["demand_class"] == cls].set_index("method").reindex(order).reset_index().dropna(subset=["rel"])
    ax.set_yticks(range(len(order)), [NAME[m] for m in order] if ax is axes[0] else [""] * len(order), fontsize=7.5)
    if g.empty:
        ax.set_title(f"{cls}: no items", fontsize=9); continue
    y = np.array([list(order).index(m) for m in g["method"]]); rel = g["rel"]
    ax.barh(y, rel, color=[COLOUR[m] for m in g["method"]], height=0.6)
    for yi, (v, s) in zip(y, zip(rel, g["short_weeks"], strict=True), strict=True):
        ax.text(v + 0.01, yi, f"{v:.2f}  ({s:.0%} short)", va="center", fontsize=7, color=plots.INK_SOFT)
    ax.set_title(f"{cls}, {g['weekly_sales'].iloc[0]:.0f} units/wk", fontsize=9); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(1.0, float(rel.max()) * 1.6))
fig.supxlabel("units left over per week at 95% service, as a share of the store's weekly sales", fontsize=9)
fig.suptitle("What each method's order costs in surplus for the same service level", fontsize=10.5)
fig.savefig(OUT / "2_surplus_by_class.png", bbox_inches="tight"); plt.close(fig)
print("wrote", OUT)
