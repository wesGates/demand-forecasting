"""The report's data figures: the demand classification of the twenty series
(figure 3) and what the fast mover's history shows, the weekly pattern and
the calendar events (figure 4). Plain labels for a non-technical reader.
Run from the repository root with PYTHONPATH=. ; argv[1] is the output folder."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import plots
from src.step1_problem import Config
from src.step2_data import load_panel, MAJOR_EVENTS
from src.step3_explore import series_stats, classification_cutoff, event_effects

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("report/figures"); OUT.mkdir(parents=True, exist_ok=True)
plots.use_style(); plt.rcParams["savefig.dpi"] = 300
ITEMS = ("FOODS_3_586", "FOODS_1_021")
NAME = {"FOODS_3_586": "fast mover", "FOODS_1_021": "slow mover"}
# Items get black and grey, since the hues are kept for models.
STYLE = {"FOODS_3_586": dict(color=plots.INK, marker="o"), "FOODS_1_021": dict(color=plots.INK_MUTED, marker="s")}
cfg = Config(item_ids=ITEMS)
df = load_panel(cfg, verbose=False)
stats = series_stats(df, cfg)
print(stats[["item_id", "store_id", "adi", "cv2", "demand_class"]].to_string(index=False))

# --- figure 3: the demand classification, one point per store and item ---
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for item in ITEMS:
    g = stats[stats["item_id"] == item]
    ax.scatter(g["adi"], g["cv2"], s=48, **STYLE[item], label=f"{NAME[item]}, one point per store", zorder=3)
ax.axvline(1.32, color=plots.INK_MUTED, lw=1); ax.axhline(0.49, color=plots.INK_MUTED, lw=1)
ax.set_xlim(0.95, 3.0); ax.set_ylim(0, 1.0)
ax.text(0.97, 0.45, "smooth", va="top", fontsize=9, color=plots.INK_SOFT); ax.text(1.35, 0.45, "intermittent", va="top", fontsize=9, color=plots.INK_SOFT)
ax.text(0.97, 0.97, "erratic", va="top", fontsize=9, color=plots.INK_SOFT); ax.text(1.35, 0.97, "lumpy", va="top", fontsize=9, color=plots.INK_SOFT)
ax.set_xlabel("average days between sales (ADI); 1.0 = sells every day")
ax.set_ylabel("variability of the amounts sold (CV²)")
ax.set_title("Demand class of each series, from the history before the test year", fontsize=10.5)
plots.legend_below(ax, ncols=2)
fig.savefig(OUT / "3_classification.png", bbox_inches="tight"); plt.close(fig)

# --- figure 4: the weekly pattern by store, and which calendar events move sales ---
item = "FOODS_3_586"
d = df[df["item_id"] == item].copy()
d["weekday"] = d["date"].dt.dayofweek
prof = d.groupby(["store_id", "weekday"])["sales"].mean().unstack("weekday")
prof = prof.div(prof.mean(axis=1), axis=0)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 7.8), gridspec_kw=dict(height_ratios=[1, 1.45]))
for store, row in prof.iterrows():
    ax1.plot(range(7), row.values, color=plots.INK_MUTED, lw=1, alpha=0.6)
ax1.plot(range(7), prof.mean().values, color=plots.INK, lw=2.4, label="average of the ten stores")
ax1.axhline(1.0, color=plots.AXIS, lw=1)
ax1.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]); ax1.set_ylabel("sales relative to the store's own average")
ax1.set_title("The weekly pattern, each store as a grey line", fontsize=10); ax1.legend(loc="upper left", frameon=False, fontsize=9)
calendar = pd.read_csv(cfg.data_dir / "calendar.csv", parse_dates=["date"])
cutoff = classification_cutoff(df, cfg)
plots.plot_event_effects(event_effects(d, calendar, cutoff), major=MAJOR_EVENTS, ax=ax2)
ax2.set_title("Calendar events, sales on the day and in the run-up, as a multiple of a normal day", fontsize=10)
fig.suptitle("What the fast mover's history shows", fontsize=11)
fig.savefig(OUT / "4_data_patterns.png", bbox_inches="tight"); plt.close(fig)
print("wrote", OUT / "3_classification.png", "and", OUT / "4_data_patterns.png")
