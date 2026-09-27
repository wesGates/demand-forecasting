"""Report figures 2 and 3 with plain labels for a non-technical reader.
Run from the fork root with PYTHONPATH=. ; writes report/figures/2_final_results.png
and report/figures/6_progression.png."""
import sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import plots

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("report/figures"); OUT.mkdir(parents=True, exist_ok=True)
plots.use_style(); plt.rcParams["savefig.dpi"] = 300  # crisp in the PDF
con = sqlite3.connect("cache/registry.sqlite")
runs = pd.read_sql_query("SELECT * FROM run WHERE code_current = 1 AND suite = 'everyday'", con)
con.close()
NAME = {"xgboost": "XGBoost, one model per store", "xgboost_rel": "XGBoost per store, level-relative",
        "arima": "ARIMA (seasonal, holiday effects)", "ets": "Exponential smoothing (ETS)",
        "arima_plain": "ARIMA, no holiday or SNAP inputs", "arma": "ARMA, no seasonality, no inputs",
        "moving_average_28": "28-day moving average", "seasonal_naive": "This day last week",
        "seasonal_naive_364": "This day last year", "mean": "Long-run average", "naive": "Yesterday's sales",
        "drift": "Yesterday plus trend"}
def label(r):
    if isinstance(r["pool_by"], str):
        return "XGBoost, pooled across stores" + (", level-relative" if r["method"] == "xgboost_rel" else "")
    return NAME[r["method"]]
runs["label"] = runs.apply(label, axis=1)
runs = runs.sort_values("recorded_at").drop_duplicates(["item_ids", "label"], keep="last")  # one row per method
ITEM = {"FOODS_3_586": "Fast mover (14–103 units a day)", "FOODS_1_021": "Slow mover in decline (0.6–7 units a day)"}

# --- figure 2: every method on both items ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
for ax, item in zip(axes, ("FOODS_3_586", "FOODS_1_021")):
    g = runs[runs["item_ids"] == item].sort_values("rmsse_mean", ascending=False)
    y = np.arange(len(g))
    col = [plots.SERIES_5 if "level-relative" in l and "pooled" in l else plots.SERIES_1 if "XGBoost" in l
           else plots.SERIES_3 if l.startswith("ARIMA (") else plots.SERIES_4 if l.startswith("ARIMA,") or l.startswith("ARMA") else plots.SERIES_2 if l.startswith("Exponential") else plots.INK_SOFT for l in g["label"]]
    ax.hlines(y, 0, g["rmsse_mean"], color=col, lw=1.2, alpha=0.6); ax.scatter(g["rmsse_mean"], y, color=col, s=44, zorder=3)
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.text(r["rmsse_mean"] + 0.012, yi, f"{r['rmsse_mean']:.2f}", va="center", fontsize=8.5, color=plots.INK_SOFT)
    ax.set_yticks(y, g["label"], fontsize=8.5); ax.axvline(1.0, color=plots.INK_MUTED, lw=1)
    ax.set_xlim(0.3, 1.2); ax.set_xlabel("scaled error (RMSSE; 1.0 = 'this day last week' on the training history)", fontsize=9)
    ax.set_title(ITEM[item], fontsize=10); ax.grid(axis="y", visible=False)
fig.suptitle("Every method on both items, ten stores, every day of one year as a starting point", fontsize=11)
fig.savefig(OUT / "2_final_results.png", bbox_inches="tight"); plt.close(fig)

# --- figure 3: the learned model, stage by stage ------------------------------
stages = [("per store", ("xgboost", None)), ("pooled", ("xgboost", "item_id")), ("pooled +\nlevel-rel.", ("xgboost_rel", "item_id"))]
def pick(item, method, pool, col):
    g = runs[(runs["item_ids"] == item) & (runs["method"] == method) & (runs["pool_by"].isna() if pool is None else runs["pool_by"] == pool)]
    return float(g.iloc[0][col])
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
panels = (("rmsse_mean", "scaled error\n(lower is better)"), ("win_vs_ref", "share of weeks beating\n'this day last week'"), ("bias_mean", "bias, units a day\n(0 = unbiased)"))
for ax, (metric, title) in zip(axes, panels):
    for item, color, marker, short in (("FOODS_3_586", plots.SERIES_1, "o", "fast mover"), ("FOODS_1_021", plots.SERIES_5, "s", "slow mover")):
        ys = [pick(item, m, p, metric) for _, (m, p) in stages]
        ax.plot(range(3), ys, marker=marker, color=color, lw=2, label=f"XGBoost, {short}")
        for ref, name, ls in (("arima", "ARIMA", "-"), ("arma", "ARMA", "--"), ("moving_average_28", "28-day average", ":")):
            v = pick(item, ref, None, metric)
            if metric == "win_vs_ref" and np.isnan(v): continue
            ax.hlines(v, -0.15, 2.15, color=color, lw=1, ls=ls, alpha=0.7, label=f"{name}, {short}")
    ax.set_xticks(range(3), [s for s, _ in stages], fontsize=8.5); ax.set_title(title, fontsize=9.5)
    if metric == "bias_mean": ax.axhline(0, color=plots.INK, lw=1)
fig.suptitle("The machine learning model at each stage, against ARIMA, ARMA and the simple average", fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=8.5)
fig.savefig(OUT / "6_progression.png", bbox_inches="tight"); plt.close(fig)
print("figures 2 and 3 written")

# --- figures 4 and 5 with plain names -----------------------------------------
from src.step1_problem import Config, STUDY_ITEMS
from src.step2_data import load_panel
from src.step3_explore import series_stats, classification_cutoff
from src.step5_evaluate import _read_cached
PLAIN = {"xgboost": "XGBoost, one model per store", "xgboost_pooled": "XGBoost, pooled", "xgboost_rel_pooled": "XGBoost, pooled, level-relative (final)",
         "moving_average_28": "28-day moving average", "arima": "ARIMA", "arma": "ARMA", "seasonal_naive": "This day last week"}
plots.MODEL_COLOURS.update({PLAIN["xgboost"]: plots.SERIES_1, PLAIN["xgboost_pooled"]: plots.SERIES_4, PLAIN["xgboost_rel_pooled"]: plots.SERIES_5, PLAIN["arima"]: plots.SERIES_3, PLAIN["arma"]: plots.SERIES_2})
plots.BENCH_LINES.update({PLAIN["moving_average_28"]: ":", PLAIN["seasonal_naive"]: "--"})
plots.BENCH_MARKERS.update({PLAIN["moving_average_28"]: "D", PLAIN["seasonal_naive"]: "s"})
def frames_for(item):
    out = []
    for pool in (None, "item_id"):
        cfg = Config(item_ids=item, pool_by=pool, fold_step=1, n_folds=358)
        for m in (["xgboost", "xgboost_rel"] + ([] if pool else ["arima", "arma", "moving_average_28", "seasonal_naive"])):
            f = _read_cached(cfg, m); assert f is not None, (item, pool, m)
            name = m + ("_pooled" if pool else "")
            if name in PLAIN:
                out.append(f.assign(method=PLAIN[name]))
    return pd.concat(out, ignore_index=True)

# figure 4: the declining item's monthly level
p = frames_for(("FOODS_1_021",))
p = p[~p["closure"].astype(bool)]
p["month"] = pd.to_datetime(p["target_date"]).dt.to_period("M")
actual = p[p["method"] == PLAIN["xgboost"]].groupby("month")["actual"].mean()
fig, ax = plt.subplots(figsize=(9.5, 3.4))
ax.plot(actual.index.to_timestamp(), actual.values, color=plots.INK, lw=2.2, label="what actually sold")
for key in ("xgboost", "xgboost_pooled", "xgboost_rel_pooled", "moving_average_28"):
    s_ = p[p["method"] == PLAIN[key]].groupby("month")["forecast"].mean()
    st = plots._method_style(PLAIN[key], "benchmark" if key == "moving_average_28" else "model")
    ax.plot(s_.index.to_timestamp(), s_.values, label=PLAIN[key], **st)
ax.set_ylabel("units a day, average over ten stores"); ax.set_title("Monthly average of forecasts against sales for the slow mover", fontsize=10.5)
plots.legend_below(ax, ncols=3)
fig.savefig(OUT / "7_declining_item.png", bbox_inches="tight"); plt.close(fig)

# figure 5: holiday weeks at the busiest store
cfg = Config(item_ids=STUDY_ITEMS, fold_step=1, n_folds=358); df = load_panel(cfg, verbose=False)
stats = series_stats(df, cfg); cutoff = classification_cutoff(df, cfg); origins = cfg.fold_origins(df["date"].max())
top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]
p = frames_for(STUDY_ITEMS)
fig, ax = plt.subplots(figsize=(11, 3.6))
plots.plot_forecast_folds(p, df, top_id, origins=origins, window=("2015-11-09", "2016-01-03"),
                          methods=[PLAIN["xgboost_rel_pooled"], PLAIN["arima"], PLAIN["arma"], PLAIN["seasonal_naive"]], ax=ax)
ax.set_title("Next-day forecasts against sales at the fast mover's busiest store, Thanksgiving to New Year", fontsize=10.5)
fig.savefig(OUT / "8_holiday_weeks.png", bbox_inches="tight"); plt.close(fig)
print("figures 4 and 5 written")
