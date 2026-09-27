"""Figures for checking items 3-5 from the cached every-day runs. No fitting.
Run from the repo root: PYTHONPATH=. python tools/render_items.py
Writes to figures/items345/ (gitignored)."""
import warnings; warnings.simplefilter("ignore")
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src import plots
from src.step1_problem import Config, STUDY_ITEMS
from src.step2_data import load_panel
from src.step3_explore import series_stats, classification_cutoff
from src.step5_evaluate import _read_cached
from src.scoring import score_folds

OUT = Path("figures/items345"); OUT.mkdir(parents=True, exist_ok=True)
plots.use_style()
plots.MODEL_COLOURS.update({"xgboost_pooled": plots.MODEL_COLOURS["xgboost@pooled"],
                            "xgboost_rel_pooled": plots.MODEL_COLOURS["xgboost_rel@pooled"]})
LAYOUT = dict(fold_step=1, n_folds=358)
FIVE = ["xgboost", "xgboost_pooled", "xgboost_rel_pooled", "arima", "ets"]
written = []

def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig); written.append(name)

for item in (STUDY_ITEMS, ("FOODS_1_021",)):
    tag = item[0]
    frames = []
    for pool in (None, "item_id"):
        cfg = Config(item_ids=item, pool_by=pool, **LAYOUT)
        methods = ["xgboost", "xgboost_rel"] + ([] if pool else ["ets", "arima", "moving_average_28", "seasonal_naive"])
        for m in methods:
            f = _read_cached(cfg, m); assert f is not None, (tag, pool, m)
            if pool: f = f.assign(method=f["method"] + "_pooled")
            frames.append(f)
    pred = pd.concat(frames, ignore_index=True)
    cfg = Config(item_ids=item, **LAYOUT)
    df = load_panel(cfg, verbose=False)
    stats = series_stats(df, cfg); cutoff = classification_cutoff(df, cfg)
    origins = cfg.fold_origins(df["date"].max())
    top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]
    scores = score_folds(pred)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    plots.plot_rmsse_by_store(scores, stats, item_id=tag, methods=FIVE + ["moving_average_28", "seasonal_naive"], ax=ax)
    ax.set_title(f"{tag} — mean RMSSE by store, every-day layout")
    save(fig, f"{tag}_1_rmsse_by_store.png")

    fig, ax = plt.subplots(figsize=(10, 3.4))
    plots.plot_bias_by_store(scores, stats, item_id=tag, methods=FIVE, ax=ax)
    ax.set_title(f"{tag} — bias by store (forecast minus actual, units/day)")
    save(fig, f"{tag}_2_bias_by_store.png")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    plots.plot_rmsse_by_horizon(pred, methods=FIVE + ["moving_average_28", "seasonal_naive"], ax=ax)
    ax.set_title(f"{tag} — RMSSE by days ahead")
    save(fig, f"{tag}_3_rmsse_by_horizon.png")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    plots.plot_forecast_folds(pred, df, top_id, origins=origins, window=("2015-11-09", "2016-01-03"),
                              methods=["xgboost_rel_pooled", "arima", "seasonal_naive"], ax=ax)
    ax.set_title(f"{tag} at {top_store} — one-step forecasts, Thanksgiving to New Year")
    save(fig, f"{tag}_4_forecasts_holidays_{top_store}.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    plots.plot_residual_diagnostics(pred, top_id, "xgboost_rel_pooled", season=cfg.season, axes=axes)
    fig.suptitle(f"xgboost_rel_pooled at {top_store} — one-step residual diagnostics", fontsize=11)
    save(fig, f"{tag}_5_residuals_rel_pooled_{top_store}.png")

    # monthly level: actual against each XGBoost variant and the moving average, all stores
    p = pred[pred["method"].isin(["xgboost", "xgboost_pooled", "xgboost_rel_pooled", "moving_average_28"])].copy()
    p = p[~p.get("closure", False).astype(bool)] if "closure" in p else p
    p["month"] = pd.to_datetime(p["target_date"] if "target_date" in p else p["date"]).dt.to_period("M")
    actual = p[p["method"] == "xgboost"].groupby("month")["actual"].mean()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(actual.index.to_timestamp(), actual.values, color=plots.INK, lw=2.2, label="actual")
    for m in ["xgboost", "xgboost_pooled", "xgboost_rel_pooled", "moving_average_28"]:
        s = p[p["method"] == m].groupby("month")["forecast"].mean()
        st = plots._method_style(m, "benchmark" if m == "moving_average_28" else "model")
        ax.plot(s.index.to_timestamp(), s.values, label=m, **st)
    ax.set_ylabel("units per day, mean over stores"); ax.set_title(f"{tag} — monthly level: actual vs forecast")
    plots.legend_below(ax)
    save(fig, f"{tag}_6_monthly_level.png")

print("wrote", len(written)); print("\n".join(written))
