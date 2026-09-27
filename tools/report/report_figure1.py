"""Figure 1 of the follow-on report: the first study's error by store, with
plain labels. Run from the first-study checkout (../demand-forecasting-first-study) with its .venv (the cached
predictions are there). Writes into the follow-on repo's report/figures/."""
from pathlib import Path
import matplotlib.pyplot as plt
from src import plots
from src.step1_problem import Config, STUDY_ITEMS
from src.step2_data import load_panel
from src.step3_explore import series_stats
from src.step5_evaluate import run_walk_forward, score_folds

OUT = Path("../demand-forecasting/report/figures/5_first_study_by_store.png")
PLAIN = {"xgboost": "XGBoost", "ets": "Exponential smoothing (ETS)", "arima": "ARIMA",
         "seasonal_naive": "This day last week", "moving_average_28": "28-day moving average",
         "seasonal_naive_364": "This day last year", "mean": "Long-run average",
         "naive": "Yesterday's sales", "drift": "Yesterday plus trend"}
plots.use_style(); plt.rcParams["savefig.dpi"] = 300  # crisp in the PDF
cfg = Config(item_ids=STUDY_ITEMS)
df = load_panel(cfg, verbose=False)
stats = series_stats(df, cfg)
scores = score_folds(run_walk_forward(df, cfg, progress=False))
scores["method"] = scores["method"].map(PLAIN)
plots.MODEL_COLOURS.update({PLAIN[k]: v for k, v in list(plots.MODEL_COLOURS.items())})
plots.BENCH_LINES.update({PLAIN[k]: v for k, v in list(plots.BENCH_LINES.items())})
plots.BENCH_MARKERS.update({PLAIN[k]: v for k, v in list(plots.BENCH_MARKERS.items())})
order = [PLAIN[k] for k in ("xgboost", "ets", "arima", "seasonal_naive", "moving_average_28",
                            "seasonal_naive_364", "mean", "naive", "drift")]
fig, ax = plt.subplots(figsize=(10, 3.8))
plots.plot_rmsse_by_store(scores, stats, item_id=STUDY_ITEMS[0], methods=order, ax=ax)
ax.set_title("Scaled error by store in the first study, busiest store on the left", fontsize=10.5)
ax.set_ylabel("scaled error (RMSSE; 1.0 = 'this day last week'\non the training history)")
fig.savefig(OUT, bbox_inches="tight"); plt.close(fig)
print("wrote", OUT.resolve())

# Per-store check for the report's "beats the benchmark at every store" claim.
t = scores.pivot_table(index="store_id", columns="method", values="rmsse", aggfunc="mean")
t = t.reindex(stats.sort_values("mean_sales", ascending=False)["store_id"])
cols = [PLAIN[k] for k in ("xgboost", "arima", "ets", "seasonal_naive", "moving_average_28")]
print(t[cols].round(3).to_string())
