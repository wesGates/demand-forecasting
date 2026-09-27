"""The first study's per-store scaled error for every method, as reported, for
report figure 5. Run inside a checkout of tag `first-study` (with `data`
linked), from its root with PYTHONPATH=. ; the fits take about 20 min. Writes
first_study_scores.csv there; copy it to tools/report/."""
from src.step1_problem import Config, STUDY_ITEMS
from src.step2_data import load_panel
from src.step3_explore import series_stats
from src.step5_evaluate import run_walk_forward, score_folds

cfg = Config(item_ids=STUDY_ITEMS)
df = load_panel(cfg, verbose=False)
stats = series_stats(df, cfg)
scores = score_folds(run_walk_forward(df, cfg, progress=False))
t = scores.pivot_table(index="store_id", columns="method", values="rmsse", aggfunc="mean")
t = t.reindex(stats.sort_values("mean_sales", ascending=False)["store_id"])  # busiest store first
t.to_csv("first_study_scores.csv")
print(t.round(3).to_string())
