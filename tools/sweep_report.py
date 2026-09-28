"""Results of a sweep by demand class: for every method, per class, the mean
RMSSE and bias, and the paired win rate and median gain against last week's
number. Reads the registry and the screen the department notebook wrote,
which carries each item's majority class. For every screened item it takes
the newest current-code run of each method and scope on the suite, whatever
its note, since the registry keeps one run per method and config and an
earlier note owns it if it ran first.

    PYTHONPATH=. python tools/sweep_report.py [--suite weekly] [--note "item 11"]

Prints a table per class and writes the same rows to
cache/sweep_by_class.csv for the findings. The pairing is on store-origin,
the same as the registry's own comparisons."""
import argparse
import pandas as pd
from src.registry import connect
from src.scoring import REFERENCE_BENCHMARK, improvement_over

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--note", default=None, help="only runs whose note starts with this")
p.add_argument("--suite", default="weekly")
p.add_argument("--screen", default="cache/dept_screen_FOODS_3.csv")
p.add_argument("--out", default="cache/sweep_by_class.csv")
p.add_argument("--methods", nargs="+", default=["seasonal_naive", "moving_average_28", "ets", "croston", "sba", "tsb",
                                                 "xgboost_rel@item_id", "xgboost_tweedie@store_id"],
               help="method labels to report, '@scope' for pooled runs; the sweep's set by default")
a = p.parse_args()

con = connect()
screen_df = pd.read_csv(a.screen)
runs = pd.read_sql_query(
    "SELECT run_id, method, pool_by, item_ids, note FROM run WHERE suite = ? AND code_current = 1 ORDER BY recorded_at",
    con, params=(a.suite,),
)
if a.note:
    runs = runs[runs["note"].fillna("").str.startswith(a.note)]
wanted = set(screen_df["item_id"]) | {",".join(sorted(screen_df["item_id"]))}  # per-item configs and the all-items pool
runs = runs[runs["item_ids"].isin(wanted)].drop_duplicates(["method", "pool_by", "item_ids"], keep="last")
scores = pd.read_sql_query(
    f"SELECT run_id, id, item_id, store_id, fold, origin, week_kind, rmsse, bias, unscored FROM fold_score "
    f"WHERE run_id IN ({','.join('?' * len(runs))})", con, params=tuple(runs["run_id"]),
)
con.close()
scores = scores.merge(runs[["run_id", "method", "pool_by"]], on="run_id")
scores["method"] = scores["method"] + scores["pool_by"].map(lambda v: f"@{v}" if isinstance(v, str) else "")
scores = scores[scores["method"].isin(a.methods)]
scores["kind"] = "model"  # improvement_over only needs the column to exist
cls = screen_df.set_index("item_id")["majority_class"]
scores["demand_class"] = scores["item_id"].map(cls)
scores = scores[scores["demand_class"].notna() & ~scores["unscored"].astype(bool)]

rows = []
for demand_class, g in scores.groupby("demand_class"):
    imp = improvement_over(g, REFERENCE_BENCHMARK).set_index("method")
    for method, m in g.groupby("method"):
        r = {"demand_class": demand_class, "method": method, "items": m["item_id"].nunique(),
             "store_weeks": len(m), "rmsse": m["rmsse"].mean(), "bias": m["bias"].mean()}
        if method in imp.index:
            r["weeks_won_vs_last_week"] = imp.loc[method, "win_rate"]
            r["median_gain_pct"] = imp.loc[method, "median_pct"]
        rows.append(r)
table = pd.DataFrame(rows).sort_values(["demand_class", "rmsse"])
table.to_csv(a.out, index=False)
for demand_class, g in table.groupby("demand_class", sort=False):
    print(f"\n## {demand_class}: {g['items'].max()} items")
    print(g.drop(columns="demand_class").round(3).to_string(index=False))
