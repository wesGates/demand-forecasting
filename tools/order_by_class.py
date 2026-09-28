"""The order step across the department: every sweep method's weekly order at
a service level, scored by demand class on what an orderer feels, the share
of weeks that ran short, the units short in those weeks, and the units left
over in the others.

    PYTHONPATH=. python tools/order_by_class.py [--taus 0.9 0.95] [--burn-in 13]

Reads the sweep's cached forecasts (the same item list and methods as
tools/sweep.py) and uses src/order.py as notebook 03 does, with one change:
the calibration window grows through the scored year instead of resting on
a separate first year. Each week's order is the point forecast plus the
tau-quantile of that method's own weekly errors on every earlier week, so
the first --burn-in weeks only calibrate and are not scored. That is what a
live system would do with one year of history. Writes cache/order_by_class.csv."""
import argparse
import pandas as pd
from src.order import calibrate_expanding, quantile_forecasts, score_quantiles, summarise_quantiles, weekly_totals
from src.step1_problem import Config
from src.step4_models import BENCHMARKS
from src.step5_evaluate import _read_cached
from src.suites import SUITES, suite_config

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--suite", default="weekly")
p.add_argument("--screen", default="cache/dept_screen_FOODS_3.csv")
p.add_argument("--methods", nargs="+", default=["seasonal_naive", "moving_average_28", "ets", "croston", "sba", "tsb"])
p.add_argument("--pooled-per-item", nargs="+", default=["xgboost_rel"], help="methods cached with pool_by=item_id")
p.add_argument("--pooled-per-store", nargs="+", default=["xgboost_tweedie"], help="methods cached with pool_by=store_id, all items in one config")
p.add_argument("--taus", nargs="+", type=float, default=[0.9, 0.95])
p.add_argument("--burn-in", type=int, default=13, help="weeks that only calibrate")
p.add_argument("--limit", type=int, default=None)
p.add_argument("--out", default="cache/order_by_class.csv")
a = p.parse_args()

screen = pd.read_csv(a.screen).set_index("item_id")
items = screen.index.tolist()[: a.limit]
frames = []
def take(cfg, method, label):
    f = _read_cached(cfg, method)
    if f is None:
        print("not cached:", label, method, flush=True); return
    frames.append(f.assign(method=label, kind="benchmark" if method in BENCHMARKS else "model"))
for item in items:
    for m in a.methods:
        take(suite_config(a.suite, item), m, m)
    for m in a.pooled_per_item:
        take(suite_config(a.suite, item, pool_by="item_id"), m, f"{m}@item_id")
for m in a.pooled_per_store:
    take(Config(item_ids=tuple(items), pool_by="store_id", **SUITES[a.suite]), m, f"{m}@store_id")
pred = pd.concat(frames, ignore_index=True)
pred = pred[pred["item_id"].isin(items)]

weekly = weekly_totals(pred)
scored_from = weekly.groupby("fold")["origin"].first().sort_index().iloc[a.burn_in]
offsets = calibrate_expanding(weekly, scored_from, taus=a.taus)
scored = score_quantiles(quantile_forecasts(weekly, offsets, scored_from))
scored["demand_class"] = scored["item_id"].map(screen["majority_class"])
scored["weekly_sales"] = scored.groupby("id", observed=True)["actual"].transform("mean")

rows = []
for demand_class, g in scored.groupby("demand_class"):
    s = summarise_quantiles(g)
    s.insert(0, "demand_class", demand_class)
    s["items"] = g["item_id"].nunique()
    s["weekly_sales"] = g["weekly_sales"].mean()
    rows.append(s)
table = pd.concat(rows, ignore_index=True)
table["short_weeks"] = 1 - table["coverage"]
table = table[["demand_class", "method", "tau", "items", "weekly_sales", "short_weeks", "shortfall", "surplus", "pinball_rel", "n_weeks"]]
table.to_csv(a.out, index=False)
print(f"scored weeks from {scored_from.date()} (folds after a {a.burn_in}-week burn-in)")
for (demand_class, tau), g in table.groupby(["demand_class", "tau"]):
    print(f"\n## {demand_class}, service level {tau:.0%}: {g['items'].iloc[0]} items, {g['weekly_sales'].iloc[0]:.1f} units a week per store")
    print(g.drop(columns=["demand_class", "tau", "items", "weekly_sales"]).sort_values("pinball_rel").round(3).to_string(index=False))
