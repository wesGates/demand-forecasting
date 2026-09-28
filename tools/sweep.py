"""Run a list of methods on every item that passed the department screen and
register each run. A loop around `record()`, nothing more.

    PYTHONPATH=. python tools/sweep.py --suite weekly --methods croston sba tsb ets \\
        seasonal_naive moving_average_28 --note "item 11 sweep"
    PYTHONPATH=. python tools/sweep.py --suite weekly --pool store_id --methods xgboost_tweedie \\
        --note "item 11 sweep, pooled per store"

Without --pool, one config per item (the harness fits per series, or per item
when --pool item_id). With --pool store_id, one config holds every item and the
model is fitted once per store across them, the way the M5 winners grouped
their data. Items come from the screen the department notebook wrote
(cache/dept_screen_FOODS_3.csv); --limit N takes the first N for a smoke test.
Cached methods load instantly and are registered under the note anyway."""
import argparse, sys, time
import pandas as pd
from src.registry import record
from src.suites import SUITES, suite_config
from src.step1_problem import Config

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--suite", required=True, choices=sorted(SUITES))
p.add_argument("--methods", nargs="+", required=True)
p.add_argument("--pool", default=None, help="item_id: one model per item across stores; store_id: one per store across items")
p.add_argument("--screen", default="cache/dept_screen_FOODS_3.csv")
p.add_argument("--limit", type=int, default=None)
p.add_argument("--note", required=True)
a = p.parse_args()

items = pd.read_csv(a.screen)["item_id"].tolist()[: a.limit]
t0 = time.time()
if a.pool == "store_id":
    cfg = Config(item_ids=tuple(items), pool_by="store_id", **SUITES[a.suite])
    ids = record(cfg, a.methods, note=a.note, progress=False)
    print(f"{len(items)} items in one pool per store: {', '.join(ids)}  {time.time() - t0:.0f} s", flush=True)
else:
    for i, item in enumerate(items, 1):
        t = time.time()
        ids = record(suite_config(a.suite, item, pool_by=a.pool), a.methods, note=a.note, progress=False)
        print(f"{i}/{len(items)} {item}: {len(ids)} runs, {time.time() - t:.0f} s (total {(time.time() - t0) / 60:.0f} min)", flush=True)
print("done", flush=True)
sys.exit(0)
