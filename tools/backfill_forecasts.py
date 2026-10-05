"""Fill cache/forecasts.sqlite from cached backtest runs instead of rerunning
the daily job once per origin. The rows are what the daily job would write
for each origin, because forecast_as_of and the backtest share one code
path (tests/test_forecast.py holds them equal).

    PYTHONPATH=. python tools/backfill_forecasts.py --suite everyday --item fast --pool item_id --methods xgboost_rel
    PYTHONPATH=. python tools/backfill_forecasts.py --suite everyday --item fast --methods ets

Rows already present are left alone, like the daily job's own writes."""
import argparse, sqlite3
from datetime import UTC, datetime
from pathlib import Path
import pandas as pd
from src import tables
from src.step5_evaluate import _method_code, _read_cached
from src.suites import SUITES, suite_config

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--suite", required=True, choices=sorted(SUITES))
p.add_argument("--item", default="fast")
p.add_argument("--pool", default=None)
p.add_argument("--methods", nargs="+", required=True)
p.add_argument("--db", default="cache/forecasts.sqlite")
a = p.parse_args()

cfg = suite_config(a.suite, a.item, pool_by=a.pool)
label = "@" + a.pool if a.pool else ""
Path(a.db).parent.mkdir(exist_ok=True)
con = sqlite3.connect(a.db)
con.execute(tables.ddl("forecast", "sqlite"))
cols = tables.columns("forecast")
for method in a.methods:
    rows = _read_cached(cfg, method)
    if rows is None:
        print(f"{method}: not cached for this config; run it first"); continue
    out = pd.DataFrame({
        "as_of": pd.to_datetime(rows["origin"]).dt.date.astype(str),
        "id": rows["id"], "item_id": rows["item_id"], "store_id": rows["store_id"],
        "method": method + label, "horizon": rows["horizon"],
        "target_date": pd.to_datetime(rows["target_date"]).dt.date.astype(str),
        "forecast": rows["forecast"].astype(float), "fallback": rows["fallback"].astype(int),
        "code_digest": _method_code(method), "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    before = con.execute("SELECT COUNT(*) FROM forecast").fetchone()[0]
    con.executemany(f"INSERT OR IGNORE INTO forecast VALUES ({','.join('?' * len(cols))})", out[cols].itertuples(index=False, name=None))
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM forecast").fetchone()[0]
    print(f"{method}{label}: {len(out)} rows, {after - before} new ({rows['origin'].nunique()} origins)")
con.close()
