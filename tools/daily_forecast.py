"""The daily job, in replay: forecast the next seven days from one date and
write the rows to a table beside the registry. Running the same date twice
leaves the table unchanged.

    PYTHONPATH=. python tools/daily_forecast.py --as-of 2016-05-01 --item fast --pool item_id --methods xgboost_rel ets
    PYTHONPATH=. python tools/daily_forecast.py --as-of 2016-05-01 --item fast --methods croston sba tsb

The forecast is made by src/forecast.forecast_as_of, which is the backtest's
own code path for one origin, so a row written here equals the backtest's
row for the same day. The table is cache/forecasts.sqlite, one row per
series, method and target day, keyed on (as_of, id, method, target_date).
tools/sqlserver/load_forecasts.py copies it to SQL Server."""
import argparse, sqlite3
from datetime import UTC, datetime
from pathlib import Path
import pandas as pd
from src import tables
from src.forecast import forecast_as_of
from src.step2_data import load_panel
from src.step5_evaluate import _method_code
from src.suites import suite_config

TABLE = tables.ddl("forecast", "sqlite")  # the one definition, src/tables.py

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--as-of", required=True, help="the origin date, YYYY-MM-DD")
p.add_argument("--item", default="fast", help="fast/slow or an M5 item id")
p.add_argument("--pool", default=None)
p.add_argument("--methods", nargs="+", required=True)
p.add_argument("--db", default="cache/forecasts.sqlite")
a = p.parse_args()

# The every-day layout, so any day of the test year is an origin.
cfg = suite_config("everyday", a.item, pool_by=a.pool)
df = load_panel(cfg, verbose=False)
rows = forecast_as_of(df, cfg, a.methods, a.as_of)
if rows.empty:
    raise SystemExit(f"no series has enough history and seven days of data after {a.as_of}")
label = "@" + a.pool if a.pool else ""
out = pd.DataFrame({
    "as_of": pd.Timestamp(a.as_of).date().isoformat(),
    "id": rows["id"], "item_id": rows["item_id"], "store_id": rows["store_id"],
    "method": rows["method"] + label, "horizon": rows["horizon"],
    "target_date": pd.to_datetime(rows["target_date"]).dt.date.astype(str),
    "forecast": rows["forecast"].astype(float), "fallback": rows["fallback"].astype(int),
    "code_digest": rows["method"].map(_method_code),
    "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
})
Path(a.db).parent.mkdir(exist_ok=True)
con = sqlite3.connect(a.db)
con.execute(TABLE)
before = con.execute("SELECT COUNT(*) FROM forecast").fetchone()[0]
con.executemany(  # OR IGNORE: a rerun of the same date writes nothing
    f"INSERT OR IGNORE INTO forecast VALUES ({','.join('?' * len(tables.columns('forecast')))})",
    out[tables.columns("forecast")].itertuples(index=False, name=None),
)
con.commit()
after = con.execute("SELECT COUNT(*) FROM forecast").fetchone()[0]
con.close()
print(f"{len(out)} forecast rows for {a.as_of}, {after - before} new in {a.db} ({after} total)")
