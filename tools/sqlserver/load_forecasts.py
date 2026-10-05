"""
Copy the daily job's forecast rows (cache/forecasts.sqlite) into SQL
Server as host.forecast (the host-side tables sit in the `host` schema,
beside the registry's dbo tables), the same way load_registry.py copies
the registry. Rows already
there are skipped, keyed on (as_of, id, method, target_date), so the copy
can run after every daily job. Run from the repository root:

    PYTHONPATH=. python tools/sqlserver/load_forecasts.py

Connection from the environment as for load_registry.py.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from src import tables

sys.path.insert(0, str(Path(__file__).parent))
from load_registry import connect, create_database, load_table  # noqa: E402

FORECASTS = Path("cache/forecasts.sqlite")
COLUMNS = tables.types("forecast", "sqlserver")  # the one definition, src/tables.py


def main() -> int:
    database = os.environ.get("MSSQL_DATABASE", "forecasting")
    src = sqlite3.connect(FORECASTS)
    rows = pd.read_sql_query("SELECT * FROM forecast", src)
    rows["fallback"] = rows["fallback"].astype("Int64")
    rows["made_at"] = pd.to_datetime(rows["made_at"], utc=True).dt.tz_localize(None)
    for c in ("as_of", "target_date"):
        rows[c] = pd.to_datetime(rows[c]).dt.date
    create_database(database)
    with connect(database) as con:
        cur = con.cursor()
        cur.execute("IF SCHEMA_ID('host') IS NULL EXEC('CREATE SCHEMA host')")
        cur.execute("IF OBJECT_ID('host.forecast') IS NULL " + tables.ddl("forecast", "sqlserver", qualified="host.forecast"))
        cur.execute("SELECT as_of, id, method, target_date FROM host.forecast")
        have = {tuple(map(str, r)) for r in cur.fetchall()}
        new = rows[[tuple(map(str, k)) not in have for k in rows[["as_of", "id", "method", "target_date"]].itertuples(index=False, name=None)]]
        n = load_table(con, "host.forecast", new, COLUMNS) if len(new) else 0
        print(f"{n} new forecast rows loaded into {database}.host.forecast ({len(rows) - n} already there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
