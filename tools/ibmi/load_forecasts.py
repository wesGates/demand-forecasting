"""Push the forecast rows in cache/forecasts.sqlite into the FORECAST table
on the IBM i host, over ODBC. Rows already there are skipped, keyed on
(as_of, id, method, target_date), so this can run after every daily job.

    PYTHONPATH=. python tools/ibmi/load_forecasts.py [--db cache/forecasts.sqlite]

Needs IBM's "IBM i Access ODBC Driver" installed and four environment
variables: IBMI_HOST, IBMI_USER, IBMI_LIBRARY (the library holding the
table, created by the IBM i project) and IBMI_PASSWORD (set in the shell
session, never in a file). The table itself is created on the host from
tools/ibmi/ddl.py's output."""
import argparse, os, sqlite3, sys
import pandas as pd
from src import tables

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--db", default="cache/forecasts.sqlite")
a = p.parse_args()

env = {k: os.environ.get(k) for k in ("IBMI_HOST", "IBMI_USER", "IBMI_LIBRARY", "IBMI_PASSWORD")}
missing = [k for k, v in env.items() if not v]
if missing:
    sys.exit(f"set {', '.join(missing)} in the environment")
import pyodbc  # noqa: E402  - after the environment check, so the message about variables comes first

cols = tables.columns("forecast")
rows = pd.read_sql_query("SELECT * FROM forecast", sqlite3.connect(a.db))
# SQLite holds dates and timestamps as text; DB2 wants real types.
for c in ("as_of", "target_date"):
    rows[c] = pd.to_datetime(rows[c]).dt.date
rows["made_at"] = pd.to_datetime(rows["made_at"], utc=True).dt.tz_localize(None).dt.to_pydatetime()
rows["fallback"] = rows["fallback"].astype(int)

table = f"{env['IBMI_LIBRARY']}.FORECAST"
con = pyodbc.connect(
    f"DRIVER={{IBM i Access ODBC Driver}};SYSTEM={env['IBMI_HOST']};UID={env['IBMI_USER']};"
    # NAM=1: SQL naming, library.table. ExtendedDynamic=0: no SQL package in QGPL, which PUB400 forbids.
    # CommitMode=0 and autocommit: no commitment control. A user library on PUB400 has no journal,
    # and an insert under commitment control into an unjournaled table fails with SQL7008. The load is
    # keyed and skips rows already there, so it needs no transaction.
    f"PWD={env['IBMI_PASSWORD']};NAM=1;ExtendedDynamic=0;CommitMode=0",
    autocommit=True,
)
cur = con.cursor()
before = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
# A blocked insert (many rows per execute) is fast but DB2 for i allows it only for INSERT ... VALUES,
# so the rows go into a session temporary table in QTEMP first, and one set-based statement then
# copies across the ones whose key is absent. The database decides what is new, and the count comes
# from the table, not from what this script believed it sent.
cols_sql = ", ".join(cols)
cur.execute(f"DECLARE GLOBAL TEMPORARY TABLE SESSION.FORECAST_IN LIKE {table} WITH REPLACE")
cur.fast_executemany = True
params = [tuple(r) for r in rows[cols].itertuples(index=False, name=None)]
BATCH = 10_000  # the driver's blocked insert takes at most 32,767 rows per execute (SQL0221)
for start in range(0, len(params), BATCH):
    cur.executemany(f"INSERT INTO SESSION.FORECAST_IN ({cols_sql}) VALUES ({', '.join('?' * len(cols))})", params[start : start + BATCH])
match = " AND ".join(f"t.{k} = s.{k}" for k in tables.FORECAST_KEY)
cur.execute(
    f"INSERT INTO {table} ({cols_sql}) SELECT {cols_sql} FROM SESSION.FORECAST_IN s "
    f"WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {match})"
)
after = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
con.close()
print(f"{after - before} new forecast rows pushed to {table} on {env['IBMI_HOST']} ({len(rows) - (after - before)} already there)")
