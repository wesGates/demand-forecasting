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
    f"PWD={env['IBMI_PASSWORD']};NAM=1;ExtendedDynamic=0",
    autocommit=False,
)
cur = con.cursor()
have = {(str(r[0]), r[1], r[2], str(r[3])) for r in cur.execute(f"SELECT as_of, id, method, target_date FROM {table}")}
keep = [k not in have for k in zip(rows["as_of"].astype(str), rows["id"], rows["method"], rows["target_date"].astype(str), strict=True)]
new = rows[keep]
if len(new):
    cur.fast_executemany = True
    cur.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [tuple(r) for r in new[cols].itertuples(index=False, name=None)],
    )
con.commit()
con.close()
print(f"{len(new)} new forecast rows pushed to {table} on {env['IBMI_HOST']} ({len(rows) - len(new)} already there)")
