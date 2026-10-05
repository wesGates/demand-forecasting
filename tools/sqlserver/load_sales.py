"""Load daily sales for a list of items at every store into SQL Server, as
host.sales, straight from the raw M5 files (no closure imputation, so the
rows match what the IBM i host loads from the same files). Rows already
there are skipped, so more items can be added later.

    PYTHONPATH=. python tools/sqlserver/load_sales.py --items fast
    PYTHONPATH=. python tools/sqlserver/load_sales.py --items FOODS_3_586 FOODS_1_021

Connection from the environment as for load_registry.py."""
import argparse, os, sys
from pathlib import Path
import pandas as pd
from src import tables
from src.step1_problem import Config
from src.suites import KNOWN_ITEMS

sys.path.insert(0, str(Path(__file__).parent))
from load_registry import connect, create_database, load_table  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--items", nargs="+", required=True, help="fast/slow or M5 item ids")
a = p.parse_args()
items = [KNOWN_ITEMS.get(i, i) for i in a.items]

data = Config().data_dir
wide = pd.read_csv(data / "sales_train_evaluation.csv")
wide = wide[wide["item_id"].isin(items)]
if wide.empty:
    sys.exit(f"no series for {items}")
days = pd.read_csv(data / "calendar.csv", usecols=["d", "date"], parse_dates=["date"]).set_index("d")["date"]
long = wide.melt(id_vars=["id", "item_id", "store_id"], value_vars=[c for c in wide if c.startswith("d_")], var_name="d", value_name="units")
long["date"] = long["d"].map(days).dt.date
long = long[tables.columns("sales")].sort_values(["id", "date"])

database = os.environ.get("MSSQL_DATABASE", "forecasting")
create_database(database)
with connect(database) as con:
    cur = con.cursor()
    cur.execute("IF SCHEMA_ID('host') IS NULL EXEC('CREATE SCHEMA host')")
    cur.execute("IF OBJECT_ID('host.sales') IS NULL " + tables.ddl("sales", "sqlserver", qualified="host.sales"))
    cur.execute("SELECT id, date FROM host.sales")
    have = {(r[0], str(r[1])) for r in cur.fetchall()}
    new = long[[(i, str(d)) not in have for i, d in zip(long["id"], long["date"], strict=True)]]
    n = load_table(con, "host.sales", new, tables.types("sales", "sqlserver")) if len(new) else 0
    print(f"{n} new sales rows loaded into {database}.host.sales for {len(items)} items ({len(long) - n} already there)")
