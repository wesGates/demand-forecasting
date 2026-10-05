"""
The tables that forecasts and sales are stored in, defined once.

Three stores hold the same tables: the SQLite file the daily job writes,
SQL Server, and DB2 on the IBM i host. Each column is named here with a
generic type, and `ddl()` renders the CREATE TABLE for a dialect. A column
added here reaches every store on the next load; the loaders never carry
their own column lists.

  forecast   one row per series, method and target day, from one origin
  sales      one row per series and day, the daily units sold
  suggested  what the host side orders from: the model's forecast where
             one exists, otherwise its own fallback; `source` says which

The RPG program on the host names these columns by hand in its embedded
SQL. That is the one copy this module cannot generate, and the
reconciliation test is what catches it drifting.
"""

from __future__ import annotations

# (name, generic type, note)
FORECAST = [
    ("as_of", "date", "the origin, the last day of sales the forecast saw"),
    ("id", "text40", "M5 series id, item and store"),
    ("item_id", "text20", ""),
    ("store_id", "text8", ""),
    ("method", "text40", "method name, '@pool_by' appended when pooled"),
    ("horizon", "int", "1..7"),
    ("target_date", "date", ""),
    ("forecast", "decimal", "units"),
    ("fallback", "bool", "the method fell back to the 28-day mean"),
    ("code_digest", "text64", "the method's code at the time, as the cache keys it"),
    ("made_at", "timestamp", "UTC"),
]
FORECAST_KEY = ("as_of", "id", "method", "target_date")

SALES = [
    ("id", "text40", "M5 series id, item and store"),
    ("item_id", "text20", ""),
    ("store_id", "text8", ""),
    ("date", "date", ""),
    ("units", "int", "units sold that day"),
]
SALES_KEY = ("id", "date")

SUGGESTED = [
    ("as_of", "date", ""),
    ("id", "text40", ""),
    ("item_id", "text20", ""),
    ("store_id", "text8", ""),
    ("horizon", "int", ""),
    ("target_date", "date", ""),
    ("forecast", "decimal", "units"),
    ("source", "text40", "the method the row came from"),
]
SUGGESTED_KEY = ("as_of", "id", "target_date")

TABLES = {
    "forecast": (FORECAST, FORECAST_KEY),
    "sales": (SALES, SALES_KEY),
    "suggested": (SUGGESTED, SUGGESTED_KEY),
}

# Generic type -> each store's type. SQLite has no date type, so dates and
# timestamps are ISO text there; DB2 for i has no boolean, so SMALLINT.
TYPES = {
    "sqlite": {
        "date": "TEXT",
        "timestamp": "TEXT",
        "text8": "TEXT",
        "text20": "TEXT",
        "text40": "TEXT",
        "text64": "TEXT",
        "int": "INTEGER",
        "decimal": "REAL",
        "bool": "INTEGER",
    },
    "sqlserver": {
        "date": "DATE",
        "timestamp": "DATETIME2(0)",
        "text8": "VARCHAR(8)",
        "text20": "VARCHAR(20)",
        "text40": "VARCHAR(40)",
        "text64": "VARCHAR(64)",
        "int": "INT",
        "decimal": "FLOAT",
        "bool": "BIT",
    },
    "db2": {
        "date": "DATE",
        "timestamp": "TIMESTAMP",
        "text8": "VARCHAR(8)",
        "text20": "VARCHAR(20)",
        "text40": "VARCHAR(40)",
        "text64": "VARCHAR(64)",
        "int": "INTEGER",
        "decimal": "DECIMAL(15, 6)",
        "bool": "SMALLINT",
    },
}


def columns(table: str) -> list[str]:
    """The column names of a table, in order."""
    return [name for name, _, _ in TABLES[table][0]]


def types(table: str, dialect: str) -> dict[str, str]:
    """Column name -> the dialect's type, for loaders that build INSERTs."""
    return {name: TYPES[dialect][kind] for name, kind, _ in TABLES[table][0]}


def ddl(table: str, dialect: str, qualified: str | None = None) -> str:
    """
    CREATE TABLE for one table in one dialect. `qualified` overrides the
    table name, e.g. 'host.forecast' on SQL Server or 'MYLIB.FORECAST' on
    DB2. SQLite gets IF NOT EXISTS; the others are created once by hand.
    """
    cols, key = TABLES[table]
    name = qualified or table
    exists = "IF NOT EXISTS " if dialect == "sqlite" else ""
    body = ",\n".join(f"    {n} {TYPES[dialect][k]} NOT NULL" for n, k, _ in cols)
    return (
        f"CREATE TABLE {exists}{name} (\n{body},\n    PRIMARY KEY ({', '.join(key)})\n)"
    )
