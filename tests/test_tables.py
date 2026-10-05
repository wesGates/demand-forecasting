"""The one table definition renders valid DDL for every store it feeds."""

from __future__ import annotations

import sqlite3

import pytest

from src import tables


@pytest.mark.parametrize("table", list(tables.TABLES))
def test_sqlite_ddl_creates_a_table_that_takes_a_row(table):
    con = sqlite3.connect(":memory:")
    con.execute(tables.ddl(table, "sqlite"))
    cols = tables.columns(table)
    con.execute(
        f"INSERT INTO {table} VALUES ({','.join('?' * len(cols))})", [1] * len(cols)
    )
    assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1


@pytest.mark.parametrize("dialect", ["sqlite", "sqlserver", "db2"])
def test_every_dialect_names_every_column_and_the_key(dialect):
    for table, (cols, key) in tables.TABLES.items():
        text = tables.ddl(table, dialect, qualified="X." + table)
        assert text.startswith("CREATE TABLE ") and "X." + table in text
        for name, _, _ in cols:
            assert f"    {name} " in text
        assert f"PRIMARY KEY ({', '.join(key)})" in text


def test_key_columns_exist():
    for table, (cols, key) in tables.TABLES.items():
        names = {n for n, _, _ in cols}
        assert set(key) <= names, table
