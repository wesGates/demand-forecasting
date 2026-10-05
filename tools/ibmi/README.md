# IBM i

The pieces of this study that talk to an IBM i host. The host side itself
(the DB2 schema, the RPG fallback program, the CL job, the deploy script)
is a separate project: github.com/wesGates/forecast-host-ports.

- `ddl.py` prints the DB2 for i `CREATE TABLE` statements for the
  forecast, sales and suggested tables from `src/tables.py`, the one
  definition those tables have in every store (SQLite, SQL Server, DB2).
  The host project commits the output as its schema.
- `load_forecasts.py` pushes the daily job's forecast rows
  (`cache/forecasts.sqlite`) into the host's `FORECAST` table over ODBC.
  Reruns add nothing. The host's RPG program reads that table and falls
  back to its own forecast where no row exists.

Needs IBM's "IBM i Access ODBC Driver" and `pyodbc`, and `IBMI_HOST`,
`IBMI_USER`, `IBMI_LIBRARY` in the environment, with `IBMI_PASSWORD` set
in the shell session only.
