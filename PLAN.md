# Plan

Read this first. It says where the project stands, what comes next, what
things cost, and the rules. `GLOSSARY.md` defines the terms; `findings/`
has the numbers.

## Where it stands

The parent repository established that XGBoost, ETS and seasonal ARIMA
beat six benchmarks on one fast-moving item at ten stores (52 weekly
folds), and that a point forecast is not an order. This repository took
that pipeline through ten items:

| item | branch | outcome |
|---|---|---|
| 1. quantile objective | on main | matches calibrated error quantiles within 0.002 at 26× the cost; ARIMA plus calibration best |
| 2. every day an origin | `item2-fold-step-1` | tables within 0.005; the naïve and drift Sunday bias was an artefact; adopted as the reported layout |
| 3. pooled XGBoost | `item3-pooling` | best method on the fast mover; Q1 improvement 9% against 1.5% per store; bias cut by two thirds |
| 4. intermittent item | `item4-intermittent` | learned models lose at every store; the cause is level drift, which trees cannot extrapolate |
| 5. level-relative target | `item5-level-relative-target` | closes the gap on the declining item (0.570 → 0.501), small gain on the fast mover (0.622 → 0.611) |
| 6. run registry | `item6-run-registry` | `run` and `fold_score` tables beside the cache, suites, `src.run`, paired comparisons |
| 7. parallel harness | `item7-parallel-harness` | one thread per fit, tasks in worker processes; forecasts identical; every-day runs 15 min per item |
| 8. review fixes | `item8-review-fixes` | closure imputation backward-only; events chosen before the cutoff; fallbacks and unscored folds counted; cache checked against the data; `step1` in the key; atomic writes; registry guards |
| 9. both items in one pool | `exp-xgb-quick-wins` | ties on the fast mover, loses on the slow one in holiday weeks; the additive target carries holiday lifts across items, so a shared pool needs a proportional target |
| 10. the registry in SQL Server | on main | the two registry tables loaded into SQL Server 2022 (Docker) in under three minutes and queried there; `tools/sqlserver/` |

The same branch added two weaker relatives of ARIMA as baselines
(`arima_plain`, no regressors; `arma`, no seasonality, no regressors) and
two dev-suite experiments (`xgboost_poisson`, `xgboost_rel_recent`) that
went no further. The report (`report/report.md`) and the two-page brief
(`report/brief.md`) are built from `tools/report/`.

The numbers to quote are in `findings/2026-09-23-item8-review-fixes.md`.
The ordering prototype (`src/order.py`, notebook 03) is from before item 1
and is where the quantile work continues.

## Next, in order

1. The pooled, level-relative model's own residuals and calibrated
   quantiles (the order step, on the model that won).
2. Intermittent demand properly: a sample of items chosen by demand class,
   Croston and TSB as registered methods (FPP §13.2), results by class, a
   service-level score beside RMSSE, and a per-method minimum history.
3. Pooling wider than one item, on a proportional target (item 9 showed
   the additive target cannot share holiday effects across volumes).
4. New items. The harness needs a year of history per series, so new items
   need their own evaluation: items launched inside the test year, against
   simple fallbacks and pretrained models that need no history.
5. The SQL Server results store, beyond the first version (the two
   registry tables loaded and queried in SQL Server 2022 on Docker,
   `tools/sqlserver/`, findings `2026-09-26-sql-server-registry.md`): a
   query-only login, runs written from `src.run`, the forecast rows, and
   per-forecast explanations (designed, below).
6. Fourier terms or an STL decomposition for the annual cycle in the
   classical models (FPP §13.1).

## Designed, not built

**SQL Server as the results store.** Built in its simplest form on
2026-09-26 (registry tables only, loaded from SQLite). The full design,
SQL Server in a Docker container as source and sink, in new files only so
no cached run is invalidated:

1. sales, calendar, store and item tables loaded from the M5 files, with a
   test that the panel read back equals `load_panel`;
2. a `run` table with one row per cache entry (run id = cache digest,
   method, pooling, layout, item and store scope, code digest, branch,
   commit, full config) and a `forecast` table keyed by run, store, item,
   origin and target date; actuals from a join on the sales table; filled
   by an idempotent sync from the cache and never edited by hand;
3. scoring views (RMSSE, bias, WAPE, MAPE, win rate, quartiles by store and
   kind of week) with a test that they equal the pandas tables;
4. a daily job that reads history up to a date, fits, and writes the next
   seven days; rerunning a date leaves the table unchanged.

Everything keeps running without a database. CI can run the tests against
a SQL Server service container on synthetic series. About 5–6 h.

**Plain-language explanations per forecast.** XGBoost's exact feature
contributions summed into a few groups (recent level, day of week,
holiday, SNAP, time of year), one sentence per forecast, a test that each
explanation adds up to its forecast, and a stability check across adjacent
origins. ETS and ARIMA explain themselves through their components and
coefficients (FPP ch. 8 and 10). An explanation describes the model, not
the customer (FPP §7.8).

**Deferred by the 2026-09-23 review.** The pooled path at hundreds of items
(build the design once per origin and index the pool; one task per fold is
serial over stores); the pooled quantile model refits per store; a worker
exception discards the whole run's rows; whether Martin Luther King Day
(an 18.7% effect) joins the holiday features is an experiment for the
registry.

## Costs

Measured 2026-09-23 under the parallel harness, every core, one thread per
fit.

| layout | all point methods | XGBoost only | ARIMA only |
|---|---|---|---|
| `dev`: 3 stores, 8 folds, weekly | ~45 s | ~2 s | ~33 s |
| `weekly`: 10 stores, 52 folds | ~3 min | ~20 s | ~2 min |
| `everyday`: 10 stores, 358 origins | ~15 min | ~2 min | ~10 min |

Two heavy jobs at once contend, so run one at a time. Develop on `dev`,
run `everyday` once per change that survives it, report from `everyday`.
Edits to the harness, the shared base, the features, the loader or the
config invalidate every method's cache, so batch them; `tools/refit_all.py`
remakes every reported run in about 45 min.

## Rules

- No employer, product or third-party document is named anywhere in the
  repository. The words to scan for are kept in the owner's private notes;
  a case-insensitive `git grep` for each must return nothing before a
  commit.
- Commits carry the owner's identity only.
- Every number in a report is read back against the table that produced
  it. The validator and the tests pass first.
- Report voice: third person, technical; cite as (FPP §5.8); say what an
  orderer can do.
- One branch per item, with a findings file; merged after review.
- After a change that touches forecasts, refit and compare
  (`tools/refit_all.py`, `tools/compare_runs.py`, `tools/compare_forecasts.py`)
  before quoting anything.

## Local setup

The checkout has its own `.venv` (Python 3.14, `requirements.txt`).
`data` holds the three M5 files (or a symlink to them). `cache/` holds the runs and the registry and is gitignored.
Figures render to `figures/`, also gitignored; the report keeps its own
copies.
