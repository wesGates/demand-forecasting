# Demand forecasting on M5

A daily, store-level demand forecast, tested on a full year at ten stores
against the forecasts an orderer might use without it. Public data: the
M5 dataset of Walmart daily sales, with a calendar of holidays and SNAP
benefit days.

**84% of store-weeks** better than ordering from last week's number, by a
median of 28%, on a fast, regular mover.

**10 of 10 stores** better than ARIMA, the strongest statistical model,
over the full year.

**23% less weekly order error** than last week's number, 36 units a week
per store down to 27.6.

Five years of daily history, ten stores, 358 forecast origins per store,
3,580 scored store-weeks per method, ten methods (two benchmarks, four
statistical models and four machine learning variants), and every model
was refit at each origin.

Each rung below is one modelling step up from the one before. The final
model is gradient-boosted trees (XGBoost) trained across all ten stores on
a level-relative target, and its gain is over a strong statistical model.

| rung | RMSSE | weekly error, units | vs previous rung, weeks won | vs previous rung, median gain | vs seasonal naive, weeks won | vs seasonal naive, median gain |
|---|---|---|---|---|---|---|
| This day last week (seasonal naive) | 0.86 | 36.0 | | | | |
| ARMA, no seasonality, no inputs | 0.70 | 33.5 | 77% | 19% | 77% | 19% |
| ARIMA, no holiday or SNAP inputs | 0.66 | 32.7 | 65% | 5% | 82% | 24% |
| ARIMA | 0.65 | 31.5 | 58% | 1% | 83% | 25% |
| **XGBoost, pooled, level-relative (final)** | 0.61 | 27.6 | 58% | 3% | 84% | 28% |

RMSSE is the scaled error the M5 competition used, with 1.0 the error of
last week's number on the training history. Weekly error is the average
miss on a week's total order, in units, about 9% of a typical week's sales
for the final model and 11% for last week's number.

On a slow mover in decline (0.6 to 7 units a day) a 28-day moving average
is the best forecast and the final model ties it (0.50 each), so items are
routed by demand class before anything is fitted, the machine learning
model for steady daily movers and the simple average for the rest.

The full story, with the figures, the tables for both items and every
method against every baseline, is [`report/report.pdf`](report/report.pdf).
The two-page summary is [`report/brief.pdf`](report/brief.pdf). Both are
built from markdown with [`tools/report/`](tools/report/).

## What the system does

- Forecasts a seven-day order window from every day of the year,
  refitting at each origin, the way an ordering system might use them.
- Holiday and SNAP effects measured from the data. Seven of thirty
  calendar events move this item's sales, and the model sees the days
  until and since each.
- Bias reported beside error throughout. A forecast that runs
  systematically high is shrink every week and one that runs low is a
  stockout.
- All resulting numbers are byte-identical on subsequent runs. Each change
  to a model is scored against the previous version on the same 3,580
  store-weeks, so noise shows up as a win rate close to 50%. The run
  records live in SQLite and load into SQL Server with one script
  ([`tools/sqlserver/`](tools/sqlserver/)), where the report's numbers
  come back by query.
- A full-year evaluation of all methods on one item in about fifteen
  minutes, and a validator that has to pass before any number is quoted.

The model's inputs are recent sales (the last three days and the same
weekday over the last four weeks), the weekly pattern, the level and its
spread over the last week to two months, the calendar and how many days
ahead the forecast is, the days until and since the seven holidays that
matter, SNAP benefit days, and, in the pooled model, the store.

## How it was built

The first study, tag `first-study`, compared XGBoost, ETS and seasonal
ARIMA against six benchmarks on one fast mover, on 52 weekly folds.
Everything after it took the same pipeline further one numbered item at a
time, each on its own branch with a dated findings file in
[`findings/`](findings/).

| item | question | answer |
|---|---|---|
| 1 | Does XGBoost trained on the quantile objective beat calibrated error quantiles? | No. Same score within 0.002, at 26× the fit cost. |
| 2 | Does making every day an origin change the results? | Tables move by less than 0.01. Two benchmarks lose a Sunday artefact. Adopted as the reported layout. |
| 3 | One model pooled across the ten stores? | Best method on the fast mover. Helps the quiet stores most and cuts bias by two thirds. |
| 4 | A slow, declining item? | The machine learning models lose at every store. The cause is level drift, which trees cannot extrapolate. |
| 5 | Trees on a level-relative target? | Closes the gap on the declining item, small gain on the fast mover. |
| 6 | A run registry | Every run on record with its config, code version, commit and scores, and paired comparisons between runs. |
| 7 | A parallel harness | Every-day runs in 15 min per item, down from about 2 h. Forecasts identical. |
| 8 | A code review | One leak fixed (the closure imputation looked forward), fallbacks and unscored folds counted, the cache checked against the data. |
| 9 | Both items in one pool? | Ties on the fast mover, loses on the slow one in holiday weeks. A shared pool needs the target on a common scale. |
| 10 | The registry in SQL Server | Both tables loaded into SQL Server 2022 in under three minutes. The report's numbers come back by query. |

[`PLAN.md`](PLAN.md) has the status, what comes next and the rules.
[`GLOSSARY.md`](GLOSSARY.md) defines the terms.

## Running it

```
python -m venv .venv                       # Python 3.14
source .venv/bin/activate
pip install -r requirements.txt
```

Put the Kaggle M5 files `sales_train_evaluation.csv`, `calendar.csv` and
`sell_prices.csv` in `data/` (gitignored). Then:

```
python -m src.validate                     # ten checks; run before quoting a number
python -m pytest tests                     # the test suite
python -m src.run --suite dev --item fast --methods xgboost_rel --note "what this tries"
python -m src.registry list everyday       # every run on record, on the reported layout
python -m src.registry last <run_id>       # a run against its predecessor
```

Suites fix the layout a change is scored on. `dev` is three stores and
eight weekly folds (under a minute), `weekly` is 52 folds (about 3 min for
every method), `everyday` is 358 origins (about 15 min). Develop on `dev`
and confirm on `everyday`. [`tools/`](tools/) has the refit, gate,
comparison and figure scripts.

## How the pipeline works

- `src/step2_data.py` loads M5, reshapes it to one row per store, item and
  day, joins the calendar, SNAP and price data, and fills the Christmas
  closure from the four preceding weeks.
- `src/step3_explore.py` screens each series for stock-out gaps and
  classifies demand by ADI and CV² on training data only.
- `src/features.py` builds lags, rolling summaries, same-weekday means and
  holiday proximity from data at or before the origin, with an assertion
  on every fold.
- `src/models/` holds one module per forecaster. `src/step4_models.py`
  registers them.
- `src/step5_evaluate.py` runs the walk-forward in parallel and caches each
  method's forecasts under a key made of the config and the code the
  method depends on. `src/scoring.py` turns forecasts into tables.
- `src/registry.py` records every run. `src/suites.py` names the layouts.
  `src/run.py` runs, registers and compares.
- `src/validate.py` is the gate, ten checks including negative controls on
  synthetic data.

Each method's forecasts are cached under `cache/predictions/`. Editing one
model refits that model only, and a cached run is checked against the
data on disk every time it is read. The registry (`cache/registry.sqlite`)
can be rebuilt from the cache with `python -m src.registry backfill`.

## Layout

```
src/
  step1_problem.py    Config, every decision in one frozen object
  step2_data.py       load, join, trim, impute closures, cache the panel
  step3_explore.py    availability screen, demand classification, event effects
  features.py         origin-based features and the leakage assertion
  step4_models.py     the forecaster registry
  models/             one module per forecaster
  step5_evaluate.py   the parallel walk-forward harness and the cache
  scoring.py          fold scores, tables, win rates, improvement quartiles
  registry.py         the run registry
  suites.py           the named layouts
  run.py              run, register, compare
  order.py            the order-quantity prototype (calibrated quantiles)
  plots.py            every figure, one palette
  render_figures.py   figures for the notebooks
  validate.py         the gate
tools/                refit, gate, comparison, figure and report scripts
tests/                pytest suite
notebooks/            01_explore, 02_evaluate, 03_order (percent-format scripts)
findings/             one dated file per item, with provenance
report/               the report and the brief (PDF); sources and figures are local
```

## Limits

Public data only. M5 has no inventory, deliveries or stockouts, so an
order can only be simulated and a zero on the shelf cannot be told from a
zero in demand. Two items at ten stores is a small study. New items with
no history are not addressed. The plan lists what comes next.
