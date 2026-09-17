# Demand forecasting on M5 — a forecaster's-toolbox build

Daily unit-sales forecasting for retail replenishment, built step by step
along the five-stage method in *Forecasting: Principles and Practice*
(Hyndman & Athanasopoulos, [otexts.com/fpppy](https://otexts.com/fpppy/)).

The question: on a fast-moving grocery item sold at ten stores, does a
gradient-boosted model beat the simple methods a person would use without one —
and does the answer depend on the store?

**Status:** steps 1–5 built and validated on one item across ten stores.
Write-up in progress. Results below are preliminary.

---

## What it does

- Loads the public M5 dataset (Walmart, 2011–2016), reshapes it to one row per
  store-item-day, and joins calendar, holiday, SNAP and price information.
- Screens each series for *availability* before anything else — a month of zero
  sales on a fast-moving item is a stocking gap, not demand, and it corrupts
  every statistic downstream.
- Classifies demand by how often and how consistently an item sells
  (ADI / CV², the Syntetos–Boylan scheme), computed on training data only.
- Forecasts seven days ahead from a single origin, daily, exactly as a
  replenishment run would: stand on Sunday, order for the week.
- Compares two learned models — **XGBoost** and **ETS** (Holt-Winters) — against
  **five benchmarks**: mean, naïve, seasonal naïve, drift, and a 28-day moving
  average.
- Scores with **RMSSE** (FPP §5.8) over walk-forward folds (FPP §5.10), plus
  RMSE, MAE, bias, and win rates.
- Refuses to report a number until a validator has passed.

## Why the safeguards are the point

An earlier version of this project produced a 90% win rate that was wrong twice
over: lag features reached into the test window, and the only benchmark was
one that a dull model beats without skill. Everything here is built so those
two failures cannot recur:

- **Nothing sees the future.** A forecaster receives a `Context` with no target
  values in it, and training rows are filtered by *when their answer became
  observable*, not by when they were built. A date-comparison assertion refuses
  anything that reaches past the origin.
- **Five benchmarks, not one.** Seasonal naïve stakes everything on one past
  day; on a noisy series its error is about √2 worse than predicting the mean.
- **`python -m src.validate`** runs six checks that each catch a different way
  of being *plausibly* wrong — closed-form benchmark answers, feature values
  recomputed by a second route, per-state SNAP flags against the raw calendar,
  a synthetic noise floor no honest model can beat, a shuffled target no model
  should learn from, and byte-identical reruns.

## Layout

```
src/
  step1_problem.py    Config — every decision in one frozen object, with FPP refs
  step2_data.py       load, reshape, join, pre-launch trim, parquet cache
  step3_explore.py    availability screen, ADI/CV² classification
  features.py         origin-based lags, rolling stats, calendar; leakage assertion
  step4_models.py     five benchmarks + XGBoost + ETS behind one interface
  step5_evaluate.py   walk-forward harness, RMSSE, tables
  plots.py            every figure, on one palette
  validate.py         the gate
  render_figures.py   regenerate all figures to figures/
notebooks/
  01_explore.py       FPP step 3 — graph the data before modelling anything
  02_evaluate.py      FPP step 5 — run the comparison, read the diagnostics
```

Notebooks are plain `.py` files with `# %%` cell markers: open in VS Code and
run cells with Shift+Enter in the Interactive Window. A cell near the top pops
figures out into their own windows.

## Setup

```
python -m venv .venv                  # Python 3.14
.venv\Scripts\activate
pip install -r requirements.txt
```

Data is the Kaggle **M5 Forecasting – Accuracy** competition set. Place these
three files in `data/` (gitignored):

```
sales_train_evaluation.csv
calendar.csv
sell_prices.csv
```

Then:

```
python -m src.validate          # must pass before any number is quoted
python -m src.render_figures    # all figures -> figures/
python -m src.step5_evaluate    # the comparison, as tables
```

## Preliminary result

One item (`FOODS_3_586`), ten stores, 8 walk-forward folds of 7 days,
RMSSE scaled by the seasonal-naïve error on training data:

| method | RMSSE mean | median |
|---|---|---|
| xgboost | 0.58 | 0.56 |
| ets | 0.60 | 0.57 |
| moving average (28) | 0.69 | 0.67 |
| seasonal naïve | 0.78 | 0.74 |
| mean | 0.83 | 0.77 |
| naïve / drift | 1.06 | 0.93 |

The two models tie with each other and clearly beat every simple method. Their
advantage is largest at the busiest stores and shrinks at the quietest, where
a 28-day moving average is nearly as good. The residual diagnostics show a
systematic over-forecast in one model at one store, which is under
investigation.

Caveat that matters: the scored window is a single spring slice (56 days). A
wider window is one config change and is the next step.

## What is deliberately not claimed

This is a learning project on public data. M5 has no inventory positions,
receipts or transfers, so the replenishment calculation a forecast feeds into
can only be simulated. Price is excluded as a feature for this item because it
changes on two fixed dates in five years — a clock, not a variable — and that
would not hold for another product. Pooling across stores is wired in but not
yet run.
