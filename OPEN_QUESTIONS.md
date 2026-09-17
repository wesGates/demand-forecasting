# Open questions and parked decisions

A living list. Each entry says what was found, what the evidence is, what the
options are, and what would settle it. Resolved items move to the bottom with
the decision recorded, so the reasoning is not lost.

Last updated: 2026-09-17.

---

## Open

### 1. XGBoost over-forecasts at the busiest store — parked

**Found:** residual diagnostics (FPP §5.4) for XGBoost at TX_2 show a mean
residual of **+7.66 units/day** on a ~100/day store — a systematic ~7.5%
over-forecast — with a right-skewed tail to +40. ETS at the same store is
essentially unbiased (+0.80). Across all ten stores XGBoost's mean bias is
only +1.24, so the problem concentrates at TX_2.

**What passes:** the residuals are uncorrelated (Ljung-Box p = 0.50), so the
method has taken everything *predictable*; the failure is in the level, not
the pattern.

**Options:**
- FPP's remedy: *"if the residuals have mean m, add m to all forecasts."* A
  per-store bias correction estimated on training folds.
- Investigate cause first: the held-out weeks are the low part of the annual
  cycle, and TX_2 was much higher in 2011–12. The rolling features should
  track the level, but the model may be leaning on `day_of_year`.
- Leave it, and report it as a finding.

**Parked until** the evaluation method is settled (fold count below), since a
wider window may change the picture.

### 2. Fold count — 8 folds is one season slice

**Found:** 8 folds × 7 days scores 28 Mar – 22 May 2016 only. Every result is
from spring. The January trough and the holiday spike — where a seasonal
model earns or loses its keep — are never scored.

**What FPP says (§5.10):** no prescribed count. The only constraint is that
"the earliest observations are not considered as test sets." The book's own
example uses *every* possible origin. So more folds, not fewer, is the
book's direction.

**Options, all one `Config` change:**

| `n_folds` | days scored | held-out from | run time |
|---|---|---|---|
| 8 (now) | 56 | 28 Mar 2016 | ~35 s |
| 26 | 182 | ~22 Nov 2015 | ~2 min |
| 52 | 364 | ~24 May 2015 | ~4 min |

**Cost:** widening moves the classification cutoff (step 3) earlier by the
same amount, since both read `Config.holdout_start`. At 52 folds the
classification window still has four years.

**Recommendation:** 26.

### 3. Horizon is confounded with weekday

**Found:** every fold origin is a Sunday, because origins step by exactly 7
days. So h = 1 is always Monday and h = 7 is always Sunday. The
error-by-horizon plot (FPP §5.10, figure 5.24) therefore mixes "how far
ahead" with "which weekday" — and Sunday is the highest-volume,
highest-variance day. That is why seasonal naive's error at h = 1 is *lower*
than at h = 7, and why the curves are not the textbook monotone rise.

**Why the book's example does not have this:** it steps origins by one day,
so they land on every weekday.

**Options:**
- Report the confound and read the horizon plot with it in mind (current).
- Add a `fold_step` field so origins can step by a number coprime to 7
  (e.g. 8 with 7-day windows: origins rotate through the week, with one-day
  gaps between scored windows).
- Step by 1 as FPP does — ~1,500 origins per store, overlapping windows,
  about two hours of compute.

**Decides it:** whether the horizon plot is going to carry weight in the
write-up. If it is, fix it; if RMSSE-by-store is the headline, note it.

### 4. A private filename is in pushed git history

Commit `34eddc0`'s `.gitignore` names a private working document. Replaced
with a wildcard pattern from `18a185e` onward, so it will not recur — but
the earlier commit is on GitHub. Removing it means rewriting history and
force-pushing. Owner's decision; not done.

---

## Next steps, not yet started

- **Pooled run.** `Config(pool_by="item_id")` is wired and untested on real
  data. One config flip, ~1 minute. Adds the cross-learning row to every
  table.
- **% improvement quartiles.** Per-series improvement over a benchmark,
  reported as Q1 / median / Q3 alongside win rate. ~10 lines in `summarise`.
- **Two calendar features.** Days until the next holiday, and days until the
  next SNAP day — both known years ahead, both cheap. Currently only on/off
  flags exist.
- **`min_train_days`.** Currently 365. Two full years (730) is a common
  choice and would cost nothing on the current item.
- **ARIMA.** ETS is the classical contender now; ARIMA (FPP Ch. 9) would be
  one registry entry if a second is wanted.
- **The write-up.** A dated report under `findings/` for the current item,
  then the final `.ipynb`.

---

## Resolved

*(decisions recorded here as they are made)*

- **RMSSE scaling** — lag 7 (seasonal naive, FPP §5.8's rule for seasonal
  data), denominator computed once per series over all pre-holdout training
  data. Alternatives (lag 1 for M5 comparability; per-fold denominator) are
  one-line switches on `Config`. Decided 2026-09-12.
- **Price as a feature** — off, via `Config.use_price`. For the current item
  price takes three values in five years on the same two dates at every
  store: a clock, not a variable. Decided 2026-09-11.
- **Availability screen** — measure and warn (`max_zero_run > 30`), do not
  silently trim. Decided 2026-09-10.
- **One item across ten stores** — scope reduced from three items to one for
  time. Consequence: every store is `smooth`, so the study answers "which
  method wins on a fast mover, and does it depend on the store" rather than
  "does the best method change with demand class." Decided 2026-09-11.
