# Findings

One dated file per item. Each states what was measured, with the numbers,
and opens with a provenance block: branch, commit, the config line that
rebuilds the run, and the cache files the numbers came from. Anyone can
check out the commit and either read the same cache file or refit and
compare.

| file | item |
|---|---|
| `2026-09-22-FOODS_3_586.md` | the first study's findings, carried over from the parent repository |
| `2026-09-22-item1-quantile-objective.md` | XGBoost on the quantile objective |
| `2026-09-23-item2-every-day-origins.md` | every day an origin |
| `2026-09-23-item3-pooling.md` | one model for all ten stores |
| `2026-09-23-item4-intermittent-item.md` | an intermittent, declining item |
| `2026-09-23-item5-level-relative-target.md` | trees on a level-relative target |
| `2026-09-23-item6-run-registry.md` | the run registry |
| `2026-09-23-item7-parallel-harness.md` | the parallel harness |
| `2026-09-23-item8-review-fixes.md` | the review fixes and the refit under them |
| `2026-09-25-item9-cross-item-pool.md` | both items in one pool, and why it lost on the slow mover |
| `2026-09-26-sql-server-registry.md` | the run registry loaded into SQL Server and queried there |

Each file stays as written at its stage. Later items do not rewrite
earlier ones; when a number is superseded, the later file says so and gives
the new one. The numbers to quote are always the latest file's.

The notebooks (`notebooks/`) are the procedure the first study followed.
They run unchanged for any item and print what they find. The checklists
at the end of `01_explore` and `02_evaluate` are what the first findings
file answers.
