"""
The validator. Run it before trusting any number this project produces.

    python -m src.validate

Six checks in five groups, each aimed at a different way of being wrong. What
they have in common is the only thing that matters: **every bug they catch
still produces plausible-looking output.** A result that is obviously broken
gets fixed the moment you see it; a result that is quietly wrong ends up in a
report.

  1. Closed-form benchmarks - does each method compute what its name says?
     On a series that is 10 every day, every method must return exactly 10. On
     a straight ramp, drift must extrapolate it perfectly. Catches off-by-one
     indexing, which on real data looks entirely reasonable.

  2. Feature spot-checks - is `lag_7` really seven days back?
     Each feature is recomputed from the raw panel by *date filtering*, while
     the implementation uses array slicing. Two different routes to the same
     number; if they disagree, one of them is wrong.

  3. SNAP flags - does each row carry its own state's schedule?
     Every store-date compared against the calendar's column for that state.
     Reading the wrong state's column would still give a plausible 0/1 flag
     on a third of days; only a date-by-date comparison catches it.

  4. Noise floor and shuffled target - is anything reading the future?
     Synthetic data whose irreducible error is known by construction, and a
     scrambled target with no pattern left to find. The structural date
     assertion in `features.py` is stronger for the leak we know about; these
     are the net for the ones we do not.

  5. Determinism - does the same input give the same answer twice?
     Cheap, and it is what makes a reported number reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    DOW_WINDOWS,
    LAGS,
    ROLL_WINDOWS,
    build_fold_features,
    build_supervised,
)
from src.step1_problem import STUDY_ITEMS, Config
from src.step2_data import SNAP_BY_STATE, load_panel
from src.step4_models import BENCHMARKS, Context, fit_predict_xgboost

TOL = 1e-9


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _series_frame(values: np.ndarray, start: str = "2013-01-07") -> pd.DataFrame:
    """A minimal panel for one synthetic series. Starts on a Monday."""
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "id": "SYNTH",
            "store_id": "S1",
            "item_id": "SYNTH_ITEM",
            "date": dates,
            "sales": np.asarray(values, dtype=float),
            "snap": 0,
            "event_name_1": pd.Series([None] * len(values), dtype=object),
        }
    )


def _context(frame: pd.DataFrame, horizon: int = 7, **kw) -> Context:
    """Split a synthetic frame into history + targets at the natural origin."""
    history = frame.iloc[:-horizon]
    targets = frame.iloc[-horizon:].drop(columns=["sales"])
    return Context(
        history=history,
        targets=targets,
        origin=history["date"].iloc[-1],
        horizon=horizon,
        series_id="SYNTH",
        **kw,
    )


def _synthetic_panel(
    n_series: int = 6, n_days: int = 1000, noise_sd: float = 2.0, seed: int = 0
) -> pd.DataFrame:
    """
    A panel whose irreducible error is known exactly.

    sales = level + weekly shape + slow trend + noise(0, noise_sd)

    Everything but the noise is a deterministic function of the date, so a
    perfect forecaster would predict it exactly and be left with the noise
    alone. `noise_sd` is therefore the best RMSE physically achievable.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-07", periods=n_days, freq="D")
    frames = []
    for k in range(n_series):
        weekly = 4.0 * np.sin(2 * np.pi * dates.dayofweek / 7)
        signal = (20 + 6 * k) + weekly + np.linspace(0, 3, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "id": f"SYNTH_{k:02d}",
                    "store_id": f"S{k:02d}",
                    "item_id": "SYNTH_ITEM",
                    "date": dates,
                    "sales": np.maximum(signal + rng.normal(0, noise_sd, n_days), 0),
                    "snap": 0,
                    "event_name_1": pd.Series([None] * n_days, dtype=object),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _walk_forward_rmse(panel: pd.DataFrame, cfg: Config) -> tuple[float, float, int]:
    """
    Run XGBoost and the mean benchmark over every fold of a synthetic panel.

    Returns (model RMSE, mean-benchmark RMSE, number of fold-series compared).
    Used by both the noise-floor and shuffled-target checks.
    """
    origins = cfg.fold_origins(panel["date"].max())
    model_err, bench_err, n = [], [], 0

    for sid, series in panel.groupby("id", sort=True, observed=True):
        series = series.sort_values("date").reset_index(drop=True)
        pool = build_supervised(series, cfg.horizon)

        for origin in origins:
            history = series[series["date"] <= origin]
            targets = series[
                (series["date"] > origin)
                & (series["date"] <= origin + pd.Timedelta(days=cfg.horizon))
            ]
            if len(targets) < cfg.horizon or len(history) < cfg.min_train_days:
                continue

            actual = targets["sales"].to_numpy(dtype=float)
            ctx = Context(
                history=history,
                targets=targets.drop(columns=["sales"]),
                origin=origin,
                horizon=cfg.horizon,
                series_id=sid,
                season=cfg.season,
                train_pool=pool,
                seed=cfg.seed,
            )
            pred = fit_predict_xgboost(ctx)
            if np.isnan(pred).any():
                continue

            model_err.append((actual - pred) ** 2)
            bench_err.append((actual - BENCHMARKS["mean"](ctx)) ** 2)
            n += 1

    return (
        float(np.sqrt(np.concatenate(model_err).mean())),
        float(np.sqrt(np.concatenate(bench_err).mean())),
        n,
    )


# --------------------------------------------------------------------------- #
# 1. Closed-form benchmarks
# --------------------------------------------------------------------------- #


def check_benchmarks_closed_form() -> dict:
    """Every benchmark, on series whose correct answer is known by hand."""
    failures = []

    # A flat series: there is only one defensible forecast, and it is 10.
    ctx = _context(_series_frame(np.full(200, 10.0)))
    for name, fn in BENCHMARKS.items():
        got = fn(ctx)
        if not np.allclose(got, 10.0, atol=TOL):
            failures.append(f"{name} on a constant-10 series returned {got}")

    # A ramp y_t = t. Slope is exactly 1 per day, so drift must continue it.
    n = 200
    ctx = _context(_series_frame(np.arange(1, n + 1, dtype=float)))
    last = float(n - 7)  # history stops 7 days before the end
    expected = {
        "naive": np.full(7, last),
        "drift": last + np.arange(1, 8),
        "mean": np.full(7, np.arange(1, n - 6).mean()),
        "seasonal_naive": np.arange(last - 6, last + 1),
        "moving_average_28": np.full(7, np.arange(last - 27, last + 1).mean()),
    }
    for name, want in expected.items():
        got = BENCHMARKS[name](ctx)
        if not np.allclose(got, want, atol=1e-6):
            failures.append(f"{name} on a ramp returned {got}, expected {want}")

    return {
        "check": "closed-form benchmarks",
        "tested": f"{len(BENCHMARKS)} methods on 2 series with known answers",
        "failures": failures,
        "verdict": "ok" if not failures else "FAILED",
    }


# --------------------------------------------------------------------------- #
# 2. Feature spot-checks
# --------------------------------------------------------------------------- #


def check_features_hand_computed(cfg: Config | None = None) -> dict:
    """
    Recompute every feature from the raw panel by date, and compare.

    The implementation slices numpy arrays by position; this recomputes the same
    quantities by filtering the DataFrame on dates. Two independent routes - a
    single off-by-one shows up as a mismatch instead of as a plausible number.
    """
    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    panel = load_panel(cfg, verbose=False)
    series = panel[panel["id"] == panel["id"].iloc[0]].sort_values("date")

    failures, checked = [], 0
    for offset in (7, 200, 900):  # several origins, not just the convenient one
        origin = series["date"].max() - pd.Timedelta(days=offset)
        history = series[series["date"] <= origin]
        targets = series[
            (series["date"] > origin)
            & (series["date"] <= origin + pd.Timedelta(days=cfg.horizon))
        ]
        feats = build_fold_features(history, targets, origin)

        # `sales` is stored as float32 to halve panel memory. The feature code
        # promotes to float64 before doing any arithmetic; this recomputation
        # must do the same, or the two sides differ at the 1e-6 level purely
        # from float32 accumulation and every rolling mean looks like a bug.
        def sales_on(day: pd.Timestamp) -> float:
            return float(series.loc[series["date"] == day, "sales"].iloc[0])

        def f64(values) -> pd.Series:
            return values.astype("float64")

        for k in LAGS:
            # lag_k counts back from the origin, so lag_1 IS the origin day.
            want = sales_on(origin - pd.Timedelta(days=k - 1))
            got = float(feats[f"lag_{k}"].iloc[0])
            checked += 1
            if abs(got - want) > TOL:
                failures.append(f"lag_{k} at {origin.date()}: got {got}, want {want}")

        for w in ROLL_WINDOWS:
            window = f64(
                history[history["date"] > origin - pd.Timedelta(days=w)]["sales"]
            )
            for stat, want in (("mean", window.mean()), ("std", window.std(ddof=1))):
                got = float(feats[f"roll_{stat}_{w}"].iloc[0])
                checked += 1
                if not np.isclose(got, float(want), rtol=1e-9, atol=1e-9):
                    failures.append(
                        f"roll_{stat}_{w} at {origin.date()}: got {got}, want {want}"
                    )

        for k in DOW_WINDOWS:
            for row in range(len(targets)):
                dow = targets["date"].iloc[row].dayofweek
                same = f64(history[history["date"].dt.dayofweek == dow]["sales"])
                want = float(same.tail(k).mean())
                got = float(feats[f"dow_mean_{k}"].iloc[row])
                checked += 1
                if not np.isclose(got, want, rtol=1e-9, atol=1e-9):
                    failures.append(
                        f"dow_mean_{k} h={row + 1} at {origin.date()}: "
                        f"got {got}, want {want}"
                    )

        # horizon must be the real day gap, 1..h
        if list(feats["horizon"]) != list(range(1, len(targets) + 1)):
            failures.append(f"horizon at {origin.date()}: {list(feats['horizon'])}")
        checked += 1

    return {
        "check": "feature spot-checks",
        "tested": f"{checked} feature values against date-filtered recomputation",
        "failures": failures[:8],
        "verdict": "ok" if not failures else "FAILED",
    }


# --------------------------------------------------------------------------- #
# 3. Noise floor and shuffled target
# --------------------------------------------------------------------------- #


def check_noise_floor(noise_sd: float = 2.0) -> dict:
    """
    Model error must not fall below the noise that was injected.

    No honest forecaster can beat noise you generated yourself. Error
    meaningfully under the floor means information is arriving from the future.
    """
    cfg = Config(n_folds=10, min_train_days=200)
    observed, _, n = _walk_forward_rmse(_synthetic_panel(noise_sd=noise_sd), cfg)
    ratio = observed / noise_sd

    # ~420 scored points, so the RMSE's own sampling error is a few percent.
    # 0.90 is roughly three standard errors below the floor.
    return {
        "check": "noise floor",
        "tested": f"{n} fold-series windows on synthetic data",
        "floor_rmse": round(noise_sd, 3),
        "observed_rmse": round(observed, 3),
        "ratio": round(ratio, 3),
        "verdict": "LEAK SUSPECTED" if ratio < 0.90 else "ok",
    }


def check_shuffled_target(seed: int = 7) -> dict:
    """
    With the target scrambled, no model may beat a mean-predicting benchmark.

    The benchmark must be mean-based. Against seasonal naive this test fires on
    a perfectly healthy pipeline: seasonal naive stakes everything on one past
    day, so on noise its error is about sqrt(2) worse than the mean, and any
    model that predicts near the mean beats it without skill.
    """
    rng = np.random.default_rng(seed)
    panel = _synthetic_panel(seed=seed)
    panel["sales"] = (
        panel.groupby("id", observed=True)["sales"]
        .transform(lambda s: rng.permutation(s.to_numpy()))
        .to_numpy()
    )

    cfg = Config(n_folds=10, min_train_days=200)
    model, bench, n = _walk_forward_rmse(panel, cfg)
    ratio = model / bench

    # Model should be no better than the mean - ratio at or above 1.0.
    return {
        "check": "shuffled target",
        "tested": f"{n} fold-series windows, pattern destroyed",
        "model_rmse": round(model, 3),
        "mean_benchmark_rmse": round(bench, 3),
        "ratio_model_over_benchmark": round(ratio, 3),
        "expected": "at or above 1.0 - nothing left to learn",
        "verdict": "LEAK SUSPECTED" if ratio < 0.95 else "ok",
    }


# --------------------------------------------------------------------------- #
# 4. Determinism
# --------------------------------------------------------------------------- #


def check_determinism(cfg: Config | None = None) -> dict:
    """Identical inputs must produce byte-identical forecasts."""
    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    panel = load_panel(cfg, verbose=False)
    series = panel[panel["id"] == panel["id"].iloc[0]].sort_values("date")
    pool = build_supervised(series, cfg.horizon)
    origin = cfg.fold_origins(panel["date"].max())[-1]

    history = series[series["date"] <= origin]
    targets = series[
        (series["date"] > origin)
        & (series["date"] <= origin + pd.Timedelta(days=cfg.horizon))
    ]
    ctx = Context(
        history=history,
        targets=targets.drop(columns=["sales"]),
        origin=origin,
        horizon=cfg.horizon,
        series_id=series["id"].iloc[0],
        season=cfg.season,
        train_pool=pool,
        seed=cfg.seed,
    )
    a, b = fit_predict_xgboost(ctx), fit_predict_xgboost(ctx)

    return {
        "check": "determinism",
        "tested": "same config, two runs, xgboost",
        "identical": bool(np.array_equal(a, b)),
        "max_difference": float(np.max(np.abs(a - b))),
        "verdict": "ok" if np.array_equal(a, b) else "FAILED",
    }


# --------------------------------------------------------------------------- #
# 5. SNAP flags
# --------------------------------------------------------------------------- #


def check_snap_flags(cfg: Config | None = None) -> dict:
    """
    Every row's SNAP flag must equal the raw calendar's column for that row's
    own state, on every date.

    The calendar carries three columns - snap_CA, snap_TX, snap_WI - and the
    loader picks one per row. The failure this guards against is quiet: if
    every row read snap_CA, the flag would still be 0/1, still fire on about a
    third of days, and every downstream plot would look perfectly plausible.
    Only a date-by-date comparison against the source catches it.

    The days-of-month are reported too, so a human can eyeball that each state
    has its own schedule rather than all three sharing one.
    """
    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    panel = load_panel(cfg, verbose=False)
    calendar = pd.read_csv(cfg.data_dir / "calendar.csv", parse_dates=["date"])

    failures, days_by_state, compared = [], {}, 0
    for state, column in SNAP_BY_STATE.items():
        rows = panel[panel["state_id"] == state].drop_duplicates("date")
        if rows.empty:
            continue
        got = rows.set_index("date")["snap"].astype(int)
        want = calendar.set_index("date")[column].reindex(got.index).astype(int)
        wrong = int((got != want).sum())
        compared += len(got)
        if wrong:
            failures.append(
                f"{state}: {wrong} of {len(got)} dates disagree with {column}"
            )
        days_by_state[state] = sorted(
            int(d) for d in rows.loc[rows["snap"] == 1, "date"].dt.day.unique()
        )

    return {
        "check": "SNAP flags per state",
        "tested": f"{compared:,} store-dates against calendar.csv, {len(days_by_state)} states",
        "snap_days_of_month": days_by_state,
        "failures": failures,
        "verdict": "ok" if not failures else "FAILED",
    }


# --------------------------------------------------------------------------- #


def run_all() -> bool:
    """Run every check. Returns True only if all pass."""
    checks = [
        check_benchmarks_closed_form,
        check_features_hand_computed,
        check_snap_flags,
        check_determinism,
        check_noise_floor,
        check_shuffled_target,
    ]

    ok = True
    for fn in checks:
        result = fn()
        name = result.pop("check")
        print(f"\n--- {name} ---")
        for key, value in result.items():
            if key == "failures" and not value:
                continue
            print(f"  {key}: {value}")
        if "ok" not in str(result.get("verdict", "")):
            ok = False

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILED - see above"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
