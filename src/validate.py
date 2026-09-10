"""
Leakage tests using synthetic data with a known noise floor.
 
The idea: build a series where you know exactly how much of it is learnable and
how much is random. Any model's error is bounded below by the random part. If the
pipeline reports an error *under* that floor, it is reading the future - no
amount of clever modelling can beat noise you generated yourself.
 
Run this after any change to features, splitting, or evaluation:
 
    python -m src.validate
 
Two tests:
  1. Noise floor  - error must not fall below the noise you injected.
  2. Shuffled target - with the pattern destroyed, the model must not beat a
     baseline. If it does, information is leaking from somewhere.
"""
 
from __future__ import annotations
 
import numpy as np
import pandas as pd
 
from src.experiment import run_walk_forward
from src.features import make_features
 
NOISE_SD = 2.0
 
 
def make_synthetic_panel(
    n_series: int = 3,
    n_days: int = 900,
    noise_sd: float = NOISE_SD,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a panel shaped like the real one, but where we know the answer.
 
    sales = base + weekly pattern + slow trend + noise(0, noise_sd)
 
    Everything except the noise is a deterministic function of the date, so a
    perfect model would predict it exactly and be left with only the noise. That
    makes `noise_sd` the theoretical best RMSE.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-01", periods=n_days, freq="D")
 
    frames = []
    for k in range(n_series):
        base = 10 + 5 * k
        weekly = 3 * np.sin(2 * np.pi * dates.dayofweek / 7)
        trend = np.linspace(0, 2, n_days)
        signal = base + weekly + trend
        noise = rng.normal(0, noise_sd, n_days)
 
        frames.append(pd.DataFrame({
            "id": f"SYNTH_{k:03d}",
            "item_id": f"SYNTH_{k:03d}",
            "dept_id": "SYNTH",
            "cat_id": "SYNTH",
            "store_id": "S1",
            "state_id": "CA",
            "date": dates,
            "sales": np.maximum(signal + noise, 0),
            "sell_price": 5.0,
            "snap": 0,
            "event_name_1": pd.Series([None] * n_days, dtype=object),
            "wm_yr_wk": 11101 + np.arange(n_days) // 7,
        }))
 
    return pd.concat(frames, ignore_index=True)
 
 
def test_noise_floor(horizon: int = 7, n_folds: int = 6) -> dict:
    """Model RMSE must not drop below the injected noise level."""
    panel = make_synthetic_panel()
    feats = make_features(panel, horizon=horizon)
    res = run_walk_forward(feats, horizon=horizon, n_folds=n_folds, min_train_days=180)
 
    observed = res["model_rmse"].mean()
    floor = NOISE_SD
    # Allow a little slack for sampling variation on short test windows.
    leaking = observed < floor * 0.75
 
    return {
        "test": "noise_floor",
        "theoretical_floor_rmse": round(floor, 3),
        "observed_model_rmse": round(observed, 3),
        "ratio": round(observed / floor, 3),
        "verdict": "LEAK SUSPECTED" if leaking else "ok",
    }
 
 
def test_shuffled_target(horizon: int = 7, n_folds: int = 6, seed: int = 1) -> dict:
    """
    With the target shuffled there is no learnable pattern left, so the model
    should not systematically beat the baseline. A high win rate here means the
    model is getting information it should not have.
 
    The baseline must be `moving_average`, not `seasonal_naive`. With no signal,
    the best possible prediction is the mean, and moving_average predicts the
    mean. seasonal_naive predicts a single random past value, whose error is
    larger by a factor of sqrt(2) - so the model would "win" against it every
    time by being sensibly dumb, and the test would fire on a pipeline that is
    perfectly fine. That is a flaw in the test, not evidence of a leak.
    """
    rng = np.random.default_rng(seed)
    panel = make_synthetic_panel()
    panel["sales"] = rng.permutation(panel["sales"].to_numpy())
 
    feats = make_features(panel, horizon=horizon)
    res = run_walk_forward(
        feats, horizon=horizon, n_folds=n_folds,
        min_train_days=180, baseline_name="moving_average",
    )
 
    win_rate = res["won"].mean() * 100
    leaking = win_rate > 75
 
    return {
        "test": "shuffled_target",
        "win_rate_pct": round(win_rate, 1),
        "baseline": "moving_average (predicts the mean)",
        "expected": "roughly 50, no better than chance",
        "verdict": "LEAK SUSPECTED" if leaking else "ok",
    }
 
 
def run_all() -> bool:
    """Run every check. Returns True if all pass."""
    results = [test_noise_floor(), test_shuffled_target()]
    ok = True
    for r in results:
        print(f"\n--- {r.pop('test')} ---")
        for k, v in r.items():
            print(f"  {k}: {v}")
        if "LEAK" in str(r.get("verdict", "")):
            ok = False
    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILED - see above"))
    return ok
 
 
if __name__ == "__main__":
    run_all()