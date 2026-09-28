"""
Croston's method and its two relatives, the benchmarks for series with many
zero days (FPP §13.2).

Croston splits the history into two series: the size of each sale and the
gap since the one before. Each is smoothed on its own, and the forecast is
the ratio, sale size over gap, which is a rate in units per day. The book's
worked example (sizes 1, 2, 1 at gaps of 4, alpha 0.1) gives 0.27 a day,
and the tests hold the module to it.

Three variants, all the same shape, none a statistical model. The book says
so and notes the consequence, that no prediction intervals exist. The order
step gets its intervals from each method's own past errors instead.

  croston   the ratio as Croston (1972) wrote it
  sba       the ratio times (1 - alpha / 2), the Syntetos-Boylan correction
            for the ratio's upward bias
  tsb       Teunter, Syntetos and Babai (2011): the chance of a sale is
            smoothed every day, sold or not, so the forecast decays when an
            item stops selling. Croston's cannot, since it only updates on a
            sale.

One smoothing weight for everything, the book's classic setting. Optimising
it is a small gain for a fitted parameter per series, and these are the
floor, not the contender.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Context, _flat, note_fallback

ALPHA = 0.1


def _croston_rate(y: np.ndarray, alpha: float = ALPHA) -> float | None:
    """Smoothed sale size over smoothed gap. None when there are not two sales."""
    idx = np.flatnonzero(y > 0)
    if len(idx) < 2:
        return None
    sizes = y[idx]
    gaps = np.diff(idx).astype(float)
    q, a = sizes[0], gaps[0]  # initialised at the first size and the first gap
    for size, gap in zip(sizes[1:], gaps, strict=True):
        q = (1 - alpha) * q + alpha * size
        a = (1 - alpha) * a + alpha * gap
    return q / a


def _too_few_sales(ctx: Context, name: str) -> np.ndarray:
    # Under two sales in the history, so there is no gap to smooth. The 28-day
    # mean is the same fallback ETS uses, and it is counted as one.
    note_fallback(f"{name}: 28-day mean")
    return _flat(ctx.y[-28:].mean(), ctx)


def fit_predict_croston(ctx: Context) -> np.ndarray:
    rate = _croston_rate(ctx.y)
    return _too_few_sales(ctx, "croston") if rate is None else _flat(rate, ctx)


def fit_predict_sba(ctx: Context) -> np.ndarray:
    rate = _croston_rate(ctx.y)
    if rate is None:
        return _too_few_sales(ctx, "sba")
    return _flat(rate * (1 - ALPHA / 2), ctx)


def fit_predict_tsb(ctx: Context, alpha: float = ALPHA) -> np.ndarray:
    """
    Chance of a sale times expected size. The chance is smoothed on every day
    (1 on a sale day, 0 otherwise), the size only on sale days. Both start at
    their averages over the history so the recursion has somewhere to begin.
    """
    y = ctx.y
    sold = y > 0
    if sold.sum() < 2:
        return _too_few_sales(ctx, "tsb")
    p = float(sold.mean())
    q = float(y[sold].mean())
    for value, s in zip(y, sold, strict=True):
        p = (1 - alpha) * p + alpha * float(s)
        if s:
            q = (1 - alpha) * q + alpha * value
    return _flat(p * q, ctx)


INTERMITTENT = {
    "croston": fit_predict_croston,
    "sba": fit_predict_sba,
    "tsb": fit_predict_tsb,
}
