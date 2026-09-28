"""
Gradient-boosted trees with a Tweedie objective, the loss most of the top M5
entries trained on.

Tweedie is a distribution with a point mass at zero and a skewed tail above
it, which is what a day of sales looks like for an item that does not sell
every day. Squared error treats a forecast of 2 on a zero day the same as 12
on a day that sold 10; Tweedie does not. The variance power sits between 1
(Poisson) and 2 (gamma); 1.3 is inside the range the M5 write-ups found
best. Same features and settings as the point model, only the loss changes.

It cannot use the level-relative target, since that can be negative, so it
carries the level the way the M5 models did: through the recent-sales
features and, when pooled, the log link, which makes every effect a
multiple of the series' own level. That is the proportional scale item 9
said a pool across items needs.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Context
from src.models.xgboost_model import fit_predict_xgboost

TWEEDIE_VARIANCE_POWER = 1.3


def fit_predict_xgboost_tweedie(ctx: Context, **overrides) -> np.ndarray:
    """The point model's trees under a Tweedie objective."""
    return fit_predict_xgboost(
        ctx,
        objective="reg:tweedie",
        tweedie_variance_power=TWEEDIE_VARIANCE_POWER,
        **overrides,
    )
