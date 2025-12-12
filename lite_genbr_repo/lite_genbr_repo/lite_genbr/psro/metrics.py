from __future__ import annotations
from typing import Callable
import numpy as np


def nashconv_zero_sum(
    payoff: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    row_br_value_fn: Callable[[np.ndarray], float],
    col_br_value_fn: Callable[[np.ndarray], float],
) -> float:
    """
    NashConv / exploitability for a two-player zero-sum matrix game.

    payoff[i,j] is row payoff; row maximizes, column minimizes.
    """
    v = float(p @ payoff @ q)
    row_regret = float(row_br_value_fn(q) - v)
    col_regret = float(v - col_br_value_fn(p))
    return float(max(0.0, row_regret) + max(0.0, col_regret))
