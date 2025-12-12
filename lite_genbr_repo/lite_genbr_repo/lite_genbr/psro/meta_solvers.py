from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    linprog = None


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = x / max(1e-9, float(tau))
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def project_to_simplex_with_floor(x: np.ndarray, floor: float) -> np.ndarray:
    """Project onto {p: p_i >= floor, sum p_i = 1} by clip + renorm."""
    n = len(x)
    x = np.asarray(x, dtype=float)
    floor = float(floor)
    x = np.maximum(x, floor)
    s = float(np.sum(x))
    if s <= 0:
        return np.ones(n) / n
    x = x / s
    x = np.maximum(x, floor)
    x = x / float(np.sum(x))
    return x


def projected_replicator_dynamics(
    payoff: np.ndarray,
    steps: int = 200,
    dt: float = 0.2,
    gamma: float = 0.05,
    rng: 'Optional[np.random.Generator]' = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PRD for two-player zero-sum where row maximizes payoff and column minimizes.
    Column dynamics run on column payoff = -payoff.
    """
    n, m = payoff.shape
    if rng is None:
        rng = np.random.default_rng(0)

    p = rng.random(n); p = p / p.sum()
    q = rng.random(m); q = q / q.sum()

    floor_p = gamma / max(1, n)
    floor_q = gamma / max(1, m)

    for _ in range(int(steps)):
        v = float(p @ payoff @ q)
        row_pay = payoff @ q
        col_pay = -(p @ payoff)  # payoff for column to maximize

        dp = p * (row_pay - v)
        dq = q * (col_pay - float(q @ col_pay))

        p = p + float(dt) * dp
        q = q + float(dt) * dq

        p = project_to_simplex_with_floor(p, floor=floor_p)
        q = project_to_simplex_with_floor(q, floor=floor_q)

    return p, q


def solve_zero_sum_lp(payoff: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve a finite zero-sum matrix game (row max, column min) using LP via SciPy.
    """
    if linprog is None:
        raise ImportError("scipy is required for LP solve. Install scipy>=1.10")

    n, m = payoff.shape
    # Shift for numeric stability
    min_pay = float(np.min(payoff))
    shift = (-min_pay + 1.0) if min_pay < 0 else 0.0
    A = payoff + shift

    # Row LP: maximize v s.t. A^T p >= v, sum p=1, p>=0
    c = np.zeros(n + 1)
    c[-1] = -1.0  # minimize -v
    A_ub = np.zeros((m, n + 1))
    A_ub[:, :n] = -A.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(m)

    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP solve failed: {res.message}")
    p = np.asarray(res.x[:n], dtype=float)
    p = p / p.sum()

    # Column LP: minimize u s.t. A q <= u, sum q=1, q>=0
    c2 = np.zeros(m + 1); c2[-1] = 1.0
    A_ub2 = np.zeros((n, m + 1))
    A_ub2[:, :m] = A
    A_ub2[:, -1] = -1.0
    b_ub2 = np.zeros(n)
    A_eq2 = np.zeros((1, m + 1)); A_eq2[0, :m] = 1.0
    b_eq2 = np.array([1.0])
    bounds2 = [(0.0, 1.0)] * m + [(None, None)]
    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
    if not res2.success:
        raise RuntimeError(f"LP solve (column) failed: {res2.message}")
    q = np.asarray(res2.x[:m], dtype=float)
    q = q / q.sum()

    v = float(p @ payoff @ q)
    return p, q, v


@dataclass
class SmoothFictitiousPlayState:
    counts_p: np.ndarray
    counts_q: np.ndarray


def smooth_fictitious_play_init(n: int, m: int) -> SmoothFictitiousPlayState:
    return SmoothFictitiousPlayState(counts_p=np.zeros(n, dtype=float), counts_q=np.zeros(m, dtype=float))


def smooth_fictitious_play_update(state: SmoothFictitiousPlayState, br_i: int, br_j: int, alpha: float = 1.0) -> None:
    state.counts_p[br_i] += float(alpha)
    state.counts_q[br_j] += float(alpha)


def smooth_fictitious_play_distribution(state: SmoothFictitiousPlayState, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    p = softmax(state.counts_p, tau=tau)
    q = softmax(state.counts_q, tau=tau)
    return p, q
