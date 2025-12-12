from __future__ import annotations
import numpy as np


def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Fréchet distance between two polylines P and Q."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0, dtype=float)

    def c(i: int, j: int) -> float:
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = float(np.linalg.norm(P[i] - Q[j]))
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), d)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    return c(n - 1, m - 1)
