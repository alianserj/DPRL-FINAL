from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def discrete_frechet(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """Discrete Fréchet distance between two trajectories.

    traj_a: (N,D), traj_b: (M,D). Uses standard DP recursion on Euclidean distances.
    This is the same logic as your snippet, just using cdist for speed.
    """
    A = np.asarray(traj_a, dtype=float)
    B = np.asarray(traj_b, dtype=float)
    n = A.shape[0]
    m = B.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    dist = cdist(A, B, metric="euclidean")  # (n,m)
    ca = np.full((n, m), -1.0, dtype=float)

    ca[0, 0] = dist[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])

    for i in range(1, n):
        # inner loop is tiny (T <= ~40 normally), OK
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), dist[i, j])

    return float(ca[n - 1, m - 1])
