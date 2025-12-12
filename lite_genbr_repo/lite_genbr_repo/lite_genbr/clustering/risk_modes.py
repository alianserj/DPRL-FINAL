from __future__ import annotations

from typing import Dict, List
import numpy as np

from lite_genbr.env.grid import GridWorld
from lite_genbr.sensors.sensor_policy import SensorPolicy


def _jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 0.0 if union <= 1e-9 else (1.0 - inter / union)


def cluster_sensor_mixture(
    grid: GridWorld,
    sensor_population: List[SensorPolicy],
    q: np.ndarray,
    L: int,
    max_iters: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Cluster sensor configs into L "risk modes" based on union-visibility similarity (Jaccard).

    Returns list of dicts:
      { weight, members, representative, risk_map }
    """
    H, W = grid.H, grid.W
    q = np.asarray(q, dtype=float)

    # Build union masks
    masks = []
    for sp in sensor_population:
        if sp.exposure_map is None:
            raise ValueError("SensorPolicy.exposure_map is None. Compute it before clustering.")
        masks.append(sp.exposure_map > 0)
    masks = np.stack(masks, axis=0)
    M = masks.shape[0]

    if M <= 1:
        return [{
            "weight": float(q[0]) if len(q) else 1.0,
            "members": [0],
            "representative": 0,
            "risk_map": sensor_population[0].exposure_map.astype(float),
        }]

    # Pairwise distances
    D = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(i + 1, M):
            d = _jaccard_distance(masks[i], masks[j])
            D[i, j] = D[j, i] = d

    # Init medoids by farthest point
    medoids = [int(rng.integers(0, M))]
    while len(medoids) < min(L, M):
        best_idx, best_val = None, -1.0
        for i in range(M):
            dmin = min(D[i, m] for m in medoids)
            if dmin > best_val:
                best_val = dmin
                best_idx = i
        medoids.append(int(best_idx))

    # k-medoids refine
    for _ in range(max_iters):
        assign = np.zeros(M, dtype=int)
        for i in range(M):
            assign[i] = int(np.argmin([D[i, m] for m in medoids]))

        new_medoids = []
        for k in range(len(medoids)):
            members = np.where(assign == k)[0].tolist()
            if not members:
                new_medoids.append(medoids[k])
                continue
            best_m, best_cost = None, float("inf")
            for m in members:
                cost = 0.0
                for j in members:
                    cost += float(q[j]) * D[m, j]
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
            new_medoids.append(int(best_m))
        if new_medoids == medoids:
            break
        medoids = new_medoids

    # Final assign
    assign = np.zeros(M, dtype=int)
    for i in range(M):
        assign[i] = int(np.argmin([D[i, m] for m in medoids]))

    modes: List[Dict] = []
    for k, med in enumerate(medoids):
        members = np.where(assign == k)[0].tolist()
        if not members:
            continue
        weight = float(q[members].sum())
        if weight <= 1e-12:
            continue
        risk = np.zeros((H, W), dtype=float)
        for j in members:
            risk += float(q[j]) * sensor_population[j].exposure_map.astype(float)
        risk = risk / weight
        modes.append({"weight": weight, "members": members, "representative": int(med), "risk_map": risk})

    modes.sort(key=lambda d: -float(d["weight"]))
    return modes[:L]
