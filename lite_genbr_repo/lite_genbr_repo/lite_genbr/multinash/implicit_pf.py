from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from lite_genbr.multinash.cptg import CPTGSpec, pack_positions, unpack_positions, penalty_cost


def implicit_pf_samples(
    X_ref: np.ndarray,
    starts: List[np.ndarray],
    goals: List[np.ndarray],
    risk_map: np.ndarray,
    cell_size_m: float,
    obstacle_boxes_m: List[Tuple[float, float, float, float]],
    world_bounds: Tuple[float, float, float, float],
    spec: CPTGSpec,
    J: int,
    noise_sigma: float,
    refine_iters: int,
    penalty_w: float,
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Generate coarse multi-modal samples around a reference trajectory.
    Uses a short L-BFGS-B run on a penalty objective as a cheap 'implicit' refinement.
    """
    if minimize is None:
        raise ImportError("scipy is required for implicit_pf_samples. Install scipy>=1.10")

    T = spec.T
    samples: List[Dict] = []
    beta = 1.0 / max(1e-9, float(np.std(risk_map) + 1.0))

    for _ in range(int(J)):
        X = np.array(X_ref, dtype=float, copy=True)
        noise = rng.normal(scale=float(noise_sigma), size=X.shape)
        noise[:, 0, :] = 0.0  # keep starts fixed
        X = X + noise

        z0 = pack_positions(X)

        def f(z):
            Xz = unpack_positions(z, T=T)
            return penalty_cost(
                Xz, starts, goals, risk_map, cell_size_m, obstacle_boxes_m, world_bounds, spec, penalty_w
            )

        res = minimize(f, z0, method="L-BFGS-B", options={"maxiter": int(refine_iters), "ftol": 1e-6})
        z = res.x if res.success else z0
        X_refined = unpack_positions(z, T=T)
        pcost = float(f(z))
        w = float(np.exp(-beta * pcost))
        samples.append({"X": X_refined, "pen_cost": pcost, "weight": w})

    wsum = sum(s["weight"] for s in samples)
    if wsum > 0:
        for s in samples:
            s["weight"] /= wsum
    return samples
