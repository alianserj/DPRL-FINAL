from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from lite_genbr.env.grid import GridWorld
from lite_genbr.planning.robot_br import robot_best_response
from lite_genbr.multinash.frechet import discrete_frechet
from lite_genbr.multinash.cptg import (
    CPTGSpec,
    pack_positions,
    unpack_positions,
    potential_cost,
    penalty_cost,
    constraint_violations,
)
from lite_genbr.multinash.implicit_pf import implicit_pf_samples


def _resample_path_to_horizon(path_xy: np.ndarray, T: int) -> np.ndarray:
    """Resample a polyline path (N,2) to exactly (T+1,2) points via arc-length interpolation."""
    P = np.asarray(path_xy, dtype=float)
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-9:
        return np.repeat(P[:1], T + 1, axis=0)

    targets = np.linspace(0.0, total, T + 1)
    out = np.zeros((T + 1, 2), dtype=float)
    j = 0
    for i, tt in enumerate(targets):
        while j < len(s) - 2 and s[j + 1] < tt:
            j += 1
        s0, s1 = s[j], s[j + 1]
        if s1 <= s0 + 1e-12:
            out[i] = P[j]
        else:
            a = (tt - s0) / (s1 - s0)
            out[i] = (1 - a) * P[j] + a * P[j + 1]
    return out


def _grid_path_to_xy(path_rc: List[Tuple[int, int]], cell_size_m: float) -> np.ndarray:
    """Convert grid (r,c) path to continuous xy at cell centers (meters)."""
    xy = []
    for r, c in path_rc:
        xy.append([(c + 0.5) * cell_size_m, (r + 0.5) * cell_size_m])
    return np.asarray(xy, dtype=float)


def _cluster_by_frechet(samples: List[Dict], eps: float, max_solutions: int) -> List[List[int]]:
    """Greedy DBSCAN-like clustering using Fréchet distance on joint trajectories (T+1,4)."""
    idxs = list(range(len(samples)))
    idxs.sort(key=lambda i: (samples[i]["pen_cost"], -samples[i]["weight"]))

    clusters: List[List[int]] = []
    assigned = set()
    for i in idxs:
        if i in assigned:
            continue
        base = [i]
        assigned.add(i)
        Ji = np.concatenate([samples[i]["X"][0], samples[i]["X"][1]], axis=1)

        for j in idxs:
            if j in assigned:
                continue
            Jj = np.concatenate([samples[j]["X"][0], samples[j]["X"][1]], axis=1)
            d = discrete_frechet(Ji, Jj)
            if d <= eps:
                base.append(j)
                assigned.add(j)

        clusters.append(base)
        if len(clusters) >= max_solutions:
            break
    return clusters


def _refine_with_slsqp(
    X0: np.ndarray,
    starts: List[np.ndarray],
    goals: List[np.ndarray],
    risk_map: np.ndarray,
    cell_size_m: float,
    obstacle_boxes_m: List[Tuple[float, float, float, float]],
    world_bounds: Tuple[float, float, float, float],
    spec: CPTGSpec,
    maxiter: int,
    penalty_w: float,
) -> Dict[str, Any]:
    """Refine candidate via constrained optimization (SLSQP) with hard start equalities + bounds."""
    if minimize is None:
        raise ImportError("scipy is required for SLSQP refinement. Install scipy>=1.10")

    T = spec.T
    z0 = pack_positions(X0)

    def eq_start(i: int, dim: int):
        return lambda z: float(unpack_positions(z, T=T)[i, 0, dim] - starts[i][dim])

    eq_cons = [{"type": "eq", "fun": eq_start(0, 0)},
               {"type": "eq", "fun": eq_start(0, 1)},
               {"type": "eq", "fun": eq_start(1, 0)},
               {"type": "eq", "fun": eq_start(1, 1)}]

    xmin, ymin, xmax, ymax = world_bounds
    bounds = []
    for _ in range(2 * (T + 1)):
        bounds.append((xmin, xmax))  # x
        bounds.append((ymin, ymax))  # y

    def objective(z):
        X = unpack_positions(z, T=T)
        return penalty_cost(X, starts, goals, risk_map, cell_size_m, obstacle_boxes_m, world_bounds, spec, penalty_w)

    res = minimize(
        objective,
        z0,
        method="SLSQP",
        constraints=eq_cons,
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-5, "disp": False},
    )
    z = res.x if res.success else z0
    X = unpack_positions(z, T=T)
    base_cost = potential_cost(X, starts, goals, risk_map, cell_size_m, spec)
    viols = constraint_violations(X, obstacle_boxes_m, world_bounds, spec)
    feas = all(float(np.max(v)) <= 1e-3 for v in viols.values())
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "X": X,
        "cost": float(base_cost),
        "feasible": bool(feas),
        "violations": {k: float(np.max(v)) for k, v in viols.items()},
    }


def multinash_pf_solve(
    grid: GridWorld,
    risk_map: np.ndarray,
    obstacle_boxes_m: List[Tuple[float, float, float, float]],
    starts: List[np.ndarray],
    goals: List[np.ndarray],
    cell_size_m: float,
    cptg_params,
    multinash_params,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """MultiNash-PF (lite) end-to-end solver for multiple local CPTG solutions."""
    spec = CPTGSpec(
        dt=float(cptg_params.dt),
        T=int(cptg_params.T),
        v_max=float(cptg_params.v_max),
        d_min=float(cptg_params.d_min),
        robot_radius=float(cptg_params.robot_radius),
        obs_margin=float(cptg_params.obs_margin),
        w_g=float(cptg_params.w_g),
        w_u=float(cptg_params.w_u),
        w_r=float(cptg_params.w_r),
    )
    world_bounds = (0.0, 0.0, grid.W * cell_size_m, grid.H * cell_size_m)

    def xy_to_cell(xy):
        x, y = float(xy[0]), float(xy[1])
        c = int(np.clip(np.floor(x / cell_size_m), 0, grid.W - 1))
        r = int(np.clip(np.floor(y / cell_size_m), 0, grid.H - 1))
        return (r, c)

    ref_paths_xy = []
    for i in range(2):
        path = robot_best_response(
            grid=grid,
            start=xy_to_cell(starts[i]),
            goal=xy_to_cell(goals[i]),
            expected_risk_map=risk_map,
            risk_w=1.0,
            allow_diagonal=True,
        )
        xy = _grid_path_to_xy(path, cell_size_m)
        xy = _resample_path_to_horizon(xy, T=spec.T)
        ref_paths_xy.append(xy)

    X_ref = np.stack(ref_paths_xy, axis=0)

    samples = implicit_pf_samples(
        X_ref=X_ref,
        starts=starts,
        goals=goals,
        risk_map=risk_map,
        cell_size_m=cell_size_m,
        obstacle_boxes_m=obstacle_boxes_m,
        world_bounds=world_bounds,
        spec=spec,
        J=int(multinash_params.J),
        noise_sigma=float(multinash_params.noise_sigma),
        refine_iters=int(multinash_params.pf_refine_iters),
        penalty_w=float(multinash_params.penalty_w),
        rng=rng,
    )

    clusters = _cluster_by_frechet(samples, eps=float(multinash_params.cluster_eps), max_solutions=int(multinash_params.max_solutions))

    solutions: List[Dict[str, Any]] = []
    for cid, members in enumerate(clusters):
        ws = np.array([samples[i]["weight"] for i in members], dtype=float)
        ws = ws / max(1e-12, ws.sum())
        X_mean = np.zeros_like(samples[members[0]]["X"])
        for w, idx in zip(ws, members):
            X_mean += float(w) * samples[idx]["X"]

        refined = _refine_with_slsqp(
            X0=X_mean,
            starts=starts,
            goals=goals,
            risk_map=risk_map,
            cell_size_m=cell_size_m,
            obstacle_boxes_m=obstacle_boxes_m,
            world_bounds=world_bounds,
            spec=spec,
            maxiter=int(multinash_params.solver_maxiter),
            penalty_w=float(multinash_params.penalty_w),
        )
        X = refined["X"]
        solutions.append(
            {
                "cluster_id": int(cid),
                "cluster_size": int(len(members)),
                "cost": float(refined["cost"]),
                "feasible": bool(refined["feasible"]),
                "violations": refined["violations"],
                "success": bool(refined["success"]),
                "message": refined["message"],
                "traj_robot_1": X[0].tolist(),
                "traj_robot_2": X[1].tolist(),
            }
        )

    solutions.sort(key=lambda s: (not s["feasible"], s["cost"]))
    return solutions
