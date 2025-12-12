from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover
    distance_transform_edt = None

try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:  # pragma: no cover
    AgglomerativeClustering = None

from lite_genbr.env.grid import GridWorld
from lite_genbr.planning.robot_br import robot_best_response
from lite_genbr.multinash.frechet import discrete_frechet
from lite_genbr.multinash.cptg import (
    CPTGSpec,
    pack_positions,
    unpack_positions,
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


def _compute_obs_dist_field_m(grid: GridWorld, cell_size_m: float) -> Optional[np.ndarray]:
    """Distance-to-nearest-obstacle field in meters (fast obstacle distance queries)."""
    if distance_transform_edt is None:
        return None
    obs = np.asarray(grid.obstacles, dtype=bool)
    free = ~obs
    dist_cells = distance_transform_edt(free)
    return dist_cells.astype(float) * float(cell_size_m)
def _bilinear_single(field: np.ndarray, x: float, y: float, cell_size_m: float) -> float:
    H, W = field.shape
    col_f = x / float(cell_size_m) - 0.5
    row_f = y / float(cell_size_m) - 0.5

    c0 = int(np.floor(col_f)); r0 = int(np.floor(row_f))
    c1 = min(max(c0 + 1, 0), W - 1)
    r1 = min(max(r0 + 1, 0), H - 1)
    c0 = min(max(c0, 0), W - 1)
    r0 = min(max(r0, 0), H - 1)

    dc = col_f - c0
    dr = row_f - r0

    v00 = float(field[r0, c0]); v01 = float(field[r0, c1])
    v10 = float(field[r1, c0]); v11 = float(field[r1, c1])

    v0 = v00 * (1 - dc) + v01 * dc
    v1 = v10 * (1 - dc) + v11 * dc
    return float(v0 * (1 - dr) + v1 * dr)


def _obs_grad(field: np.ndarray, p: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Finite-diff gradient of obstacle distance field at p (meters)."""
    eps = float(cell_size_m)
    x = float(p[0]); y = float(p[1])
    fx1 = _bilinear_single(field, x + eps, y, cell_size_m)
    fx0 = _bilinear_single(field, x - eps, y, cell_size_m)
    fy1 = _bilinear_single(field, x, y + eps, cell_size_m)
    fy0 = _bilinear_single(field, x, y - eps, cell_size_m)
    gx = (fx1 - fx0) / (2 * eps)
    gy = (fy1 - fy0) / (2 * eps)
    g = np.array([gx, gy], dtype=float)
    n = float(np.linalg.norm(g))
    if n <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return g / n


def _fast_project_refine(
    X: np.ndarray,
    *,
    starts: List[np.ndarray],
    goals: List[np.ndarray],
    world_bounds: Tuple[float, float, float, float],
    spec: CPTGSpec,
    obstacle_boxes_m: List[Tuple[float, float, float, float]],
    obs_dist_field_m: Optional[np.ndarray],
    cell_size_m: float,
    iters: int = 30,
) -> np.ndarray:
    """Very fast feasibility repair (no NLP).

    - Pushes points out of obstacles using distance-field gradient (or box distance fallback)
    - Separates robots if too close
    - Enforces speed bound by clipping step lengths
    - Keeps start and goal fixed
    """
    X = np.asarray(X, dtype=float).copy()
    T = int(spec.T)
    dt = float(spec.dt)
    v_step = float(spec.v_max) * dt
    d_min = float(spec.d_min)
    xmin, ymin, xmax, ymax = world_bounds
    clearance = float(spec.robot_radius + spec.obs_margin)

    X[0, 0] = starts[0]
    X[1, 0] = starts[1]
    X[0, T] = goals[0]
    X[1, T] = goals[1]

    def obs_dist(p: np.ndarray) -> float:
        if obs_dist_field_m is not None:
            return _bilinear_single(obs_dist_field_m, float(p[0]), float(p[1]), float(cell_size_m))
        # fallback: coarse box distance
        # (imported in implicit_pf; we avoid extra import here, keep simple)
        # use min over boxes (slow but only for rare fallback)
        d = float("inf")
        x, y = float(p[0]), float(p[1])
        for (bx0, by0, bx1, by1) in obstacle_boxes_m:
            dx = max(bx0 - x, 0.0, x - bx1)
            dy = max(by0 - y, 0.0, y - by1)
            d = min(d, float(np.hypot(dx, dy)))
        return d

    for _ in range(int(iters)):
        # obstacle push + bounds (skip fixed endpoints)
        for i in range(2):
            for t in range(1, T):
                p = X[i, t]
                # bounds clamp
                p[0] = float(np.clip(p[0], xmin, xmax))
                p[1] = float(np.clip(p[1], ymin, ymax))

                d = obs_dist(p)
                if d < clearance:
                    if obs_dist_field_m is not None:
                        g = _obs_grad(obs_dist_field_m, p, float(cell_size_m))
                    else:
                        # crude direction away from center of nearest box: approximate with x-axis
                        g = np.array([1.0, 0.0], dtype=float)
                    p[:] = p + (clearance - d + 1e-3) * g

        # collision separation
        for t in range(1, T):
            p1 = X[0, t]
            p2 = X[1, t]
            d = float(np.linalg.norm(p1 - p2))
            if d < d_min:
                dir = (p1 - p2)
                n = float(np.linalg.norm(dir))
                if n <= 1e-12:
                    dir = np.array([1.0, 0.0], dtype=float)
                    n = 1.0
                dir = dir / n
                push = 0.5 * (d_min - d + 1e-3)
                X[0, t] = p1 + push * dir
                X[1, t] = p2 - push * dir

        # speed forward pass
        for i in range(2):
            for t in range(1, T + 1):
                if t == T:
                    # keep goal fixed
                    continue
                prev = X[i, t - 1]
                cur = X[i, t]
                step = cur - prev
                n = float(np.linalg.norm(step))
                if n > v_step:
                    X[i, t] = prev + (v_step / n) * step

        # speed backward pass (keep start fixed)
        for i in range(2):
            for t in range(T - 1, -1, -1):
                if t == 0:
                    continue
                nxt = X[i, t + 1]
                cur = X[i, t]
                step = cur - nxt
                n = float(np.linalg.norm(step))
                if n > v_step:
                    X[i, t] = nxt + (v_step / n) * step

        # re-impose endpoints
        X[0, 0] = starts[0]
        X[1, 0] = starts[1]
        X[0, T] = goals[0]
        X[1, T] = goals[1]

    return X




def _xy_to_cell(xy: np.ndarray, grid: GridWorld, cell_size_m: float) -> Tuple[int, int]:
    x, y = float(xy[0]), float(xy[1])
    c = int(np.clip(np.floor(x / cell_size_m), 0, grid.W - 1))
    r = int(np.clip(np.floor(y / cell_size_m), 0, grid.H - 1))
    return (r, c)


def _cell_to_xy(cell: Tuple[int, int], cell_size_m: float) -> np.ndarray:
    r, c = cell
    return np.array([(c + 0.5) * cell_size_m, (r + 0.5) * cell_size_m], dtype=float)


def _refine_with_slsqp(
    X0: np.ndarray,
    starts: List[np.ndarray],
    goals: List[np.ndarray],
    risk_map: np.ndarray,
    cell_size_m: float,
    obstacle_boxes_m: List[Tuple[float, float, float, float]],
    world_bounds: Tuple[float, float, float, float],
    spec: CPTGSpec,
    penalty_w: float,
    obs_dist_field_m: Optional[np.ndarray],
    solver_maxiter: int,
    *,
    knot_stride: int = 3,
) -> np.ndarray:
    """Fast refinement using SLSQP on a reduced knot set (big speedup).

    We optimize only every `knot_stride` timesteps and linearly interpolate between knots.
    This keeps behavior similar to full-horizon SLSQP but is much faster on Windows.
    """
    if minimize is None or solver_maxiter <= 0:
        return X0

    T = int(spec.T)
    stride = max(1, int(knot_stride))

    # knot times excluding 0, always include T
    knots = list(range(1, T + 1, stride))
    if knots[-1] != T:
        knots.append(T)
    K = len(knots)

    xmin, ymin, xmax, ymax = world_bounds

    # decision vars: [robot0_knot_xy..., robot1_knot_xy...] flattened
    z0 = []
    for i in range(2):
        for t in knots:
            z0.extend([float(X0[i, t, 0]), float(X0[i, t, 1])])
    z0 = np.asarray(z0, dtype=float)

    bounds = []
    for _ in range(2 * K):
        bounds.append((xmin, xmax))  # x
        bounds.append((ymin, ymax))  # y

    def unpack_knots(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(2, K, 2)
        X = np.zeros((2, T + 1, 2), dtype=float)
        X[:, 0, :] = np.stack([starts[0], starts[1]], axis=0)
        # set knots
        for i in range(2):
            for kk, t in enumerate(knots):
                X[i, t, :] = z[i, kk, :]
        # enforce terminal exactly at goals
        X[0, T, :] = goals[0]
        X[1, T, :] = goals[1]
        # linear interpolation between consecutive knots (including t=0)
        for i in range(2):
            prev_t = 0
            prev_p = X[i, 0, :].copy()
            for kk, t in enumerate(knots):
                p = X[i, t, :]
                dt_seg = max(1, t - prev_t)
                for tt in range(prev_t + 1, t):
                    a = (tt - prev_t) / dt_seg
                    X[i, tt, :] = (1 - a) * prev_p + a * p
                prev_t = t
                prev_p = p.copy()
        return X

    def f(z: np.ndarray) -> float:
        X = unpack_knots(z)
        return float(
            penalty_cost(
                X,
                starts,
                goals,
                risk_map,
                cell_size_m,
                obstacle_boxes_m,
                world_bounds,
                spec,
                penalty_w,
                obs_dist_field_m=obs_dist_field_m,
            )
        )

    res = minimize(
        f,
        z0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": int(solver_maxiter), "ftol": 1e-4, "disp": False},
    )
    z = res.x if getattr(res, "success", False) else z0
    return unpack_knots(z)


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
    """MultiNash-PF implementation aligned with your provided logic:

    1) Generate J coarse trajectories using a PF-style UKF update (implicit PF).
    2) Cluster those coarse trajectories by Fréchet distance (Agglomerative).
    3) For each cluster: take mean trajectory + refine with local NLP solver (SLSQP).
    """
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

    # Fast obstacle distance field (big speedup vs iterating over many boxes)
    obs_dist_field_m = _compute_obs_dist_field_m(grid, cell_size_m)

    # ----- Reference (cheap) -----
    X_ref = np.zeros((2, spec.T + 1, 2), dtype=float)
    for i in range(2):
        s_cell = _xy_to_cell(starts[i], grid, cell_size_m)
        g_cell = _xy_to_cell(goals[i], grid, cell_size_m)
        path_cells = robot_best_response(
            grid=grid,
            start=s_cell,
            goal=g_cell,
            expected_risk_map=risk_map,
            risk_w=float(cptg_params.w_r),
            allow_diagonal=True,
        )
        path_xy = np.array([_cell_to_xy(c, cell_size_m) for c in path_cells], dtype=float)
        X_ref[i] = _resample_path_to_horizon(path_xy, spec.T)
        X_ref[i, 0] = starts[i]
        X_ref[i, -1] = goals[i]

    # ----- PF / coarse proposals -----
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
        refine_iters=int(getattr(multinash_params, "pf_refine_iters", 0)),
        penalty_w=float(multinash_params.penalty_w),
        rng=rng,
        obs_dist_field_m=obs_dist_field_m,
    )

    # ----- clustering (Fréchet) -----
    J = len(samples)
    # joint trajectory feature: [x1,y1,x2,y2]
    joint_xy = np.zeros((J, spec.T + 1, 4), dtype=float)
    for j in range(J):
        X = samples[j]["X"]
        joint_xy[j, :, 0:2] = X[0]
        joint_xy[j, :, 2:4] = X[1]

    stride = int(getattr(multinash_params, "frechet_stride", 2))
    stride = max(1, stride)
    joint_xy_s = joint_xy[:, ::stride, :]

    # pairwise distance matrix
    dist = np.zeros((J, J), dtype=float)
    for i in range(J):
        for k in range(i + 1, J):
            d = discrete_frechet(joint_xy_s[i], joint_xy_s[k])
            dist[i, k] = d
            dist[k, i] = d

    if AgglomerativeClustering is not None:
        n_clusters = int(getattr(multinash_params, "n_clusters", min(int(multinash_params.max_solutions), J)))
        n_clusters = max(1, min(n_clusters, J))
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        labels = clustering.fit_predict(dist)
    else:
        # fallback: threshold clustering
        eps = float(getattr(multinash_params, "cluster_eps", 1.5))
        labels = -np.ones(J, dtype=int)
        cid = 0
        for i in range(J):
            if labels[i] >= 0:
                continue
            labels[i] = cid
            for k in range(i + 1, J):
                if dist[i, k] <= eps:
                    labels[k] = cid
            cid += 1

    # ----- refine each cluster mean -----
    solutions: List[Dict[str, Any]] = []
    solver_maxiter = int(multinash_params.solver_maxiter)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        ws = np.array([samples[i]["weight"] for i in idx], dtype=float)
        ws = ws / max(1e-12, ws.sum())

        X_mean = np.zeros_like(samples[idx[0]]["X"])
        for w, i in zip(ws, idx):
            X_mean += float(w) * samples[i]["X"]

        # Fast feasibility repair first (very cheap)
        X_star = _fast_project_refine(
            X_mean,
            starts=starts,
            goals=goals,
            world_bounds=world_bounds,
            spec=spec,
            obstacle_boxes_m=obstacle_boxes_m,
            obs_dist_field_m=obs_dist_field_m,
            cell_size_m=cell_size_m,
            iters=int(getattr(multinash_params, "project_iters", 25)),
        )

        # Optional polish (only if still violating constraints)
        viol_tmp = constraint_violations(
            X_star,
            obstacle_boxes_m=obstacle_boxes_m,
            world_bounds=world_bounds,
            spec=spec,
            obs_dist_field_m=obs_dist_field_m,
            cell_size_m=cell_size_m,
        )
        if any(float(np.max(v)) > 1e-3 for v in viol_tmp.values()):
            maxiter_eff = min(
                int(multinash_params.solver_maxiter),
                int(getattr(multinash_params, "solver_maxiter_cap", 60)),
            )
            X_star = _refine_with_slsqp(
                X0=X_star,
                starts=starts,
                goals=goals,
                risk_map=risk_map,
                cell_size_m=cell_size_m,
                obstacle_boxes_m=obstacle_boxes_m,
                world_bounds=world_bounds,
                spec=spec,
                penalty_w=float(multinash_params.penalty_w),
                obs_dist_field_m=obs_dist_field_m,
                solver_maxiter=maxiter_eff,
                knot_stride=int(getattr(multinash_params, "refine_stride", 3)),
            )

        viol = constraint_violations(
            X_star,
            obstacle_boxes_m=obstacle_boxes_m,
            world_bounds=world_bounds,
            spec=spec,
            obs_dist_field_m=obs_dist_field_m,
            cell_size_m=cell_size_m,
        )
        feasible = all(float(np.max(v)) <= 1e-3 for v in viol.values())

        cost = float(
            penalty_cost(
                X_star,
                starts,
                goals,
                risk_map,
                cell_size_m,
                obstacle_boxes_m,
                world_bounds,
                spec,
                float(multinash_params.penalty_w),
                obs_dist_field_m=obs_dist_field_m,
            )
        )

        solutions.append(
            {
                "traj_robot_1": X_star[0].tolist(),
                "traj_robot_2": X_star[1].tolist(),
                "cost": cost,
                "feasible": bool(feasible),
                "violations": {k: float(np.max(np.asarray(v))) for k, v in viol.items()},
                "cluster_size": int(len(idx)),
            }
        )

    # sort and trim
    solutions.sort(key=lambda d: float(d["cost"]))
    return solutions[: int(multinash_params.max_solutions)]
