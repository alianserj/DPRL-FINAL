from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

Box = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def min_distance_to_boxes(points: np.ndarray, boxes: List[Box]) -> np.ndarray:
    """Vectorized min distance from each point (N,2) to any axis-aligned box. Returns (N,) distances."""
    if not boxes:
        return np.full((points.shape[0],), np.inf, dtype=float)
    P = np.asarray(points, dtype=float)
    B = np.asarray(boxes, dtype=float)  # (M,4)
    x = P[:, 0:1]  # (N,1)
    y = P[:, 1:2]
    xmin = B[None, :, 0]
    ymin = B[None, :, 1]
    xmax = B[None, :, 2]
    ymax = B[None, :, 3]
    dx = np.maximum(xmin - x, 0.0)
    dx = np.maximum(dx, x - xmax)
    dy = np.maximum(ymin - y, 0.0)
    dy = np.maximum(dy, y - ymax)
    d = np.sqrt(dx * dx + dy * dy)  # (N,M)
    return np.min(d, axis=1)


def bilinear_sample(grid: np.ndarray, x: float, y: float, cell_size: float) -> float:
    """Bilinear sampling of a grid (row,col) defined over cell centers in meters."""
    H, W = grid.shape
    col_f = x / cell_size - 0.5
    row_f = y / cell_size - 0.5

    c0 = int(np.floor(col_f)); r0 = int(np.floor(row_f))
    c1 = c0 + 1; r1 = r0 + 1

    def clamp(v, lo, hi): return max(lo, min(hi, v))
    c0 = clamp(c0, 0, W - 1); c1 = clamp(c1, 0, W - 1)
    r0 = clamp(r0, 0, H - 1); r1 = clamp(r1, 0, H - 1)

    dc = col_f - c0
    dr = row_f - r0

    v00 = float(grid[r0, c0]); v01 = float(grid[r0, c1])
    v10 = float(grid[r1, c0]); v11 = float(grid[r1, c1])

    v0 = v00 * (1 - dc) + v01 * dc
    v1 = v10 * (1 - dc) + v11 * dc
    return float(v0 * (1 - dr) + v1 * dr)


def _bilinear_field(field: np.ndarray, xy: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Vectorized bilinear sampling of a field (H,W) at xy points in meters."""
    H, W = field.shape
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]

    col_f = x / float(cell_size_m) - 0.5
    row_f = y / float(cell_size_m) - 0.5

    c0 = np.floor(col_f).astype(int)
    r0 = np.floor(row_f).astype(int)
    c1 = np.clip(c0 + 1, 0, W - 1)
    r1 = np.clip(r0 + 1, 0, H - 1)
    c0 = np.clip(c0, 0, W - 1)
    r0 = np.clip(r0, 0, H - 1)

    dc = (col_f - c0).astype(float)
    dr = (row_f - r0).astype(float)

    v00 = field[r0, c0]
    v01 = field[r0, c1]
    v10 = field[r1, c0]
    v11 = field[r1, c1]

    v0 = v00 * (1 - dc) + v01 * dc
    v1 = v10 * (1 - dc) + v11 * dc
    out = v0 * (1 - dr) + v1 * dr
    return out.reshape(xy.shape[:-1])


@dataclass(frozen=True)
class CPTGSpec:
    dt: float
    T: int
    v_max: float
    d_min: float
    robot_radius: float
    obs_margin: float
    w_g: float
    w_u: float
    w_r: float


def derive_controls_from_positions(X: np.ndarray, dt: float) -> np.ndarray:
    return (X[:, 1:, :] - X[:, :-1, :]) / float(dt)


def potential_cost(
    X: np.ndarray,
    starts: List[np.ndarray],
    goals: List[np.ndarray],
    risk_map: np.ndarray,
    cell_size_m: float,
    spec: CPTGSpec,
) -> float:
    """Potential = sum_i (goal + control + risk). X: (2,T+1,2)"""
    X = np.asarray(X, dtype=float)
    U = derive_controls_from_positions(X, dt=spec.dt)

    # goal + control
    cost = 0.0
    for i in range(2):
        cost += 1e4 * float(np.sum((X[i, 0] - starts[i]) ** 2))
        cost += float(spec.w_g) * float(np.sum((X[i, -1] - goals[i]) ** 2))
    cost += float(spec.w_u) * float(np.sum(U * U))

    # risk (vectorized)
    r = _bilinear_field(risk_map, X, cell_size_m)  # (2,T+1)
    cost += float(spec.w_r) * float(np.sum(r))

    return float(cost)


def constraint_violations(
    X: np.ndarray,
    obstacle_boxes_m: List[Box],
    world_bounds: Box,
    spec: CPTGSpec,
    *,
    obs_dist_field_m: Optional[np.ndarray] = None,
    cell_size_m: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Return nonnegative violation magnitudes (0 means satisfied).

    Speed, bounds, collision are vectorized.
    Obstacle distance can be:
      - fast: obs_dist_field_m (H,W) in meters + bilinear sampling (needs cell_size_m)
      - fallback: min distance to boxes (slower)
    """
    X = np.asarray(X, dtype=float)
    dt = float(spec.dt)
    v_max_step = float(spec.v_max) * dt
    xmin, ymin, xmax, ymax = world_bounds

    # speed
    steps0 = np.linalg.norm(X[0, 1:, :] - X[0, :-1, :], axis=1)
    steps1 = np.linalg.norm(X[1, 1:, :] - X[1, :-1, :], axis=1)
    speed_viols = np.concatenate([np.maximum(0.0, steps0 - v_max_step), np.maximum(0.0, steps1 - v_max_step)])

    # collision
    d = np.linalg.norm(X[0] - X[1], axis=1)
    coll_viol = np.maximum(0.0, float(spec.d_min) - d)

    # bounds (vectorized)
    bx = np.maximum(0.0, xmin - X[..., 0]) + np.maximum(0.0, X[..., 0] - xmax)
    by = np.maximum(0.0, ymin - X[..., 1]) + np.maximum(0.0, X[..., 1] - ymax)
    bound_viol = (bx + by).reshape(-1)

    # obstacles
    clearance = float(spec.robot_radius + spec.obs_margin)
    if obs_dist_field_m is not None and cell_size_m is not None:
        d0 = _bilinear_field(obs_dist_field_m, X[0], float(cell_size_m))  # (T+1,)
        d1 = _bilinear_field(obs_dist_field_m, X[1], float(cell_size_m))  # (T+1,)
        obs_viol = np.concatenate([np.maximum(0.0, clearance - d0), np.maximum(0.0, clearance - d1)])
    else:
        # fallback: boxes
        pts0 = X[0].reshape(-1, 2)
        pts1 = X[1].reshape(-1, 2)
        d0 = min_distance_to_boxes(pts0, obstacle_boxes_m)
        d1 = min_distance_to_boxes(pts1, obstacle_boxes_m)
        obs_viol = np.concatenate([np.maximum(0.0, clearance - d0), np.maximum(0.0, clearance - d1)])

    return {"speed": speed_viols, "collision": coll_viol, "bounds": bound_viol, "obstacles": obs_viol}


def penalty_cost(
    X: np.ndarray,
    starts: List[np.ndarray],
    goals: List[np.ndarray],
    risk_map: np.ndarray,
    cell_size_m: float,
    obstacle_boxes_m: List[Box],
    world_bounds: Box,
    spec: CPTGSpec,
    penalty_w: float,
    *,
    obs_dist_field_m: Optional[np.ndarray] = None,
) -> float:
    base = potential_cost(X, starts, goals, risk_map, cell_size_m, spec)
    viols = constraint_violations(
        X,
        obstacle_boxes_m=obstacle_boxes_m,
        world_bounds=world_bounds,
        spec=spec,
        obs_dist_field_m=obs_dist_field_m,
        cell_size_m=cell_size_m,
    )
    pen = 0.0
    for v in viols.values():
        vv = np.asarray(v, dtype=float)
        pen += float(np.sum(vv * vv))
    return float(base + float(penalty_w) * pen)


def pack_positions(X: np.ndarray) -> np.ndarray:
    """Pack decision vars (exclude t=0 which is fixed): (2,T+1,2)->(4*T,)"""
    X = np.asarray(X, dtype=float)
    return X[:, 1:, :].reshape(-1)


def unpack_positions(z: np.ndarray, T: int, starts: List[np.ndarray]) -> np.ndarray:
    """Unpack vars into full trajectory with fixed starts. Output (2,T+1,2)."""
    z = np.asarray(z, dtype=float).reshape(2, T, 2)
    X = np.zeros((2, T + 1, 2), dtype=float)
    X[0, 0] = np.asarray(starts[0], dtype=float)
    X[1, 0] = np.asarray(starts[1], dtype=float)
    X[:, 1:, :] = z
    return X
