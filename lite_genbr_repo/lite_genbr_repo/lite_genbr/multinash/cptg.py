from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Box = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


def point_to_box_distance(p: np.ndarray, box: Box) -> float:
    """Euclidean distance from point p=[x,y] to axis-aligned box (0 if inside)."""
    x, y = float(p[0]), float(p[1])
    xmin, ymin, xmax, ymax = box
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return float(np.hypot(dx, dy))


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


def pack_positions(X: np.ndarray) -> np.ndarray:
    return X.reshape(-1)


def unpack_positions(z: np.ndarray, T: int) -> np.ndarray:
    return np.asarray(z, dtype=float).reshape(2, T + 1, 2)


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
    """Potential = sum_i (goal + control + risk)."""
    U = derive_controls_from_positions(X, dt=spec.dt)
    cost = 0.0
    for i in range(2):
        cost += 1e4 * float(np.sum((X[i, 0] - starts[i]) ** 2))
        cost += spec.w_g * float(np.sum((X[i, -1] - goals[i]) ** 2))
        cost += spec.w_u * float(np.sum(U[i] ** 2))
        rsum = 0.0
        for t in range(spec.T + 1):
            rsum += bilinear_sample(risk_map, X[i, t, 0], X[i, t, 1], cell_size=cell_size_m)
        cost += spec.w_r * float(rsum)
    return float(cost)


def constraint_violations(
    X: np.ndarray,
    obstacle_boxes_m: List[Box],
    world_bounds: Box,
    spec: CPTGSpec,
) -> dict:
    """Return nonnegative violation magnitudes (0 means satisfied)."""
    dt = spec.dt
    v_max_step = spec.v_max * dt
    xmin, ymin, xmax, ymax = world_bounds

    # speed
    speed_viols = []
    for i in range(2):
        steps = np.linalg.norm(X[i, 1:, :] - X[i, :-1, :], axis=1)
        speed_viols.append(np.maximum(0.0, steps - v_max_step))

    # inter-robot
    d = np.linalg.norm(X[0] - X[1], axis=1)
    coll_viol = np.maximum(0.0, spec.d_min - d)

    # bounds
    bound_viol = []
    for i in range(2):
        x = X[i, :, 0]; y = X[i, :, 1]
        bound_viol.append(np.maximum(0.0, xmin - x))
        bound_viol.append(np.maximum(0.0, x - xmax))
        bound_viol.append(np.maximum(0.0, ymin - y))
        bound_viol.append(np.maximum(0.0, y - ymax))
    bound_viol = np.concatenate(bound_viol)

    # obstacles
    clearance = spec.robot_radius + spec.obs_margin
    obs_viol = []
    for i in range(2):
        for t in range(spec.T + 1):
            p = X[i, t]
            min_dist = min(point_to_box_distance(p, box) for box in obstacle_boxes_m) if obstacle_boxes_m else float("inf")
            obs_viol.append(max(0.0, clearance - min_dist))
    obs_viol = np.asarray(obs_viol, dtype=float)

    return {"speed": np.concatenate(speed_viols), "collision": coll_viol, "bounds": bound_viol, "obstacles": obs_viol}


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
) -> float:
    base = potential_cost(X, starts, goals, risk_map, cell_size_m, spec)
    viols = constraint_violations(X, obstacle_boxes_m, world_bounds, spec)
    pen = 0.0
    for v in viols.values():
        pen += float(np.sum(np.asarray(v, dtype=float) ** 2))
    return float(base + penalty_w * pen)
