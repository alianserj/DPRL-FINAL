from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from lite_genbr.multinash.cptg import CPTGSpec, min_distance_to_boxes, penalty_cost, softplus


@dataclass
class UKFNode:
    """Minimal UKF node for fast implicit-PF proposals.

    State:  joint positions [x1,y1,x2,y2]
    Meas.:  [x1,y1,x2,y2, b_obs1, b_obs2, b_col]
    """

    dim_x: int
    dim_z: int
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        self.lam = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        nlam = self.dim_x + self.lam
        self.Wm = np.full(2 * self.dim_x + 1, 0.5 / nlam, dtype=float)
        self.Wc = np.full(2 * self.dim_x + 1, 0.5 / nlam, dtype=float)
        self.Wm[0] = self.lam / nlam
        self.Wc[0] = self.lam / nlam + (1 - self.alpha**2 + self.beta)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n = self.dim_x
        nlam = n + self.lam
        try:
            L = np.linalg.cholesky(nlam * P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(nlam * (P + np.eye(n) * 1e-8))
        sig = np.zeros((2 * n + 1, n), dtype=float)
        sig[0] = x
        for k in range(n):
            sig[k + 1] = x + L[:, k]
            sig[n + k + 1] = x - L[:, k]
        return sig

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        h_func,
        z_target: np.ndarray,
        R: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sig_x = self.sigma_points(x_pred, P_pred)
        sig_z = np.asarray([h_func(s) for s in sig_x], dtype=float)

        z_mean = (self.Wm[:, None] * sig_z).sum(axis=0)

        Pz = np.zeros((self.dim_z, self.dim_z), dtype=float)
        for i in range(sig_z.shape[0]):
            dz = sig_z[i] - z_mean
            Pz += self.Wc[i] * np.outer(dz, dz)
        Pz += R

        Pxz = np.zeros((self.dim_x, self.dim_z), dtype=float)
        for i in range(sig_z.shape[0]):
            dx = sig_x[i] - x_pred
            dz = sig_z[i] - z_mean
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(Pz)
        x_new = x_pred + K @ (z_target - z_mean)
        P_new = P_pred - K @ Pz @ K.T
        return x_new, P_new


def _log_barrier(val: float, alpha: float = 1.0) -> float:
    """Soft barrier (positive val => constraint violation)."""
    return float(softplus(val) / float(alpha))


def _bilinear_single(field: np.ndarray, x: float, y: float, cell_size_m: float) -> float:
    """Fast bilinear sampling at one point (meters)."""
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
    refine_iters: int,  # kept for API compatibility; not used (UKF does the cheap 'refine')
    penalty_w: float,
    rng: np.random.Generator,
    *,
    obs_dist_field_m: Optional[np.ndarray] = None,
) -> List[Dict]:
    """UKF-based implicit particle filter (fast).

    This matches the spirit of your provided code:
      - propose control toward a reference
      - UKF update using virtual constraints (obstacle + collision)
      - cluster trajectories later with Fréchet distance

    Output: list of samples: {"X": (2,T+1,2), "pen_cost": float, "weight": float}
    """
    T = int(spec.T)
    dt = float(spec.dt)
    xmin, ymin, xmax, ymax = world_bounds

    dim_x = 4
    dim_z = 7
    ukf = UKFNode(dim_x=dim_x, dim_z=dim_z)

    # Process / measurement noise
    Q = np.eye(dim_x, dtype=float) * 0.05
    R = np.diag([0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.08]).astype(float)

    ref = np.concatenate([X_ref[0], X_ref[1]], axis=1)  # (T+1,4)

    x0 = np.concatenate([starts[0], starts[1]], axis=0).astype(float)
    particles = np.repeat(x0[None, :], int(J), axis=0)
    particles += rng.normal(scale=float(noise_sigma), size=particles.shape) * 0.2

    P_bank = np.repeat((np.eye(dim_x) * 0.3)[None, :, :], int(J), axis=0)

    # (J,T+1,2,2)
    traj = np.zeros((int(J), T + 1, 2, 2), dtype=float)
    traj[:, 0, 0, :] = starts[0]
    traj[:, 0, 1, :] = starts[1]

    clearance = float(spec.robot_radius + spec.obs_margin)

    def h_func(x_joint: np.ndarray) -> np.ndarray:
        p1 = x_joint[0:2]
        p2 = x_joint[2:4]

        if obs_dist_field_m is not None:
            d1 = _bilinear_single(obs_dist_field_m, float(p1[0]), float(p1[1]), float(cell_size_m))
            d2 = _bilinear_single(obs_dist_field_m, float(p2[0]), float(p2[1]), float(cell_size_m))
        else:
            d1 = float(min_distance_to_boxes(p1[None, :], obstacle_boxes_m)[0])
            d2 = float(min_distance_to_boxes(p2[None, :], obstacle_boxes_m)[0])

        b1 = _log_barrier(clearance - d1, alpha=1.0)
        b2 = _log_barrier(clearance - d2, alpha=1.0)

        dcol = float(np.linalg.norm(p1 - p2))
        bc = _log_barrier(float(spec.d_min) - dcol, alpha=1.0)

        return np.array([x_joint[0], x_joint[1], x_joint[2], x_joint[3], b1, b2, bc], dtype=float)

    v_max = float(spec.v_max)
    for t in range(T):
        ref_next = ref[t + 1]
        z_target = np.array([ref_next[0], ref_next[1], ref_next[2], ref_next[3], 0.0, 0.0, 0.0], dtype=float)

        for j in range(int(J)):
            x = particles[j]

            # propose control toward reference + noise
            desired = (ref_next - x) / max(1e-9, dt)
            u = desired + rng.normal(scale=float(noise_sigma), size=4)

            # clip per-robot speed
            u1 = u[0:2]; u2 = u[2:4]
            s1 = float(np.linalg.norm(u1)); s2 = float(np.linalg.norm(u2))
            if s1 > v_max:
                u1 = (v_max / max(1e-9, s1)) * u1
            if s2 > v_max:
                u2 = (v_max / max(1e-9, s2)) * u2
            u = np.concatenate([u1, u2], axis=0)

            # predict
            x_pred = x + u * dt
            x_pred[0] = float(np.clip(x_pred[0], xmin, xmax))
            x_pred[1] = float(np.clip(x_pred[1], ymin, ymax))
            x_pred[2] = float(np.clip(x_pred[2], xmin, xmax))
            x_pred[3] = float(np.clip(x_pred[3], ymin, ymax))

            P_pred = P_bank[j] + Q

            # update
            x_new, P_new = ukf.update(x_pred, P_pred, h_func, z_target, R)

            # clamp
            x_new[0] = float(np.clip(x_new[0], xmin, xmax))
            x_new[1] = float(np.clip(x_new[1], ymin, ymax))
            x_new[2] = float(np.clip(x_new[2], xmin, xmax))
            x_new[3] = float(np.clip(x_new[3], ymin, ymax))

            particles[j] = x_new
            P_bank[j] = P_new
            traj[j, t + 1, 0, :] = x_new[0:2]
            traj[j, t + 1, 1, :] = x_new[2:4]

    # score particles -> weights
    samples: List[Dict] = []
    beta = 1.0 / max(1e-6, float(np.std(risk_map) + 1.0))

    for j in range(int(J)):
        X = traj[j].transpose(1, 0, 2)  # (2,T+1,2)
        pcost = float(
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
        w = float(np.exp(-beta * pcost))
        samples.append({"X": X, "pen_cost": pcost, "weight": w})

    wsum = float(sum(s["weight"] for s in samples))
    if wsum > 0:
        for s in samples:
            s["weight"] /= wsum
    return samples
