from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lite_genbr.env.grid import GridWorld, Cell
from lite_genbr.sensors.sensor_policy import SensorPolicy
from lite_genbr.sensors.visibility import VisDict

import math

def _cell_center_xy(cell: Cell, cell_size: float = 1.0) -> Tuple[float, float]:
    r, c = cell
    return (c + 0.5) * cell_size, (r + 0.5) * cell_size


def plot_world(
    grid: GridWorld,
    start: Cell,
    goal: Cell,
    candidates: Optional[List[Cell]],
    out_path: str,
    title: str = "World (obstacles + start/goal + sensor candidates)",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.imshow(grid.obstacles.astype(float), origin="upper")
    sr, sc = start
    gr, gc = goal
    plt.scatter([sc], [sr], s=80, marker="o", label="Start")
    plt.scatter([gc], [gr], s=80, marker="*", label="Goal")

    if candidates is not None and len(candidates) > 0:
        plt.scatter([c for r, c in candidates], [r for r, c in candidates], s=20, marker="x", label="Sensor candidates")

    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap_with_paths(
    grid: GridWorld,
    heat: np.ndarray,
    out_path: str,
    title: str,
    starts_goals: Optional[List[Tuple[Cell, Cell]]] = None,
    paths: Optional[List[List[Cell]]] = None,
    path_labels: Optional[List[str]] = None,
    alpha: float = 0.85,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 7))
    # Mask obstacles for readability
    heat_show = heat.astype(float).copy()
    heat_show[grid.obstacles] = np.nan
    im = plt.imshow(heat_show, origin="upper", alpha=alpha)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # draw obstacles outline on top
    obs = grid.obstacles.astype(int)
    plt.imshow(np.where(obs == 1, 1.0, np.nan), origin="upper", alpha=0.25)

    if starts_goals is not None:
        for idx, (s, g) in enumerate(starts_goals):
            sr, sc = s
            gr, gc = g
            plt.scatter([sc], [sr], s=70, marker="o")
            plt.scatter([gc], [gr], s=70, marker="*")
            plt.text(sc + 0.2, sr + 0.2, f"S{idx+1}", fontsize=9)
            plt.text(gc + 0.2, gr + 0.2, f"G{idx+1}", fontsize=9)

    if paths is not None:
        for i, path in enumerate(paths):
            if len(path) == 0:
                continue
            xs = [c for r, c in path]
            ys = [r for r, c in path]
            label = path_labels[i] if (path_labels is not None and i < len(path_labels)) else None
            plt.plot(xs, ys, linewidth=2.0, label=label)
        if path_labels is not None:
            plt.legend(loc="lower left")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mixture_bars(
    weights: np.ndarray,
    out_path: str,
    title: str,
    top_k: int = 12,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    w = np.array(weights, dtype=float).copy()
    order = np.argsort(-w)
    order = order[: min(top_k, len(order))]
    vals = w[order]
    labels = [str(int(i)) for i in order]

    plt.figure(figsize=(9, 3.5))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=0)
    plt.ylabel("Probability mass")
    plt.xlabel("Strategy index (top-k)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_payoff_matrix(
    payoff: np.ndarray,
    out_path: str,
    title: str = "Empirical payoff matrix (robot payoff = -cost)",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    im = plt.imshow(payoff, aspect="auto", origin="upper")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Sensor strategy index")
    plt.ylabel("Robot strategy index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _draw_fov_cone(
    ax: Any,
    center_rc: Cell,
    orient_deg: float,
    fov_deg: float,
    range_cells: float,
    color: str = "cyan",
    lw: float = 1.2,
) -> None:
    # Plot in grid coordinates: x=col, y=row with origin='upper'
    r, c = center_rc
    x0, y0 = c + 0.5, r + 0.5
    half = fov_deg / 2.0
    angles = [math.radians(orient_deg - half), math.radians(orient_deg + half)]
    for ang in angles:
        x1 = x0 + range_cells * math.cos(ang)
        y1 = y0 - range_cells * math.sin(ang)  # y down in imshow upper-origin
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.9)
    # orientation arrow
    ang0 = math.radians(orient_deg)
    xh = x0 + (0.6 * range_cells) * math.cos(ang0)
    yh = y0 - (0.6 * range_cells) * math.sin(ang0)
    ax.plot([x0, xh], [y0, yh], color=color, linewidth=lw + 0.3, alpha=0.9)


def plot_top_sensor_configs(
    grid: GridWorld,
    candidates: List[Cell],
    sensor_population: List[SensorPolicy],
    q: np.ndarray,
    orientations_deg: List[int],
    fov_deg: float,
    range_cells: int,
    out_path: str,
    top_k: int = 3,
) -> None:
    import math
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    w = np.array(q, dtype=float)
    order = np.argsort(-w)
    order = order[: min(top_k, len(order))]

    plt.figure(figsize=(7, 7))
    plt.imshow(grid.obstacles.astype(float), origin="upper")
    plt.title(f"Top-{len(order)} sensor configs by q-mass (showing FOV cones)")
    plt.scatter([c for r, c in candidates], [r for r, c in candidates], s=10, marker="x", alpha=0.25)

    ax = plt.gca()
    colors = ["cyan", "magenta", "yellow", "lime", "orange"]
    for rank, j in enumerate(order):
        sp = sensor_population[int(j)]
        col = colors[rank % len(colors)]
        for (ci, oi) in sp.sensors:
            cr, cc = candidates[int(ci)]
            # draw point
            ax.scatter([cc], [cr], s=45, marker="s", color=col, alpha=0.9)
            # draw cone + orientation
            _draw_fov_cone(
                ax=ax,
                center_rc=(cr, cc),
                orient_deg=float(orientations_deg[int(oi)]),
                fov_deg=float(fov_deg),
                range_cells=float(range_cells),
                color=col,
            )
        ax.text(0.02, 0.98 - 0.04 * rank, f"cfg {int(j)}: q={w[int(j)]:.3f}", transform=ax.transAxes, color=col, fontsize=9, va="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def expected_risk_maps_from_mixture(
    grid: GridWorld,
    sensor_population: List[SensorPolicy],
    q: np.ndarray,
    vis: VisDict,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (expected_exposure_count, expected_detection_prob)."""
    H, W = grid.H, grid.W
    exp_count = np.zeros((H, W), dtype=float)
    det_prob = np.zeros((H, W), dtype=float)
    for w, sp in zip(q, sensor_population):
        exp = sp.compute_exposure_map(vis, grid_shape=(H, W)).astype(float)
        exp_count += float(w) * exp
        det_prob += float(w) * (1.0 - np.exp(-float(beta) * exp))
    return exp_count, det_prob
