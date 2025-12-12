from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lite_genbr.env.grid import GridWorld


def plot_multinash_solutions(
    grid: GridWorld,
    obstacle_boxes_m: List[Tuple[float, float, float, float]],
    sols: List[Dict[str, Any]],
    out_path: str,
    cell_size_m: float,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    for (xmin, ymin, xmax, ymax) in obstacle_boxes_m:
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(xs, ys)
    for s in sols[:8]:
        t1 = np.array(s["traj_robot_1"], dtype=float)
        t2 = np.array(s["traj_robot_2"], dtype=float)
        plt.plot(t1[:, 0], t1[:, 1])
        plt.plot(t2[:, 0], t2[:, 1])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, grid.W * cell_size_m)
    plt.ylim(grid.H * cell_size_m, 0)
    plt.title("MultiNash-PF (lite): multiple local solutions")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
