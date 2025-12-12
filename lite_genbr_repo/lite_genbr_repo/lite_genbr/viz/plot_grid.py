from __future__ import annotations
import os
from typing import List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lite_genbr.env.grid import GridWorld
from lite_genbr.sensors.sensor_policy import SensorPolicy


def plot_grid_and_sensors(
    grid: GridWorld,
    candidates: List[tuple[int, int]],
    sensor_policies: List[SensorPolicy],
    q: np.ndarray,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    obs = grid.obstacles.astype(float)
    plt.imshow(obs, origin="upper")
    plt.title("Grid obstacles (black=obstacle)")
    plt.scatter([c for r, c in candidates], [r for r, c in candidates], s=12, marker="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
