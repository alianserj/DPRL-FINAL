from __future__ import annotations
from typing import List, Tuple
import numpy as np
from lite_genbr.env.grid import Cell


def path_length(path: List[Cell]) -> int:
    return max(0, len(path) - 1)


def occupancy_from_mixture(
    grid_shape: Tuple[int, int],
    paths: List[List[Cell]],
    weights: np.ndarray,
) -> np.ndarray:
    """Expected visitation counts per cell from a mixture over paths."""
    H, W = grid_shape
    occ = np.zeros((H, W), dtype=float)
    for w, path in zip(weights, paths):
        for (r, c) in path:
            occ[r, c] += float(w)
    return occ


def cost_of_path(
    path: List[Cell],
    exposure_map: np.ndarray,
    move_w: float,
    risk_w: float,
) -> float:
    """Robot cost = move_w*path_length + risk_w*sum exposure(cell) along path."""
    if not path:
        return float("inf")
    length = path_length(path)
    exposure_sum = 0.0
    for cell in path:
        exposure_sum += float(exposure_map[cell])
    return float(move_w * length + risk_w * exposure_sum)
