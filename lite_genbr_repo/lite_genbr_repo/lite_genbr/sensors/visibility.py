from __future__ import annotations

from typing import Dict, List, Tuple
import math
import numpy as np

from lite_genbr.env.grid import GridWorld, Cell

VisKey = Tuple[int, int]  # (cand_idx, orient_idx)
VisDict = Dict[VisKey, np.ndarray]  # bool[H,W]


def bresenham_line(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """Bresenham's line algorithm between two grid cells."""
    points: List[Tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


def angle_deg_from_to(src: Cell, dst: Cell) -> float:
    """
    Angle in degrees with grid coordinates:
      0 degrees -> +x (east), 90 degrees -> +y (south).
    """
    r0, c0 = src
    r1, c1 = dst
    dx = c1 - c0
    dy = r1 - r0  # down is positive
    return math.degrees(math.atan2(dy, dx)) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def has_line_of_sight(grid: GridWorld, src: Cell, dst: Cell) -> bool:
    """True if Bresenham ray from src to dst does not cross obstacle cells (excluding src)."""
    pts = bresenham_line(src[0], src[1], dst[0], dst[1])
    for rr, cc in pts[1:]:
        if not grid.in_bounds((rr, cc)):
            return False
        if grid.obstacles[rr, cc]:
            return False
    return True


def precompute_visibility(
    grid: GridWorld,
    candidates: List[Cell],
    orientations_deg: List[int],
    fov_deg: float,
    range_cells: int,
) -> VisDict:
    """
    Precompute visibility masks for each candidate + orientation.

    vis[(cand_idx, orient_idx)] is bool[H,W] with True where the sensor sees the cell.
    """
    H, W = grid.H, grid.W
    vis: VisDict = {}
    half_fov = 0.5 * float(fov_deg)

    for ci, (sr, sc) in enumerate(candidates):
        for oi, odeg in enumerate(orientations_deg):
            mask = np.zeros((H, W), dtype=bool)
            rmin = max(0, sr - range_cells)
            rmax = min(H - 1, sr + range_cells)
            cmin = max(0, sc - range_cells)
            cmax = min(W - 1, sc + range_cells)
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if grid.obstacles[r, c]:
                        continue
                    dr = r - sr
                    dc = c - sc
                    dist = math.sqrt(dr * dr + dc * dc)
                    if dist <= 1e-9 or dist > range_cells + 1e-9:
                        continue
                    ang = angle_deg_from_to((sr, sc), (r, c))
                    if angle_diff_deg(ang, float(odeg)) > half_fov + 1e-9:
                        continue
                    if not has_line_of_sight(grid, (sr, sc), (r, c)):
                        continue
                    mask[r, c] = True
            vis[(ci, oi)] = mask
    return vis
