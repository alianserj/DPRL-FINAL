from __future__ import annotations

from typing import List, Tuple
from lite_genbr.env.grid import GridWorld

Box = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) in meters


def grid_to_obstacle_boxes_m(grid: GridWorld, cell_size_m: float) -> List[Box]:
    """
    Convert obstacle cells into a modest set of axis-aligned boxes in meters.

    We run-length merge per row, then merge vertically if spans match.
    """
    obs = grid.obstacles
    H, W = obs.shape

    row_segs = []  # (r, c0, c1_excl)
    for r in range(H):
        c = 0
        while c < W:
            if not obs[r, c]:
                c += 1
                continue
            c0 = c
            while c < W and obs[r, c]:
                c += 1
            row_segs.append((r, c0, c))

    boxes: List[Box] = []
    used = set()
    for idx, (r, c0, c1) in enumerate(row_segs):
        if idx in used:
            continue
        r0 = r
        r1 = r + 1
        used.add(idx)
        rr = r + 1
        while rr < H:
            found = None
            for j, (rj, cc0, cc1) in enumerate(row_segs):
                if j in used:
                    continue
                if rj == rr and cc0 == c0 and cc1 == c1:
                    found = j
                    break
            if found is None:
                break
            used.add(found)
            rr += 1
            r1 = rr

        xmin = c0 * cell_size_m
        xmax = c1 * cell_size_m
        ymin = r0 * cell_size_m
        ymax = r1 * cell_size_m
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes
