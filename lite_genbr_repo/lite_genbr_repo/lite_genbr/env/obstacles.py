from __future__ import annotations
from lite_genbr.env.grid import GridWorld


def add_multimodal_obstacles(grid: GridWorld, layout: str = "multimodal_v1") -> None:
    """
    Deterministic obstacle layout engineered to create multi-modal navigation.

    Layout "multimodal_v1":
      - central rectangular block creates left/right corridors
      - two small "pinch" blocks encourage tighter interactions
    """
    if layout != "multimodal_v1":
        raise ValueError(f"Unknown layout: {layout}")

    H, W = grid.H, grid.W

    # Central block
    r0, r1 = int(0.33 * H), int(0.67 * H)
    c0, c1 = int(0.40 * W), int(0.60 * W)
    grid.set_obstacle_rect(r0, r1, c0, c1)

    # Two small pinch blocks
    grid.set_obstacle_rect(max(0, r0 - 3), max(0, r0 - 1), min(W, c0 + 1), min(W, c0 + 4))
    grid.set_obstacle_rect(min(H, r1 + 1), min(H, r1 + 3), max(0, c1 - 4), max(0, c1 - 1))
