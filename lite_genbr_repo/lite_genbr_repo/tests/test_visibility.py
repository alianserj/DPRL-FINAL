from __future__ import annotations
import numpy as np
from lite_genbr.env.grid import GridWorld
from lite_genbr.env.obstacles import add_multimodal_obstacles
from lite_genbr.sensors.visibility import precompute_visibility


def test_visibility_shapes():
    grid = GridWorld(H=30, W=30)
    add_multimodal_obstacles(grid)
    rng = np.random.default_rng(0)
    candidates = grid.sample_sensor_candidates(M=8, wall_bias=1.0, rng=rng)
    vis = precompute_visibility(grid, candidates, orientations_deg=[0, 90], fov_deg=90.0, range_cells=6)
    assert len(vis) == len(candidates) * 2
    for m in vis.values():
        assert m.shape == (30, 30)
