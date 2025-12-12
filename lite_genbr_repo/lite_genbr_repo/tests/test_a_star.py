from __future__ import annotations
import numpy as np
from lite_genbr.env.grid import GridWorld
from lite_genbr.env.obstacles import add_multimodal_obstacles
from lite_genbr.planning.a_star import a_star


def test_a_star_finds_path():
    grid = GridWorld(H=30, W=30)
    add_multimodal_obstacles(grid)
    risk = np.zeros((30, 30), dtype=float)
    res = a_star(grid, (2, 2), (27, 27), risk, risk_w=1.0, allow_diagonal=True)
    assert res.path
    assert res.path[0] == (2, 2)
    assert res.path[-1] == (27, 27)
