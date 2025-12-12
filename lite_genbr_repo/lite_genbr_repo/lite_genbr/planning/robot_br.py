from __future__ import annotations
from typing import List
import numpy as np
from lite_genbr.env.grid import GridWorld, Cell
from lite_genbr.planning.a_star import a_star


def robot_best_response(
    grid: GridWorld,
    start: Cell,
    goal: Cell,
    expected_risk_map: np.ndarray,
    risk_w: float,
    allow_diagonal: bool,
) -> List[Cell]:
    """Best response path for robot via A* on movement + risk."""
    res = a_star(
        grid=grid,
        start=start,
        goal=goal,
        risk_map=expected_risk_map,
        risk_w=risk_w,
        allow_diagonal=allow_diagonal,
    )
    if not res.path:
        raise RuntimeError("A* failed to find a path. Check obstacle layout.")
    return res.path
