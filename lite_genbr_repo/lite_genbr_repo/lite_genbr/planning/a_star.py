from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from lite_genbr.env.grid import GridWorld, Cell


@dataclass
class AStarResult:
    path: List[Cell]
    cost: float
    expansions: int


def heuristic(a: Cell, b: Cell) -> float:
    (r1, c1), (r2, c2) = a, b
    return float(np.hypot(r2 - r1, c2 - c1))


def reconstruct(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(
    grid: GridWorld,
    start: Cell,
    goal: Cell,
    risk_map: np.ndarray,
    risk_w: float,
    allow_diagonal: bool,
) -> AStarResult:
    """
    A* over the grid with cost:
      step_cost = move_cost + risk_w * risk_map[next_cell]
    """
    if not grid.is_free(start) or not grid.is_free(goal):
        raise ValueError("Start or goal is not in a free cell.")

    open_heap: List[Tuple[float, Cell]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}
    expansions = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        expansions += 1
        if current == goal:
            path = reconstruct(came_from, current)
            return AStarResult(path=path, cost=g_score[current], expansions=expansions)

        for nxt, move_cost in grid.neighbors(current, allow_diagonal=allow_diagonal):
            tentative = g_score[current] + float(move_cost) + float(risk_w * risk_map[nxt])
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f_score = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, nxt))

    return AStarResult(path=[], cost=float("inf"), expansions=expansions)
