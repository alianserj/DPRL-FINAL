from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np

Cell = Tuple[int, int]  # (row, col)


@dataclass
class GridWorld:
    """Simple 2D grid with obstacles."""
    H: int
    W: int
    obstacles: 'Optional[np.ndarray]' = None  # bool[H,W]

    def __post_init__(self) -> None:
        if self.obstacles is None:
            self.obstacles = np.zeros((self.H, self.W), dtype=bool)
        assert self.obstacles.shape == (self.H, self.W)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.H and 0 <= c < self.W

    def is_free(self, cell: Cell) -> bool:
        r, c = cell
        return self.in_bounds(cell) and not bool(self.obstacles[r, c])

    def set_obstacle_rect(self, r0: int, r1: int, c0: int, c1: int) -> None:
        """Inclusive-exclusive rectangle [r0:r1, c0:c1] set to obstacles."""
        self.obstacles[r0:r1, c0:c1] = True

    def neighbors(self, cell: Cell, allow_diagonal: bool = True) -> Iterable[Tuple[Cell, float]]:
        """Return (neighbor_cell, move_cost)."""
        r, c = cell
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        if allow_diagonal:
            d = float(np.sqrt(2.0))
            steps += [(-1, -1, d), (-1, 1, d), (1, -1, d), (1, 1, d)]
        for dr, dc, cost in steps:
            nxt = (r + dr, c + dc)
            if self.is_free(nxt):
                yield nxt, cost

    def free_cells(self) -> np.ndarray:
        return np.argwhere(~self.obstacles)

    def sample_sensor_candidates(
        self,
        M: int,
        wall_bias: float,
        rng: np.random.Generator,
        wall_margin: int = 2,
    ) -> List[Cell]:
        """Sample M free cells; wall_bias controls preference for boundary-adjacent cells."""
        free = self.free_cells()
        if len(free) == 0:
            raise ValueError("No free cells to sample candidates from.")

        near = []
        for r, c in free:
            if (r < wall_margin) or (r >= self.H - wall_margin) or (c < wall_margin) or (c >= self.W - wall_margin):
                near.append((int(r), int(c)))
        if len(near) == 0:
            near = [(int(r), int(c)) for r, c in free]

        all_free = [(int(r), int(c)) for r, c in free]
        candidates: List[Cell] = []
        seen = set()
        while len(candidates) < M:
            pool = near if rng.random() < wall_bias else all_free
            cell = pool[int(rng.integers(0, len(pool)))]
            if cell in seen:
                continue
            seen.add(cell)
            candidates.append(cell)
            if len(seen) >= len(all_free):
                break
        return candidates
