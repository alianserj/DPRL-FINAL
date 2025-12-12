from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from lite_genbr.sensors.visibility import VisDict, VisKey


@dataclass
class SensorPolicy:
    """
    A sensor configuration = K sensors chosen from candidate locations, each with an orientation.

    sensors: list of (cand_idx, orient_idx)
    """
    sensors: List[VisKey]
    exposure_map: 'Optional[np.ndarray]' = None  # int[H,W], computed lazily

    def key(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(sorted(tuple(s) for s in self.sensors))

    def compute_exposure_map(self, vis: VisDict, grid_shape: Tuple[int, int]) -> np.ndarray:
        """Compute exposure count per cell by summing the boolean visibility masks."""
        if self.exposure_map is not None:
            return self.exposure_map
        H, W = grid_shape
        exp = np.zeros((H, W), dtype=np.int16)
        for (ci, oi) in self.sensors:
            exp += vis[(ci, oi)].astype(np.int16)
        self.exposure_map = exp
        return exp
