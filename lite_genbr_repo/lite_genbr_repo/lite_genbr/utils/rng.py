from __future__ import annotations
import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a numpy RNG with a fixed seed."""
    return np.random.default_rng(int(seed))
