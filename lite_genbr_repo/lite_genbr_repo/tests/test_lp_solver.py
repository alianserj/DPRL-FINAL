from __future__ import annotations
import numpy as np
from lite_genbr.psro.meta_solvers import solve_zero_sum_lp


def test_lp_solver_rps():
    M = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    p, q, v = solve_zero_sum_lp(M)
    assert np.allclose(p, np.ones(3) / 3, atol=1e-2)
    assert np.allclose(q, np.ones(3) / 3, atol=1e-2)
    assert abs(v) < 1e-2
