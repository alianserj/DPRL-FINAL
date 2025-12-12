from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class GridParams:
    H: int = 30
    W: int = 30
    cell_size_m: float = 0.5
    seed: int = 0
    obstacle_layout: str = "multimodal_v1"
    start_a: Tuple[int, int] = (2, 2)
    goal_a: Tuple[int, int] = (27, 27)
    start_b: Tuple[int, int] = (27, 2)
    goal_b: Tuple[int, int] = (2, 27)
    allow_diagonal: bool = True


@dataclass(frozen=True)
class SensorParams:
    K: int = 4
    M: int = 32
    range_cells: int = 6
    fov_deg: float = 90.0
    orientations_deg: List[int] = field(
        default_factory=lambda: [0, 45, 90, 135, 180, 225, 270, 315]
    )
    wall_bias: float = 0.8
    beta_detection: float = 1.0  # diminishing returns for sensor BR


@dataclass(frozen=True)
class OuterCostParams:
    move_w: float = 1.0
    risk_w: float = 1.0  # also used as lambda in robot A* edge cost


@dataclass(frozen=True)
class PSROParams:
    iters: int = 25
    meta_solver: str = "sfp"  # {"sfp","prd","lp"}
    tau: float = 1.2  # softmax temperature for sfp
    prd_steps: int = 250
    prd_dt: float = 0.25
    prd_gamma: float = 0.05
    lp_eps: float = 1e-9
    seed: int = 0
    stop_exploitability: float = 1e-3


@dataclass(frozen=True)
class RiskModeParams:
    L: int = 3
    max_kmedoids_iters: int = 10


@dataclass(frozen=True)
class CPTGParams:
    dt: float = 0.2
    T: int = 40
    v_max: float = 1.0
    d_min: float = 0.8
    robot_radius: float = 0.25
    obs_margin: float = 0.25
    w_g: float = 50.0
    w_u: float = 0.2
    w_r: float = 2.0


@dataclass(frozen=True)
class MultiNashParams:
    # PF particle count (bigger -> more modes but more compute)
    J: int = 30
    # stddev of gaussian noise on waypoints (meters)
    noise_sigma: float = 0.35
    # max L-BFGS-B iterations per particle (cheap local refinement)
    pf_refine_iters: int = 10
    # Fréchet distance threshold used for clustering particles into modes
    cluster_eps: float = 1.5
    # maximum number of distinct local solutions to refine
    max_solutions: int = 6
    # SLSQP iteration budget for final refinement
    solver_maxiter: int = 200
    # penalty weight for constraint violations (higher -> more feasibility)
    penalty_w: float = 150.0
    seed: int = 0


@dataclass(frozen=True)
class RunParams:

    grid: GridParams = GridParams()
    sensors: SensorParams = SensorParams()
    outer_cost: OuterCostParams = OuterCostParams()
    psro: PSROParams = PSROParams()
    risk_modes: RiskModeParams = RiskModeParams()
    cptg: CPTGParams = CPTGParams()
    multinash: MultiNashParams = MultiNashParams()
