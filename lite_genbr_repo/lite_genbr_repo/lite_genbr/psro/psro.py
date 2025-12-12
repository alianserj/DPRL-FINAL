from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import trange

from lite_genbr.env.grid import GridWorld, Cell
from lite_genbr.planning.robot_br import robot_best_response
from lite_genbr.planning.occupancy import occupancy_from_mixture, cost_of_path
from lite_genbr.sensors.sensor_br import sensor_best_response
from lite_genbr.sensors.sensor_policy import SensorPolicy
from lite_genbr.sensors.visibility import VisDict
from lite_genbr.psro.meta_solvers import (
    SmoothFictitiousPlayState,
    projected_replicator_dynamics,
    smooth_fictitious_play_distribution,
    smooth_fictitious_play_init,
    smooth_fictitious_play_update,
    solve_zero_sum_lp,
)
from lite_genbr.psro.metrics import nashconv_zero_sum


def _expected_risk_map(
    grid_shape: Tuple[int, int],
    sensor_population: List[SensorPolicy],
    q: np.ndarray,
    vis: VisDict,
) -> np.ndarray:
    H, W = grid_shape
    risk = np.zeros((H, W), dtype=float)
    for w, sp in zip(q, sensor_population):
        exp = sp.compute_exposure_map(vis, grid_shape=(H, W))
        risk += float(w) * exp.astype(float)
    return risk


def _row_br_value_oracle(
    grid: GridWorld,
    start: Cell,
    goal: Cell,
    vis: VisDict,
    sensor_population: List[SensorPolicy],
    q: np.ndarray,
    outer_move_w: float,
    outer_risk_w: float,
    allow_diagonal: bool,
) -> float:
    """Return row player's best achievable payoff (=-cost) vs sensor mixture q."""
    risk_map = _expected_risk_map((grid.H, grid.W), sensor_population, q, vis)
    path = robot_best_response(
        grid=grid,
        start=start,
        goal=goal,
        expected_risk_map=risk_map,
        risk_w=outer_risk_w,
        allow_diagonal=allow_diagonal,
    )
    exp_cost = 0.0
    for w, sp in zip(q, sensor_population):
        exp_map = sp.compute_exposure_map(vis, grid_shape=(grid.H, grid.W))
        exp_cost += float(w) * cost_of_path(path, exp_map, move_w=outer_move_w, risk_w=outer_risk_w)
    return -float(exp_cost)


def _col_br_value_oracle(
    grid_shape: Tuple[int, int],
    vis: VisDict,
    robot_population: List[List[Cell]],
    p: np.ndarray,
    outer_move_w: float,
    outer_risk_w: float,
    K: int,
    beta: float,
    rng: np.random.Generator,
) -> float:
    """Return min expected row payoff (=-cost) achievable by column via greedy BR."""
    occ = occupancy_from_mixture(grid_shape=grid_shape, paths=robot_population, weights=p)
    sp = sensor_best_response(occ_map=occ, vis=vis, K=K, beta=beta, rng=rng)

    exp_map = sp.compute_exposure_map(vis, grid_shape=grid_shape)
    exp_cost = 0.0
    for w, path in zip(p, robot_population):
        exp_cost += float(w) * cost_of_path(path, exp_map, move_w=outer_move_w, risk_w=outer_risk_w)
    return -float(exp_cost)


def _evaluate_payoff_matrix(
    grid_shape: Tuple[int, int],
    robot_population: List[List[Cell]],
    sensor_population: List[SensorPolicy],
    vis: VisDict,
    move_w: float,
    risk_w: float,
) -> np.ndarray:
    """payoff[i,j] = row payoff for robot path i vs sensor config j = -cost."""
    H, W = grid_shape
    n = len(robot_population)
    m = len(sensor_population)
    M = np.zeros((n, m), dtype=float)
    for j, sp in enumerate(sensor_population):
        exp = sp.compute_exposure_map(vis, grid_shape=(H, W))
        for i, path in enumerate(robot_population):
            cost = cost_of_path(path, exp, move_w=move_w, risk_w=risk_w)
            M[i, j] = -float(cost)
    return M


def psro_lite_genbr(
    grid: GridWorld,
    vis: VisDict,
    init_robot_paths: List[List[Cell]],
    init_sensor_policies: List[SensorPolicy],
    start: Cell,
    goal: Cell,
    cost_params,
    psro_params,
    sensor_K: int,
    sensor_beta: float,
    allow_diagonal: bool,
    rng: np.random.Generator,
    save_dir: str,
) -> Dict[str, Any]:
    """PSRO/DO-style loop with planning-based best response oracles."""
    move_w = float(cost_params.move_w)
    risk_w = float(cost_params.risk_w)

    robot_pop = list(init_robot_paths)
    sensor_pop = list(init_sensor_policies)

    payoff = _evaluate_payoff_matrix((grid.H, grid.W), robot_pop, sensor_pop, vis, move_w, risk_w)

    sfp_state = smooth_fictitious_play_init(len(robot_pop), len(sensor_pop))

    p = np.ones(len(robot_pop)) / len(robot_pop)
    q = np.ones(len(sensor_pop)) / len(sensor_pop)

    trace: List[float] = []
    log_path = os.path.join(save_dir, "psro_trace.jsonl")
    os.makedirs(save_dir, exist_ok=True)

    for it in trange(int(psro_params.iters), desc="PSRO"):
        # Meta-solver
        if psro_params.meta_solver == "lp":
            p, q, _ = solve_zero_sum_lp(payoff, eps=psro_params.lp_eps)
        elif psro_params.meta_solver == "prd":
            p, q = projected_replicator_dynamics(
                payoff,
                steps=int(psro_params.prd_steps),
                dt=float(psro_params.prd_dt),
                gamma=float(psro_params.prd_gamma),
                rng=rng,
            )
        elif psro_params.meta_solver == "sfp":
            p, q = smooth_fictitious_play_distribution(sfp_state, tau=float(psro_params.tau))
        else:
            raise ValueError(f"Unknown meta_solver: {psro_params.meta_solver}")

        # Exploitability via full-game oracles
        ex = nashconv_zero_sum(
            payoff=payoff,
            p=p,
            q=q,
            row_br_value_fn=lambda qq: _row_br_value_oracle(
                grid=grid,
                start=start,
                goal=goal,
                vis=vis,
                sensor_population=sensor_pop,
                q=qq,
                outer_move_w=move_w,
                outer_risk_w=risk_w,
                allow_diagonal=allow_diagonal,
            ),
            col_br_value_fn=lambda pp: _col_br_value_oracle(
                grid_shape=(grid.H, grid.W),
                vis=vis,
                robot_population=robot_pop,
                p=pp,
                outer_move_w=move_w,
                outer_risk_w=risk_w,
                K=int(sensor_K),
                beta=float(sensor_beta),
                rng=rng,
            ),
        )
        trace.append(float(ex))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"iter": it, "exploitability": float(ex), "n": len(robot_pop), "m": len(sensor_pop)}) + "\n")

        if float(ex) < float(psro_params.stop_exploitability):
            break

        # Oracles
        risk_map = _expected_risk_map((grid.H, grid.W), sensor_pop, q, vis)
        br_path = robot_best_response(
            grid=grid,
            start=start,
            goal=goal,
            expected_risk_map=risk_map,
            risk_w=risk_w,
            allow_diagonal=allow_diagonal,
        )

        occ = occupancy_from_mixture(grid_shape=(grid.H, grid.W), paths=robot_pop, weights=p)
        br_sensor = sensor_best_response(occ_map=occ, vis=vis, K=int(sensor_K), beta=float(sensor_beta), rng=rng)

        br_i = None
        br_j = None

        # Add robot
        if br_path not in robot_pop:
            robot_pop.append(br_path)
            br_i = len(robot_pop) - 1
            new_row = np.zeros((1, payoff.shape[1]), dtype=float)
            for j, sp in enumerate(sensor_pop):
                exp = sp.compute_exposure_map(vis, grid_shape=(grid.H, grid.W))
                new_row[0, j] = -cost_of_path(br_path, exp, move_w=move_w, risk_w=risk_w)
            payoff = np.vstack([payoff, new_row])
            sfp_state.counts_p = np.append(sfp_state.counts_p, 0.0)
        else:
            br_i = robot_pop.index(br_path)

        # Add sensor
        existing_keys = {sp.key(): idx for idx, sp in enumerate(sensor_pop)}
        k = br_sensor.key()
        if k not in existing_keys:
            sensor_pop.append(br_sensor)
            br_j = len(sensor_pop) - 1
            exp = br_sensor.compute_exposure_map(vis, grid_shape=(grid.H, grid.W))
            new_col = np.zeros((payoff.shape[0], 1), dtype=float)
            for i, path in enumerate(robot_pop):
                new_col[i, 0] = -cost_of_path(path, exp, move_w=move_w, risk_w=risk_w)
            payoff = np.hstack([payoff, new_col])
            sfp_state.counts_q = np.append(sfp_state.counts_q, 0.0)
        else:
            br_j = existing_keys[k]

        if psro_params.meta_solver == "sfp":
            smooth_fictitious_play_update(sfp_state, br_i=br_i, br_j=br_j, alpha=1.0)

    # Final mixture
    if psro_params.meta_solver == "lp":
        p, q, _ = solve_zero_sum_lp(payoff, eps=psro_params.lp_eps)
    elif psro_params.meta_solver == "prd":
        p, q = projected_replicator_dynamics(
            payoff,
            steps=int(psro_params.prd_steps),
            dt=float(psro_params.prd_dt),
            gamma=float(psro_params.prd_gamma),
            rng=rng,
        )
    else:
        p, q = smooth_fictitious_play_distribution(sfp_state, tau=float(psro_params.tau))

    with open(os.path.join(save_dir, "psro_result.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "p": p.tolist(),
                "q": q.tolist(),
                "exploitability_trace": trace,
                "n_robot": len(robot_pop),
                "n_sensor": len(sensor_pop),
            },
            f,
            indent=2,
        )

    return {
        "robot_population": robot_pop,
        "sensor_population": sensor_pop,
        "p": p,
        "q": q,
        "exploitability_trace": trace,
        "payoff_matrix": payoff,
    }
