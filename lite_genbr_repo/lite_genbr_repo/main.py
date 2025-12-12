from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np

from params import RunParams
from lite_genbr.env.grid import GridWorld
from lite_genbr.env.obstacles import add_multimodal_obstacles
from lite_genbr.sensors.visibility import precompute_visibility
from lite_genbr.planning.robot_br import robot_best_response
from lite_genbr.planning.occupancy import occupancy_from_mixture
from lite_genbr.sensors.sensor_br import sensor_best_response
from lite_genbr.psro.psro import psro_lite_genbr
from lite_genbr.clustering.risk_modes import cluster_sensor_mixture
from lite_genbr.env.conversions import grid_to_obstacle_boxes_m
from lite_genbr.multinash.multinash_pf import multinash_pf_solve
from lite_genbr.viz.plot_grid import plot_grid_and_sensors
from lite_genbr.viz.plot_psro import plot_exploitability
from lite_genbr.viz.plot_trajectories import plot_multinash_solutions
from lite_genbr.utils.rng import make_rng
from lite_genbr.utils.logging_utils import log, StageTimer



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--meta_solver", type=str, default=None, choices=["sfp", "prd", "lp"])
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="runs/demo")
    p.add_argument("--no_plots", action="store_true")

    # Optional speed knobs / staging
    p.add_argument("--outer_only", action="store_true", help="Run only the outer PSRO game (skip MultiNash).")
    p.add_argument("--risk_modes_L", type=int, default=None, help="Number of risk modes to extract (default from params).")

    p.add_argument("--multinash_J", type=int, default=None, help="Particle count per mode (default from params).")
    p.add_argument("--multinash_refine_iters", type=int, default=None, help="L-BFGS-B iterations per particle.")
    p.add_argument("--multinash_max_solutions", type=int, default=None, help="Max local solutions refined per mode.")
    p.add_argument("--multinash_solver_maxiter", type=int, default=None, help="SLSQP iteration budget per solution.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rp = RunParams()

    log("===== Lite Gen-BR Pipeline =====")
    log(f"meta_solver={rp.psro.meta_solver}, iters={rp.psro.iters}, seed={args.seed}")
    log(f"save_dir={args.save_dir}")


    # Build updated sub-params (avoid a full config system, keep it simple)
    risk_modes = type(rp.risk_modes)(
        **{**asdict(rp.risk_modes), "L": args.risk_modes_L if args.risk_modes_L is not None else rp.risk_modes.L}
    )

    multinash = type(rp.multinash)(
        **{
            **asdict(rp.multinash),
            "seed": args.seed,
            "J": args.multinash_J if args.multinash_J is not None else rp.multinash.J,
            "pf_refine_iters": args.multinash_refine_iters if args.multinash_refine_iters is not None else rp.multinash.pf_refine_iters,
            "max_solutions": args.multinash_max_solutions if args.multinash_max_solutions is not None else rp.multinash.max_solutions,
            "solver_maxiter": args.multinash_solver_maxiter if args.multinash_solver_maxiter is not None else rp.multinash.solver_maxiter,
        }
    )

    rp = RunParams(
        grid=type(rp.grid)(**{**asdict(rp.grid), "seed": args.seed}),
        sensors=rp.sensors,
        outer_cost=rp.outer_cost,
        psro=type(rp.psro)(
            **{
                **asdict(rp.psro),
                "seed": args.seed,
                "meta_solver": args.meta_solver or rp.psro.meta_solver,
                "iters": args.iters or rp.psro.iters,
            }
        ),
        risk_modes=risk_modes,
        cptg=rp.cptg,
        multinash=multinash,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "run_params.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(rp), f, indent=2)

    rng = make_rng(args.seed)

    # ---------------- Stage 0: build grid ----------------
    with StageTimer("Stage 0: Build grid + obstacles"):
        grid = GridWorld(H=rp.grid.H, W=rp.grid.W)
        add_multimodal_obstacles(grid, layout=rp.grid.obstacle_layout)
        log(f"Grid size = {grid.H}x{grid.W}, obstacles = {int(grid.obstacles.sum())}")

    # Sample candidates + precompute visibility
    with StageTimer("Stage 0b: Sample sensor candidates"):
        candidates = grid.sample_sensor_candidates(
            M=rp.sensors.M, wall_bias=rp.sensors.wall_bias, rng=rng
        )
        log(f"Sampled candidates M={len(candidates)}")

    with StageTimer("Stage 0c: Precompute visibility (this can be slow)"):
        log(
            f"Orientations={len(rp.sensors.orientations_deg)}, "
            f"range_cells={rp.sensors.range_cells}, fov={rp.sensors.fov_deg}"
        )
        vis = precompute_visibility(
            grid=grid,
            candidates=candidates,
            orientations_deg=rp.sensors.orientations_deg,
            fov_deg=rp.sensors.fov_deg,
            range_cells=rp.sensors.range_cells,
        )
        log(f"Visibility entries = {len(vis)} (M * orientations)")

    # ---------------- Initialize populations ----------------
    # Initial robot path: ignore sensors (risk=0)
    zero_risk = np.zeros((grid.H, grid.W), dtype=float)
    init_path = robot_best_response(
        grid=grid,
        start=rp.grid.start_a,
        goal=rp.grid.goal_a,
        expected_risk_map=zero_risk,
        risk_w=rp.outer_cost.risk_w,
        allow_diagonal=rp.grid.allow_diagonal,
    )

    # Initial sensor config: best-respond to this single path occupancy
    occ0 = occupancy_from_mixture(
        grid_shape=(grid.H, grid.W),
        paths=[init_path],
        weights=np.array([1.0]),
    )
    init_sensor = sensor_best_response(
        occ_map=occ0,
        vis=vis,
        K=rp.sensors.K,
        beta=rp.sensors.beta_detection,
        rng=rng,
    )
    # IMPORTANT: compute exposure map once so clustering can use it later
    init_sensor.compute_exposure_map(vis, grid_shape=(grid.H, grid.W))

    # ---------------- Stage 2: PSRO outer loop ----------------
    with StageTimer("Stage 2: PSRO outer loop (Robot vs Sensors)"):
        result = psro_lite_genbr(
            grid=grid,
            vis=vis,
            init_robot_paths=[init_path],
            init_sensor_policies=[init_sensor],
            start=rp.grid.start_a,
            goal=rp.grid.goal_a,
            cost_params=rp.outer_cost,
            psro_params=rp.psro,
            sensor_K=rp.sensors.K,
            sensor_beta=rp.sensors.beta_detection,
            allow_diagonal=rp.grid.allow_diagonal,
            rng=rng,
            save_dir=args.save_dir,
        )
        log(f"PSRO finished. Robot pop={len(result['robot_population'])}, Sensor pop={len(result['sensor_population'])}")
        log(f"Final exploitability = {result['exploitability_trace'][-1]:.6f}")

    if not args.no_plots:
        plot_exploitability(
            result["exploitability_trace"],
            os.path.join(args.save_dir, "psro_exploitability.png"),
        )
        plot_grid_and_sensors(
            grid=grid,
            candidates=candidates,
            sensor_policies=result["sensor_population"],
            q=result["q"],
            out_path=os.path.join(args.save_dir, "final_sensors.png"),
        )

    # Ensure exposure maps exist (PSRO populates more sensors)
    for sp in result["sensor_population"]:
        sp.compute_exposure_map(vis, grid_shape=(grid.H, grid.W))

    
    if args.outer_only:
        print("\n=== OUTER ONLY DONE ===")
        print(f"Saved to: {args.save_dir}")
        print(f"Final exploitability: {result['exploitability_trace'][-1]:.6f}")
        return

# ---------------- Stage 3: risk modes ----------------
    with StageTimer("Stage 3: Risk mode clustering"):
        for sp in result["sensor_population"]:
            sp.compute_exposure_map(vis, grid_shape=(grid.H, grid.W))

        modes = cluster_sensor_mixture(
            grid=grid,
            sensor_population=result["sensor_population"],
            q=result["q"],
            L=rp.risk_modes.L,
            max_iters=rp.risk_modes.max_kmedoids_iters,
            rng=rng,
        )
        log(f"Extracted L={len(modes)} risk modes")

    with open(os.path.join(args.save_dir, "risk_modes.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "weight": float(m["weight"]),
                    "members": m["members"],
                    "representative": m["representative"],
                }
                for m in modes
            ],
            f,
            indent=2,
        )

    # ---------------- Stage 4: CPTG + MultiNash-PF (lite) ----------------
    obstacle_boxes_m = grid_to_obstacle_boxes_m(grid, cell_size_m=rp.grid.cell_size_m)

    def cell_to_xy(cell):
        r, c = cell
        return np.array([(c + 0.5) * rp.grid.cell_size_m, (r + 0.5) * rp.grid.cell_size_m], dtype=float)

    starts = [cell_to_xy(rp.grid.start_a), cell_to_xy(rp.grid.start_b)]
    goals = [cell_to_xy(rp.grid.goal_a), cell_to_xy(rp.grid.goal_b)]

    all_mode_solutions: List[Dict[str, Any]] = []
    with StageTimer("Stage 4: MultiNash-PF per risk mode"):
        for mi, mode in enumerate(modes):
            with StageTimer(f"MultiNash mode {mi} (weight={mode['weight']:.3f})"):
                sols = multinash_pf_solve(
                    grid=grid,
                    risk_map=mode["risk_map"],
                    obstacle_boxes_m=obstacle_boxes_m,
                    starts=starts,
                    goals=goals,
                    cell_size_m=rp.grid.cell_size_m,
                    cptg_params=rp.cptg,
                    multinash_params=rp.multinash,
                    rng=rng,
                )
                log(f"Mode {mi}: found {len(sols)} solutions")

    print("\n=== DONE ===")
    print(f"Saved to: {args.save_dir}")
    print(f"Final exploitability: {result['exploitability_trace'][-1]:.6f}")
    print(f"Risk modes: {len(modes)}")
    for pack in all_mode_solutions:
        print(f"  Mode {pack['mode_index']}: {len(pack['solutions'])} local solutions")


if __name__ == "__main__":
    main()
