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
from lite_genbr.viz.plot_outer_game import (
    plot_world,
    plot_heatmap_with_paths,
    plot_mixture_bars,
    plot_payoff_matrix,
    plot_top_sensor_configs,
    expected_risk_maps_from_mixture,
)
from lite_genbr.utils.rng import make_rng


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
    grid = GridWorld(H=rp.grid.H, W=rp.grid.W)
    add_multimodal_obstacles(grid, layout=rp.grid.obstacle_layout)

    # Sample candidates + precompute visibility
    candidates = grid.sample_sensor_candidates(
        M=rp.sensors.M, wall_bias=rp.sensors.wall_bias, rng=rng
    )
    vis = precompute_visibility(
        grid=grid,
        candidates=candidates,
        orientations_deg=rp.sensors.orientations_deg,
        fov_deg=rp.sensors.fov_deg,
        range_cells=rp.sensors.range_cells,
    )

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

# ---------------- Outer-game visual diagnostics ----------------
    if not args.no_plots:
        # (1) World view
        plot_world(
            grid=grid,
            start=rp.grid.start_a,
            goal=rp.grid.goal_a,
            candidates=candidates,
            out_path=os.path.join(args.save_dir, "world_outer.png"),
            title="Outer game: world + robot start/goal + sensor candidates",
        )

        # (2) Mixture weight summaries
        plot_mixture_bars(
            weights=result["p"],
            out_path=os.path.join(args.save_dir, "robot_mixture_weights.png"),
            title="Robot mixture p (top mass paths)",
        )
        plot_mixture_bars(
            weights=result["q"],
            out_path=os.path.join(args.save_dir, "sensor_mixture_weights.png"),
            title="Sensor mixture q (top mass configs)",
        )

        # (3) Payoff matrix snapshot (row payoff = -cost)
        plot_payoff_matrix(
            payoff=result["payoff_matrix"],
            out_path=os.path.join(args.save_dir, "payoff_matrix.png"),
        )

        # (4) Expected risk maps under sensor mixture q
        exp_count, det_prob = expected_risk_maps_from_mixture(
            grid=grid,
            sensor_population=result["sensor_population"],
            q=result["q"],
            vis=vis,
            beta=rp.sensors.beta_detection,
        )

        # (5) A final robot best-response path to the *expected* detection probability map
        br_path_final = robot_best_response(
            grid=grid,
            start=rp.grid.start_a,
            goal=rp.grid.goal_a,
            expected_risk_map=det_prob,
            risk_w=rp.outer_cost.risk_w,
            allow_diagonal=rp.grid.allow_diagonal,
        )

        plot_heatmap_with_paths(
            grid=grid,
            heat=exp_count,
            out_path=os.path.join(args.save_dir, "expected_exposure_count.png"),
            title="Expected exposure count E_q[c(x)] (higher = more sensors see the cell)",
            starts_goals=[(rp.grid.start_a, rp.grid.goal_a)],
            paths=[br_path_final],
            path_labels=["Robot BR (to expected risk)"],
        )

        plot_heatmap_with_paths(
            grid=grid,
            heat=det_prob,
            out_path=os.path.join(args.save_dir, "expected_detection_prob.png"),
            title="Expected detection probability E_q[1-exp(-beta*c(x))]",
            starts_goals=[(rp.grid.start_a, rp.grid.goal_a)],
            paths=[br_path_final],
            path_labels=["Robot BR (to expected risk)"],
        )

        # (6) Robot mixture occupancy heatmap (where the robot population tends to go)
        occ = occupancy_from_mixture(
            grid_shape=(grid.H, grid.W),
            paths=result["robot_population"],
            weights=result["p"],
        )

        # overlay top-k paths by probability mass
        p = np.array(result["p"], dtype=float)
        topk = int(min(5, len(p)))
        top_idx = list(np.argsort(-p)[:topk])
        top_paths = [result["robot_population"][int(i)] for i in top_idx]
        top_labels = [f"path {int(i)} (p={p[int(i)]:.2f})" for i in top_idx]

        plot_heatmap_with_paths(
            grid=grid,
            heat=occ,
            out_path=os.path.join(args.save_dir, "robot_occupancy.png"),
            title="Robot mixture occupancy E_p[visits(x)] (hot = visited often)",
            starts_goals=[(rp.grid.start_a, rp.grid.goal_a)],
            paths=top_paths,
            path_labels=top_labels,
            alpha=0.9,
        )

        # (7) Top sensor configs with FOV cones (what the learned sensor mixture looks like)
        plot_top_sensor_configs(
            grid=grid,
            candidates=candidates,
            sensor_population=result["sensor_population"],
            q=result["q"],
            orientations_deg=rp.sensors.orientations_deg,
            fov_deg=rp.sensors.fov_deg,
            range_cells=rp.sensors.range_cells,
            out_path=os.path.join(args.save_dir, "top_sensor_configs.png"),
            top_k=3,
        )

        
        if args.outer_only:
            print("\n=== OUTER ONLY DONE ===")
            print(f"Saved to: {args.save_dir}")
            print(f"Final exploitability: {result['exploitability_trace'][-1]:.6f}")
            raise SystemExit(0)

    # ---------------- Stage 3: risk modes ----------------
        modes = cluster_sensor_mixture(
            grid=grid,
            sensor_population=result["sensor_population"],
            q=result["q"],
            L=rp.risk_modes.L,
            max_iters=rp.risk_modes.max_kmedoids_iters,
            rng=rng,
        )
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
        for mi, mode in enumerate(modes):
            risk_map = mode["risk_map"]
            sols = multinash_pf_solve(
                grid=grid,
                risk_map=risk_map,
                obstacle_boxes_m=obstacle_boxes_m,
                starts=starts,
                goals=goals,
                cell_size_m=rp.grid.cell_size_m,
                cptg_params=rp.cptg,
                multinash_params=rp.multinash,
                rng=rng,
            )
            out_json = os.path.join(args.save_dir, f"multinash_mode_{mi}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(sols, f, indent=2)
            all_mode_solutions.append({"mode_index": mi, "solutions": sols})

            if not args.no_plots:
                plot_multinash_solutions(
                    grid=grid,
                    obstacle_boxes_m=obstacle_boxes_m,
                    sols=sols,
                    out_path=os.path.join(args.save_dir, f"multinash_mode_{mi}.png"),
                    cell_size_m=rp.grid.cell_size_m,
                )

        print("\n=== DONE ===")
        print(f"Saved to: {args.save_dir}")
        print(f"Final exploitability: {result['exploitability_trace'][-1]:.6f}")
        print(f"Risk modes: {len(modes)}")
        for pack in all_mode_solutions:
            print(f"  Mode {pack['mode_index']}: {len(pack['solutions'])} local solutions")


if __name__ == "__main__":
    main()
