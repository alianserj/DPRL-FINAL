# Lite Gen-BR (planning-based PSRO) + 2-robot CPTG + MultiNash-PF (lite)

This repo is a **laptop-first** end-to-end pipeline that matches your outline:

1) **Outer adversarial zero-sum game**: robot path planner vs sensor placement/orientation adversary  
   - **Robot BR**: A* on weighted grid (movement + λ·expected risk)  
   - **Sensor BR**: greedy selection of K sensors using a diminishing-returns detection model  
   - **Outer solver**: PSRO-style loop with pluggable meta-solver:
     - `sfp` = Smooth Fictitious Play (softmaxed counts)  
     - `prd` = Projected Replicator Dynamics  
     - `lp`  = exact restricted-game zero-sum solve via LP (SciPy linprog)

2) **Inner interaction game** (per risk-mode): 2-robot **Constrained Potential Trajectory Game (CPTG)**  
   - potential = sum of both robots' costs (goal tracking + control + risk integral)
   - constraints = speed limits, obstacle avoidance, inter-robot collision avoidance

3) **MultiNash-PF (lite implementation)**:
   - implicit-PF-style sampling around references
   - clustering with (discrete) **Fréchet distance**
   - refinement with SciPy `SLSQP` (instead of IPOPT) to get multiple local feasible minima

> Notes:
> - This is a *faithful pipeline* with **practical substitutions** for tooling (IPOPT → SciPy).
> - The code is modular and easy to swap components (oracle types, meta-solvers, optimizers).

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --meta_solver sfp --iters 25 --seed 0 --save_dir runs/demo
```

Artifacts will be written to the `runs/...` directory:
- PSRO exploitability trace
- final robot/sensor mixtures
- clustered risk modes
- MultiNash solutions (multiple local equilibria per mode)
- plots (PNG)

---

## Directory layout

- `main.py` — orchestrates the full pipeline
- `params.py` — all hyperparameters in one place
- `lite_genbr/` — library code
  - `env/` — grid + obstacles
  - `sensors/` — visibility + sensor BR
  - `planning/` — A* + robot BR + occupancy
  - `psro/` — meta-game, meta-solvers, PSRO loop, metrics
  - `clustering/` — sensor-mixture clustering into risk modes
  - `multinash/` — CPTG cost/constraints + implicit PF + Fréchet + refinement
  - `viz/` — plotting utilities
- `tests/` — small sanity tests

---

## License

MIT (see `LICENSE`).
## Useful run modes

Outer game only (fast):

```bash
python main.py --outer_only --meta_solver sfp --iters 25 --seed 0 --save_dir runs/outer_only
```

Full pipeline but smaller MultiNash budgets (faster):

```bash
python main.py --meta_solver sfp --iters 10 --seed 0 --save_dir runs/fast_full \
  --risk_modes_L 2 \
  --multinash_J 10 --multinash_refine_iters 5 \
  --multinash_max_solutions 3 --multinash_solver_maxiter 80
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
