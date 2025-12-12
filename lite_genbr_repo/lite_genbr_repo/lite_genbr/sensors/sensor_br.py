from __future__ import annotations

from typing import List, Tuple
import numpy as np

from lite_genbr.sensors.visibility import VisDict, VisKey
from lite_genbr.sensors.sensor_policy import SensorPolicy


def sensor_best_response(
    occ_map: np.ndarray,
    vis: VisDict,
    K: int,
    beta: float,
    rng: np.random.Generator,
) -> SensorPolicy:
    """
    Greedy sensor placement to maximize a diminishing-returns detection objective.

    Let c(x) = number of sensors seeing cell x.
    Detection utility per cell: f(c)=1-exp(-beta*c).
    Objective: sum_x occ[x] * f(c(x)).

    Greedy adds the action (candidate, orient) with highest marginal gain given current c(x).
    """
    H, W = occ_map.shape
    all_actions: List[VisKey] = list(vis.keys())

    chosen: List[VisKey] = []
    chosen_set = set()
    c_map = np.zeros((H, W), dtype=np.int16)
    one_minus = float(1.0 - np.exp(-beta))

    for _ in range(K):
        weight_map = occ_map * np.exp(-beta * c_map.astype(float))
        best_gain = -1.0
        best_act = None

        order = rng.permutation(len(all_actions))
        for idx in order:
            act = all_actions[int(idx)]
            if act in chosen_set:
                continue
            mask = vis[act]
            gain = one_minus * float(weight_map[mask].sum())
            if gain > best_gain:
                best_gain = gain
                best_act = act

        if best_act is None:
            break
        chosen.append(best_act)
        chosen_set.add(best_act)
        c_map += vis[best_act].astype(np.int16)

    return SensorPolicy(sensors=chosen)
