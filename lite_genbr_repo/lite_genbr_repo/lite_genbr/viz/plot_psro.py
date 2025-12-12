from __future__ import annotations
import os
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_exploitability(trace: List[float], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(list(range(len(trace))), trace)
    plt.xlabel("PSRO iteration")
    plt.ylabel("Exploitability (NashConv)")
    plt.title("Outer game exploitability over PSRO iterations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
