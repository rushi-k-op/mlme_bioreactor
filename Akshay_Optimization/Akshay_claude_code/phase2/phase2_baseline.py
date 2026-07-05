"""
Phase 2 - Non-Adaptive Baseline
================================
Mandatory task covered in this file:
  Task 5: Compare the BO loop against a non-adaptive baseline (e.g. Sobol or
          Latin Hypercube sampling spent across the same total budget) and
          discuss the result.

This script purposely spends the WHOLE budget on random (LHS) pilot-only
recipes, with no learning between iterations. It is meant to be the "dumb"
reference that phase3_bo.py must beat. That comparison plot is produced by
phase3_bo.py (compare_vs_baseline.png), which loads this script's output CSV.

Fixes applied vs. the original script:
  - FIX: the whole 15000 EUR budget is now used (old code floor-divided
    15000 // 2000 = 7 runs = 14000 EUR, silently leaving 1000 EUR unspent,
    which is not a fair equal-budget comparison against Phase 3).
  - FIX: stdout log file handle is now closed properly at the end.
  - ADDED: a fixed random seed argument so results are reproducible and so
    multiple baseline seeds can be produced to match Phase 3's multi-seed
    reporting (mandatory Task 4 asks for the BO loop to be run multiple
    times - the baseline should be compared on the same footing).
"""
import pandas as pd
import numpy as np
from scipy.stats import qmc
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from client import BioreactorClient, USER, PASSWORD, BASE_URL

class Logger(object):
    def __init__(self, filename="baseline_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

BOUNDS = {
    "T": (20.0, 60.0), "pH": (3.0, 9.5),
    "F1": (0.0, 2.0), "F2": (0.0, 2.0), "F3": (0.0, 2.0)
}

TOTAL_BUDGET = 15000
PILOT_COST = 2000


def generate_lhs_samples(n_samples: int, bounds: dict, seed: int = None) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    sample_unit = sampler.random(n=n_samples)
    lower_bounds = [b[0] for b in bounds.values()]
    upper_bounds = [b[1] for b in bounds.values()]
    samples_scaled = qmc.scale(sample_unit, lower_bounds, upper_bounds)
    return pd.DataFrame(samples_scaled, columns=bounds.keys())


def run_baseline(seed: int = 0) -> pd.DataFrame:
    client = BioreactorClient(BASE_URL)
    print("Logging into the bioreactor...")
    client.login(USER, PASSWORD)

    num_runs = TOTAL_BUDGET // PILOT_COST
    leftover = TOTAL_BUDGET - num_runs * PILOT_COST
    print(f"Budget: {TOTAL_BUDGET} EUR. Affordable Pilot runs: {num_runs} "
          f"(leaves {leftover} EUR unused - pilot cost does not divide budget evenly).")

    recipes_df = generate_lhs_samples(num_runs, BOUNDS, seed=seed)

    best_y = 0.0
    total_cost = 0.0
    results = []

    print(f"\nStarting Baseline LHS Campaign (Pilot Only, seed={seed})...")
    for idx, row in recipes_df.iterrows():
        recipe_dict = row.to_dict()
        print(f"  Run {idx+1}/{num_runs}...", end=" ")

        res = client.run(scale="pilot", **recipe_dict)
        y_val = res.get("Y")
        cost_val = res.get("cost_eur")
        if y_val is None or cost_val is None:
            raise KeyError(f"Unexpected response format: {res}")

        total_cost += cost_val
        if y_val > best_y:
            best_y = y_val

        print(f"Result: Y = {y_val:.4f} (Best so far: {best_y:.4f})")

        results.append({
            "Iteration": idx + 1, "scale": "pilot",
            "T": recipe_dict["T"], "pH": recipe_dict["pH"],
            "F1": recipe_dict["F1"], "F2": recipe_dict["F2"], "F3": recipe_dict["F3"],
            "Y": y_val, "Best_Y": best_y, "Cumulative_Cost": total_cost
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv("baseline_results.csv", index=False)

    print("\n=====================================")
    print(" BASELINE CAMPAIGN COMPLETE")
    print("=====================================")
    print(f"Total Cost Spent: {total_cost} EUR")
    print(f"Best Pilot Y Found: {best_y:.4f} g/L")
    print("Data saved to 'baseline_results.csv'")
    return df_results


def plot_baseline_results(df: pd.DataFrame):
    print("\nGenerating Phase 2 visualization...")
    plt.figure(figsize=(9, 6), dpi=300)
    costs = [0] + df['Cumulative_Cost'].tolist()
    yields = [0] + df['Best_Y'].tolist()

    plt.step(costs, yields, where='post', color='#7f7f7f', linewidth=2.5,
              label='Random Pilot Search (LHS baseline)')
    plt.axhline(y=14.0, color='#d62728', linestyle='--', alpha=0.9, linewidth=1.5,
                label="Grading Target (14.0 g/L)")

    plt.title("Phase 2: Baseline (Non-Adaptive) Optimization Trajectory",
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Cumulative Capital Expenditure (EUR)", fontsize=12, fontweight='bold')
    plt.ylabel("Maximum Verified Pilot Yield (g/L)", fontsize=12, fontweight='bold')
    plt.xlim(0, max(TOTAL_BUDGET, max(costs)))
    plt.ylim(0, max(15.0, max(yields) + 2.0))
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig("baseline_trajectory.png", bbox_inches='tight')
    plt.close()
    print(" -> Saved visualization to 'baseline_trajectory.png'")


if __name__ == "__main__":
    logger = Logger("baseline_output.txt")
    sys.stdout = logger
    try:
        df = run_baseline(seed=0)
        plot_baseline_results(df)
    finally:
        sys.stdout = logger.terminal
        logger.close()