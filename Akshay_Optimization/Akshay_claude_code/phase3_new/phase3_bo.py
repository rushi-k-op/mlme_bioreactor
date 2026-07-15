"""
Phase 3 - Multi-Fidelity Bayesian Optimization across Reactor Scales
=====================================================================
Mandatory tasks covered in this file:
  Task 3: Design and implement a Bayesian Optimization loop that selects, at
          EACH iteration, both the next recipe AND the next scale.
  Task 4: Run the BO loop MULTIPLE TIMES and report the best pilot product
          amount obtained, the total cost spent, and the per-iteration
          trajectory of both quantities.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, linregress, qmc
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from client import BioreactorClient, USER, PASSWORD, BASE_URL

# ==========================================
# CUSTOM TERMINAL LOGGER
# ==========================================
class Logger(object):
    def __init__(self, filename="bo_output.txt"):
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

# ==========================================
# CONFIGURATION
# ==========================================
HISTORY_FILE = "exploratory_data.csv"   
BO_BUDGET = 15000.0                      # Match Felix's budget to maximize yield potential
N_SEEDS = 10                             
TARGET_YIELD = 14.5                   # Lowered slightly to focus on beating the 14.0 baseline

BOUNDS_LIST = [(20.0, 60.0), (3.0, 9.5), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
VAR_NAMES = ["T", "pH", "F1", "F2", "F3"]
SCALE_COSTS = {"micro": 10, "bench": 500, "pilot": 2000}

MICRO_SCREEN_BUDGET = 2500.0   # 250 micro runs - much better 5D coverage (was 300/30 runs)
MIN_RESERVE_FOR_PILOT = SCALE_COSTS["pilot"]  

DEFAULT_NOISE = {"micro": 5e-3, "bench": 5e-3, "pilot": 1e-3}  


# ==========================================
# NORMALIZATION HELPERS
# ==========================================
def normalize(x_raw):
    return np.array([(x_raw[i] - BOUNDS_LIST[i][0]) / (BOUNDS_LIST[i][1] - BOUNDS_LIST[i][0])
                      for i in range(5)])

def denormalize(x_norm):
    return np.array([BOUNDS_LIST[i][0] + x_norm[i] * (BOUNDS_LIST[i][1] - BOUNDS_LIST[i][0])
                      for i in range(5)])

def recipe_dict_from_norm(x_norm):
    raw = denormalize(x_norm)
    return {name: float(val) for name, val in zip(VAR_NAMES, raw)}


# ==========================================
# CALIBRATION
# ==========================================
def fit_calibration(df_history: pd.DataFrame) -> dict:
    calib = {"pilot": (1.0, 0.0, 1.0, 0.0)}  

    df = df_history.copy()
    for v in VAR_NAMES:
        df[f"_{v}_r"] = df[v].round(4)
    key_cols = [f"_{v}_r" for v in VAR_NAMES]

    means = df.groupby(key_cols + ["scale"])["Y"].mean().reset_index()
    wide = means.pivot_table(index=key_cols, columns="scale", values="Y")

    pilot_scale_guess = (wide["pilot"].std() if "pilot" in wide.columns
                          and wide["pilot"].notna().sum() > 1 else 5.0)

    for cheap_scale in ["bench", "micro"]:
        if cheap_scale in wide.columns and "pilot" in wide.columns:
            pair = wide[[cheap_scale, "pilot"]].dropna()
            if len(pair) >= 3:
                slope, intercept, r, p, se = linregress(pair[cheap_scale], pair["pilot"])
                residuals = pair["pilot"] - (slope * pair[cheap_scale] + intercept)
                residual_var = float(np.var(residuals, ddof=max(len(pair) - 2, 1)))
                calib[cheap_scale] = (slope, intercept, r ** 2, max(residual_var, 1e-4))
                continue
        calib[cheap_scale] = (0.0, 0.0, 0.0, (pilot_scale_guess ** 2) * 10)
    return calib


def estimate_noise_variance(df_history: pd.DataFrame) -> dict:
    df = df_history.copy()
    for v in VAR_NAMES:
        df[f"_{v}_r"] = df[v].round(4)
    key_cols = [f"_{v}_r" for v in VAR_NAMES]
    var_by_recipe_scale = df.groupby(key_cols + ["scale"])["Y"].var()
    noise = var_by_recipe_scale.groupby("scale").mean().to_dict()
    for s in SCALE_COSTS:
        if s not in noise or pd.isna(noise[s]):
            noise[s] = DEFAULT_NOISE[s]
    return noise


def to_pilot_equivalent(df_history: pd.DataFrame, calib: dict, raw_noise: dict):
    X, y_eq, var_eq, scales = [], [], [], []
    for _, row in df_history.iterrows():
        scale = row["scale"]
        slope, intercept, r2, residual_var = calib.get(scale, (1.0, 0.0, 1.0, 0.0))
        if scale != "pilot" and abs(slope) < 1e-6:
            continue  

        y_val = row["Y"]
        y_conv = y_val if scale == "pilot" else slope * y_val + intercept

        if scale == "pilot":
            var_conv = raw_noise.get("pilot", DEFAULT_NOISE["pilot"])
        else:
            base_var = raw_noise.get(scale, DEFAULT_NOISE.get(scale, 1e-2))
            var_conv = max((slope ** 2) * base_var + residual_var, 1e-4)

        X.append(normalize([row[v] for v in VAR_NAMES]))
        y_eq.append(y_conv)
        var_eq.append(var_conv)
        scales.append(scale)

    return np.array(X), np.array(y_eq).reshape(-1, 1), np.array(var_eq), scales
# ==========================================
# GP + ACQUISITION
# ==========================================
def build_gp(X, y_eq, var_eq, random_state=None):
    kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=[0.3] * 5, length_scale_bounds=(0.05, 2.0))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=var_eq, n_restarts_optimizer=6,
                                    normalize_y=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X, y_eq)
    return gpr

def expected_improvement(X_norm, best_y_eq, gpr, xi=0.05):
    mu, sigma = gpr.predict(X_norm, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = mu - best_y_eq - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-9] = 0.0
    return ei

def optimize_acquisition(best_y_eq, gpr, X_history=None, seed=None):
    bounds = [(0.0, 1.0)] * 5
    from scipy.optimize import differential_evolution, minimize

    def objective(x):
        x_reshaped = x.reshape(1, -1)
        # Standard exploration: Expected Improvement
        ei = expected_improvement(x_reshaped, best_y_eq, gpr)
        return -float(ei.item())

    # 1. Memory Check: Always evaluate our historically known points to prevent DE failure
    best_x = None
    best_val = float('inf')
    if X_history is not None and len(X_history) > 0:
        for x_h in X_history:
            v = objective(x_h)
            if v < best_val:
                best_val = v
                best_x = x_h.copy()

    # 2. Global Search: Run Differential Evolution (seeded for reproducibility)
    res = differential_evolution(objective, bounds, maxiter=50, popsize=20,
                                  polish=True, seed=seed)
    
    # 3. Guardrail: If the optimizer got lost, fall back to our historical best
    if best_x is None or res.fun < best_val:
        best_x = res.x
        best_val = res.fun
        
    # 4. Polish: Use gradient descent to climb to the exact mathematical peak
    local_res = minimize(objective, best_x, bounds=bounds, method="L-BFGS-B")
    if local_res.fun < best_val:
        best_x = local_res.x

    return best_x


# ==========================================
# ONE FULL BO CAMPAIGN (ONE SEED)
# ==========================================
def run_bo_campaign(client, seed: int, budget: float, base_history: pd.DataFrame, log=print):
    rng = np.random.default_rng(seed)
    df_history = base_history.copy()

    budget_spent_this_run = 0.0
    trajectory = []
    best_pilot_y = df_history[df_history["scale"] == "pilot"]["Y"].max() \
        if (df_history["scale"] == "pilot").any() else 0.0
    pilot_evals = 0
    iteration = 0

    def log_row(scale, recipe, y_val, cost, reason):
        nonlocal budget_spent_this_run
        budget_spent_this_run += cost
        trajectory.append({
            "seed": seed, "iteration": iteration, "scale": scale,
            **recipe, "Y": y_val, "cost_eur": cost,
            "cumulative_cost": budget_spent_this_run,
            "best_pilot_y_so_far": best_pilot_y, "reason": reason
        })

    log(f"\n===== SEED {seed}: STAGE A - cheap wide MICRO screening =====")
    n_micro = int(MICRO_SCREEN_BUDGET // SCALE_COSTS["micro"])
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    screen_unit = sampler.random(n=n_micro)
    lower = [b[0] for b in BOUNDS_LIST]
    upper = [b[1] for b in BOUNDS_LIST]
    screen_raw = qmc.scale(screen_unit, lower, upper)

    for i in range(n_micro):
        if budget_spent_this_run + SCALE_COSTS["micro"] > budget:
            break
        recipe = {name: float(v) for name, v in zip(VAR_NAMES, screen_raw[i])}
        res = client.run(scale="micro", **recipe)
        y_val = res.get("Y", res.get("y"))
        new_row = pd.DataFrame([{**recipe, "scale": "micro", "Y": y_val,
                                  "cost_eur": SCALE_COSTS["micro"]}])
        df_history = pd.concat([df_history, new_row], ignore_index=True)
        log_row("micro", recipe, y_val, SCALE_COSTS["micro"], "wide screening")
        iteration += 1

    log(f"  Screening complete: {n_micro} micro runs, "
        f"{budget_spent_this_run:.0f}/{budget:.0f} EUR spent so far.")

    log(f"\n===== SEED {seed}: STAGE B & C - Merve's Scheduled BO =====")
    while budget_spent_this_run + min(SCALE_COSTS.values()) <= budget:
        calib = fit_calibration(df_history)
        raw_noise = estimate_noise_variance(df_history)
        X, y_eq, var_eq, scales_used = to_pilot_equivalent(df_history, calib, raw_noise)

        if len(X) < 4:
            log("  Not enough data yet to fit GP, skipping iteration.")
            break

        # Derive a fresh, deterministic sub-seed for this iteration from the
        # campaign's own RNG stream, so re-running the same `seed` always
        # reproduces the same GP fit and the same DE search.
        iter_seed = int(rng.integers(0, 2**31 - 1))
        gpr = build_gp(X, y_eq, var_eq, random_state=iter_seed)
        best_y_eq_so_far = float(np.max(y_eq))
        
        # Optimize Expected Improvement using our memory-reinforced algorithm
        next_norm = optimize_acquisition(best_y_eq_so_far, gpr, X_history=X, seed=iter_seed)
        pred_mean, pred_std = gpr.predict(next_norm.reshape(1, -1), return_std=True)
        recipe = recipe_dict_from_norm(next_norm)

        # MERVE'S SCHEDULE: 
        # Force bench runs until we have exactly enough money left for 5 Pilot runs
        budget_remaining = budget - budget_spent_this_run
        
        if budget_remaining >= (5 * SCALE_COSTS["pilot"]) + SCALE_COSTS["bench"]:
            chosen_scale = "bench"
            reason = "Scheduled Bench Region Finding"
        elif budget_remaining >= SCALE_COSTS["pilot"]:
            chosen_scale = "pilot"
            reason = "Scheduled Pilot Peak Climbing (Merve Strategy)"
        else:
            break # Budget exhausted

        cost = SCALE_COSTS[chosen_scale]
        log(f"\n--- Seed {seed} | Iter {iteration} | Spent {budget_spent_this_run:.0f}/{budget:.0f} EUR "
            f"| Best pilot Y so far: {best_pilot_y:.3f} ---")
        log(f"  Candidate recipe -> " + ", ".join(f"{k}={v:.2f}" for k, v in recipe.items()))
        log(f"  GP prediction (pilot-eq) -> {pred_mean.item():.3f} +/- {pred_std.item():.3f}")
        log(f"  SCALE CHOSEN -> {chosen_scale.upper()}  ({reason})")

        res = client.run(scale=chosen_scale, **recipe)
        y_val = res.get("Y", res.get("y"))
        log(f"  Result -> Actual Y = {y_val:.4f}")

        new_row = pd.DataFrame([{**recipe, "scale": chosen_scale, "Y": y_val, "cost_eur": cost}])
        df_history = pd.concat([df_history, new_row], ignore_index=True)

        if chosen_scale == "pilot":
            pilot_evals += 1
            if y_val > best_pilot_y:
                best_pilot_y = y_val
                log(f"  *** NEW BEST PILOT Y: {best_pilot_y:.4f} ***")

        log_row(chosen_scale, recipe, y_val, cost, reason)
        iteration += 1

    log(f"\n===== SEED {seed} COMPLETE: Best pilot Y = {best_pilot_y:.4f} g/L, "
        f"Total spent = {budget_spent_this_run:.0f} EUR, Pilot evals = {pilot_evals} =====")

    traj_df = pd.DataFrame(trajectory)
    return traj_df, best_pilot_y, budget_spent_this_run, pilot_evals, df_history


# ==========================================
# MULTI-SEED DRIVER (Mandatory Task 4)
# ==========================================
def run_multi_seed_bo(n_seeds=N_SEEDS, budget=BO_BUDGET):
    client = BioreactorClient(BASE_URL)
    print("Logging into the bioreactor...")
    client.login(USER, PASSWORD)

    try:
        base_history = pd.read_csv(HISTORY_FILE)
        if "Scale" in base_history.columns:
            base_history = base_history.rename(columns={"Scale": "scale"})
        base_history["scale"] = base_history["scale"].str.lower()
        if "cost_eur" not in base_history.columns and "Cost" in base_history.columns:
            base_history = base_history.rename(columns={"Cost": "cost_eur"})
    except FileNotFoundError:
        print(f"Error: could not find '{HISTORY_FILE}'. Run Phase 1 first.")
        return

    all_trajectories = []
    summary_rows = []

    for seed in range(n_seeds):
        traj_df, best_y, total_cost, pilot_evals, _ = run_bo_campaign(
            client, seed, budget, base_history, log=print)
        all_trajectories.append(traj_df)
        summary_rows.append({
            "seed": seed, "best_pilot_Y": best_y,
            "total_cost_eur": total_cost, "pilot_evaluations": pilot_evals,
            "reached_target": best_y >= TARGET_YIELD
        })

    full_traj = pd.concat(all_trajectories, ignore_index=True)
    full_traj.to_csv("bo_all_trajectories.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("bo_summary_per_seed.csv", index=False)

    print("\n=====================================")
    print(" MULTI-SEED BO CAMPAIGN COMPLETE")
    print("=====================================")
    print(summary_df.to_string(index=False))
    print(f"\nMean best pilot Y : {summary_df['best_pilot_Y'].mean():.3f} "
          f"+/- {summary_df['best_pilot_Y'].std():.3f} g/L")
    print(f"Mean total cost   : {summary_df['total_cost_eur'].mean():.1f} "
          f"+/- {summary_df['total_cost_eur'].std():.1f} EUR")
    print(f"Felix benchmark   : mean best pilot Y = 14 g/L @ mean cost = 15000 EUR")

    return full_traj, summary_df


# ==========================================
# PLOTTING
# ==========================================
def plot_bo_trajectories(full_traj: pd.DataFrame, summary_df: pd.DataFrame):
    print("\nGenerating Phase 3 visualizations...")

   # 1) Per-iteration trajectory of best-pilot-Y vs cumulative cost (MEAN ± STD)
    plt.figure(figsize=(9, 6), dpi=300)
    
    # Create a common cost grid to calculate the mean across different seed lengths
    from scipy.interpolate import interp1d
    common_cost = np.linspace(0, summary_df["total_cost_eur"].max(), 500)
    interp_yields = []
    
    for seed, g in full_traj.groupby("seed"):
        g = g.sort_values("cumulative_cost")
        # Track the highest verified pilot yield at each step
        running_max = g["Y"].where(g["scale"] == "pilot").cummax().ffill().fillna(0)
        
        # Interpolate this seed's performance onto the common cost grid
        f = interp1d(g["cumulative_cost"], running_max, kind="previous", 
                     bounds_error=False, fill_value=(0, running_max.iloc[-1]))
        interp_yields.append(f(common_cost))
        
    interp_yields = np.array(interp_yields)
    mean_yield = np.mean(interp_yields, axis=0)
    std_yield = np.std(interp_yields, axis=0)

    # Plot the Mean Line and the ± STD Shaded Band requested by Merve
    plt.plot(common_cost, mean_yield, color="#1f77b4", linewidth=2.5, label="Mean Best Verified Pilot Yield")
    plt.fill_between(common_cost, mean_yield - std_yield, mean_yield + std_yield, color="#1f77b4", alpha=0.2, label="± 1 Std Dev")

    plt.axhline(y=14.0, color='#7f7f7f', linestyle='--', linewidth=1.5, label="Felix Baseline (14.0 g/L)")
    plt.title("Phase 3: Multi-Fidelity BO - Mean Performance (10 Seeds)", fontsize=14, fontweight='bold')
    plt.xlabel("Cumulative Cost (EUR)", fontweight='bold')
    plt.ylabel("Best Verified Pilot Yield (g/L)", fontweight='bold')
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend(loc='lower right', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig("bo_trajectory.png", bbox_inches='tight')
    plt.close()

    # 2) Scale-selection pattern per iteration (one seed, seed 0, as illustrative example)
    g0 = full_traj[full_traj["seed"] == 0].sort_values("iteration")
    if not g0.empty:
        plt.figure(figsize=(11, 4), dpi=200)
        colors = {"micro": "#2ca02c", "bench": "#ff7f0e", "pilot": "#d62728"}
        for scale, sub in g0.groupby("scale"):
            plt.scatter(sub["iteration"], [scale] * len(sub), color=colors[scale],
                       s=40, label=scale.upper())
        plt.title("Phase 3: Per-Iteration Scale-Selection Pattern (Seed 0)",
                  fontsize=13, fontweight='bold')
        plt.xlabel("Iteration", fontweight='bold')
        plt.ylabel("Scale chosen", fontweight='bold')
        plt.grid(alpha=0.3, axis='x', linestyle='--')
        plt.tight_layout()
        plt.savefig("bo_scale_selection_pattern.png", bbox_inches='tight')
        plt.close()

    # 3) Summary bar: mean best pilot Y and mean cost vs Felix benchmark
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    labels = ["Our BO\n(mean)", "Felix\nbenchmark"]
    yields = [summary_df["best_pilot_Y"].mean(), 14.0]
    yerr = [summary_df["best_pilot_Y"].std(), 0]
    costs = [summary_df["total_cost_eur"].mean(), 15000]
    costerr = [summary_df["total_cost_eur"].std(), 0]

    axes[0].bar(labels, yields, yerr=yerr, color=["#1f77b4", "#7f7f7f"], capsize=6)
    axes[0].axhline(14.0, color='#d62728', linestyle='--', linewidth=1)
    axes[0].set_title("Mean Best Pilot Yield (g/L)", fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')

    axes[1].bar(labels, costs, yerr=costerr, color=["#1f77b4", "#7f7f7f"], capsize=6)
    axes[1].set_title("Mean Total Cost (EUR)", fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    fig.suptitle("Phase 3 vs. Felix's Benchmark (Table 2)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("bo_vs_felix_benchmark.png", bbox_inches='tight')
    plt.close()

    # 4) BO vs Phase-2 non-adaptive baseline (mandatory Task 5, cross-referenced here)
    try:
        baseline_df = pd.read_csv("baseline_results.csv")
        plt.figure(figsize=(9, 6), dpi=300)
        costs_b = [0] + baseline_df["Cumulative_Cost"].tolist()
        yields_b = [0] + baseline_df["Best_Y"].tolist()
        plt.step(costs_b, yields_b, where="post", color="#7f7f7f", linewidth=2.5,
                  label="Phase 2: Non-adaptive LHS baseline (pilot only)")

        best_seed = summary_df.loc[summary_df["best_pilot_Y"].idxmax(), "seed"]
        g_best = full_traj[full_traj["seed"] == best_seed].sort_values("cumulative_cost")
        running_max = g_best["Y"].where(g_best["scale"] == "pilot").cummax().ffill().fillna(0)
        plt.step(g_best["cumulative_cost"], running_max, where="post", color="#1f77b4",
                  linewidth=2.5, label=f"Phase 3: Multi-fidelity BO (seed {best_seed})")

        plt.axhline(y=TARGET_YIELD, color='#d62728', linestyle='--', linewidth=1.5,
                    label=f"Target ({TARGET_YIELD} g/L)")
        plt.title("BO vs. Non-Adaptive Baseline (equal budget)", fontsize=14, fontweight='bold')
        plt.xlabel("Cumulative Cost (EUR)", fontweight='bold')
        plt.ylabel("Best Verified Pilot Yield (g/L)", fontweight='bold')
        plt.grid(alpha=0.4, linestyle='--')
        plt.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        plt.savefig("compare_vs_baseline.png", bbox_inches='tight')
        plt.close()
        print(" -> Saved 'compare_vs_baseline.png' (BO vs Phase 2 baseline)")
    except FileNotFoundError:
        print(" (Skipped BO-vs-baseline plot: run phase2_baseline.py first to produce "
              "'baseline_results.csv')")

    print(" -> Saved 'bo_trajectory.png', 'bo_scale_selection_pattern.png', "
          "'bo_vs_felix_benchmark.png'")


if __name__ == "__main__":
    logger = Logger("bo_output.txt")
    sys.stdout = logger
    try:
        full_traj, summary_df = run_multi_seed_bo(n_seeds=N_SEEDS, budget=BO_BUDGET)
        plot_bo_trajectories(full_traj, summary_df)
    finally:
        sys.stdout = logger.terminal
        logger.close()
