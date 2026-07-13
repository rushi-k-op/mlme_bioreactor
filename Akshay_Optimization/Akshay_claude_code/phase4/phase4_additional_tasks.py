"""
Phase 4 - Additional Tasks
===========================
Covers all "additional tasks" suggested in the project PDF, on top of the
already-completed mandatory tasks (Phases 1-3):

  A1: Compare acquisition functions (Expected Improvement vs Upper
      Confidence Bound) on the same simulator.
  A2: Compare surrogate model classes (pooled multi-scale GP vs
      independent per-scale GPs vs Random Forest).
  A3: Compare the multi-scale surrogate against a pilot-only BO loop at
      equal total cost - does the cheap-scale information actually help?
  A4: Visualize the algorithm's per-iteration scale-selection pattern
      (across ALL seeds, not just one) and discuss what drives the
      transitions between scales.
  A5: Global sensitivity analysis (Sobol indices) on the fitted surrogate.

IMPORTANT - BUDGET NOTE:
  A1, A2 and A3 each launch NEW multi-seed BO campaigns against the live
  simulator, so each one spends real EUR from your group's budget. A4 and
  A5 do NOT call the simulator at all - they re-analyze the CSVs you
  already produced ('bo_all_trajectories.csv', 'exploratory_data.csv'),
  so run those first/freely.

  Default seeds/budget per comparison arm are kept small on purpose so you
  don't blow your whole remaining budget on one comparison. Adjust via the
  CLI flags below once you know how much budget you have left.

USAGE:
    python phase4_additional_tasks.py --task a1 --seeds 5 --budget 8000
    python phase4_additional_tasks.py --task a2 --seeds 5 --budget 8000
    python phase4_additional_tasks.py --task a3 --seeds 5 --budget 15000
    python phase4_additional_tasks.py --task a4                 # free
    python phase4_additional_tasks.py --task a5                 # free
    python phase4_additional_tasks.py --task all --seeds 5 --budget 8000

Design note: every campaign here reuses the exact GP-building, calibration
and Merve-schedule (bench-until-exactly-5-pilot-runs-left) logic from
phase3_bo.py, so A1/A2/A3 differ from your Phase-3 result in ONLY the one
variable each task is meant to isolate (acquisition fn / surrogate class /
scale strategy). That's what makes the comparisons fair.
"""
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import RandomForestRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from client import BioreactorClient, USER, PASSWORD, BASE_URL

# Reuse everything already built and validated in Phase 3
from phase3_bo import (
    normalize, denormalize, recipe_dict_from_norm,
    fit_calibration, estimate_noise_variance, to_pilot_equivalent,
    build_gp, BOUNDS_LIST, VAR_NAMES, SCALE_COSTS, HISTORY_FILE, Logger,
)

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_SEEDS = 5
DEFAULT_BUDGET = 8000.0        # per-arm budget for A1 / A2 (kept modest)
PILOT_ONLY_BUDGET = 15000.0    # equal-footing budget for A3 (matches Phase 3)
MICRO_SCREEN_BUDGET = 300.0    # same wide-screening budget as Phase 3


def load_base_history() -> pd.DataFrame:
    df = pd.read_csv(HISTORY_FILE)
    if "Scale" in df.columns:
        df = df.rename(columns={"Scale": "scale"})
    df["scale"] = df["scale"].str.lower()
    if "cost_eur" not in df.columns and "Cost" in df.columns:
        df = df.rename(columns={"Cost": "cost_eur"})
    return df


# ==========================================
# A1: ACQUISITION FUNCTIONS (EI vs UCB)
# ==========================================
def acq_ei(X_norm, model, best_y, xi=0.05):
    mu, sigma = model.predict(X_norm, return_std=True)
    mu = np.asarray(mu).reshape(-1, 1)
    sigma = np.asarray(sigma).reshape(-1, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = mu - best_y - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-9] = 0.0
    return ei


def acq_ucb(X_norm, model, best_y=None, kappa=2.0):
    mu, sigma = model.predict(X_norm, return_std=True)
    mu = np.asarray(mu).reshape(-1, 1)
    sigma = np.asarray(sigma).reshape(-1, 1)
    return mu + kappa * sigma


ACQUISITIONS = {"EI": acq_ei, "UCB": acq_ucb}


def optimize_acquisition_generic(model, acquisition, best_y, X_history=None, kappa=2.0):
    bounds = [(0.0, 1.0)] * 5
    fn = ACQUISITIONS[acquisition]

    def objective(x):
        extra = {"kappa": kappa} if acquisition == "UCB" else {}
        val = fn(x.reshape(1, -1), model, best_y, **extra)
        return -float(np.asarray(val).item())

    best_x, best_val = None, float('inf')
    if X_history is not None and len(X_history) > 0:
        for xh in X_history:
            v = objective(xh)
            if v < best_val:
                best_val, best_x = v, xh.copy()

    res = differential_evolution(objective, bounds, maxiter=40, popsize=18, polish=True)
    if best_x is None or res.fun < best_val:
        best_x, best_val = res.x, res.fun

    local = minimize(objective, best_x, bounds=bounds, method="L-BFGS-B")
    if local.fun < best_val:
        best_x = local.x
    return best_x


# ==========================================
# A2: SURROGATE MODEL CLASSES
# ==========================================
class RFSurrogate:
    """Random-Forest surrogate exposing a GP-like predict(return_std=True)
    API: the mean and std are taken across the individual trees' outputs,
    which is a standard way to get an uncertainty estimate from a Random
    Forest for use inside an acquisition function."""
    def __init__(self, n_estimators=300, random_state=0):
        self.rf = RandomForestRegressor(n_estimators=n_estimators,
                                         random_state=random_state, n_jobs=-1)

    def fit(self, X, y, var=None):
        self.rf.fit(X, np.ravel(y))
        return self

    def predict(self, X, return_std=False):
        tree_preds = np.stack([t.predict(X) for t in self.rf.estimators_], axis=0)
        mu = tree_preds.mean(axis=0)
        if return_std:
            sigma = tree_preds.std(axis=0) + 1e-6
            return mu, sigma
        return mu


def build_surrogate(kind, X, y_eq, var_eq):
    if kind == "pooled_gp":
        return build_gp(X, y_eq, var_eq)
    elif kind == "rf":
        return RFSurrogate().fit(X, y_eq)
    else:
        raise ValueError(f"Unknown pooled surrogate kind: {kind}")


def build_per_scale_gps(df_history: pd.DataFrame) -> dict:
    """Fit one INDEPENDENT GP per scale on its own RAW (uncalibrated) Y
    values - i.e. with NO cross-scale information sharing at all. This is
    the contrast case against the pooled multi-fidelity GP: it isolates
    whether calibrating cheap scales onto the pilot scale is actually
    what's buying us performance, or whether per-scale models would do
    just as well."""
    models = {}
    for scale in ["micro", "bench", "pilot"]:
        sub = df_history[df_history["scale"] == scale]
        if len(sub) < 4:
            continue
        X = np.array([normalize([row[v] for v in VAR_NAMES]) for _, row in sub.iterrows()])
        y = sub["Y"].values.reshape(-1, 1)
        noise_floor = max(float(sub["Y"].var()), 1e-4) * 0.05
        models[scale] = build_gp(X, y, np.full(len(X), noise_floor))
    return models


# ==========================================
# GENERALIZED BO CAMPAIGN (drives A1, A2, A3)
# ==========================================
def run_generalized_campaign(client, seed, budget, base_history, acquisition="EI",
                              surrogate="pooled_gp", scale_mode="multi_fidelity",
                              kappa=2.0, log=print):
    rng = np.random.default_rng(seed)
    df_history = base_history.copy()
    budget_spent = 0.0
    trajectory = []
    best_pilot_y = (df_history[df_history["scale"] == "pilot"]["Y"].max()
                     if (df_history["scale"] == "pilot").any() else 0.0)
    pilot_evals = 0
    iteration = 0

    def log_row(scale, recipe, y_val, cost, reason):
        nonlocal budget_spent
        budget_spent += cost
        trajectory.append({
            "seed": seed, "iteration": iteration, "scale": scale, **recipe,
            "Y": y_val, "cost_eur": cost, "cumulative_cost": budget_spent,
            "best_pilot_y_so_far": best_pilot_y, "reason": reason,
            "acquisition": acquisition, "surrogate": surrogate, "scale_mode": scale_mode,
        })

    # ---- Stage A: wide micro screening, only for the multi-fidelity mode ----
    if scale_mode == "multi_fidelity":
        n_micro = int(MICRO_SCREEN_BUDGET // SCALE_COSTS["micro"])
        sampler = qmc.LatinHypercube(d=5, seed=seed)
        screen_unit = sampler.random(n=n_micro)
        lower = [b[0] for b in BOUNDS_LIST]
        upper = [b[1] for b in BOUNDS_LIST]
        screen_raw = qmc.scale(screen_unit, lower, upper)
        for i in range(n_micro):
            if budget_spent + SCALE_COSTS["micro"] > budget:
                break
            recipe = {name: float(v) for name, v in zip(VAR_NAMES, screen_raw[i])}
            res = client.run(scale="micro", **recipe)
            y_val = res.get("Y", res.get("y"))
            df_history = pd.concat([df_history, pd.DataFrame(
                [{**recipe, "scale": "micro", "Y": y_val, "cost_eur": SCALE_COSTS["micro"]}])],
                ignore_index=True)
            log_row("micro", recipe, y_val, SCALE_COSTS["micro"], "wide screening")
            iteration += 1

    # ---- Stage B: adaptive loop ----
    min_step = SCALE_COSTS["pilot"] if scale_mode == "pilot_only" else min(SCALE_COSTS.values())
    while budget_spent + min_step <= budget:
        # -- choose scale --
        if scale_mode == "pilot_only":
            chosen_scale, reason = "pilot", "Pilot-only adaptive BO (single-fidelity baseline)"
        else:
            remaining = budget - budget_spent
            if remaining >= (5 * SCALE_COSTS["pilot"]) + SCALE_COSTS["bench"]:
                chosen_scale, reason = "bench", "Scheduled Bench Region Finding"
            elif remaining >= SCALE_COSTS["pilot"]:
                chosen_scale, reason = "pilot", "Scheduled Pilot Peak Climbing"
            else:
                break

        if budget_spent + SCALE_COSTS[chosen_scale] > budget:
            break

        # -- choose recipe via the surrogate under test --
        if surrogate == "per_scale_gp":
            models = build_per_scale_gps(df_history)
            model = models.get(chosen_scale)
            if model is None:
                # not enough data yet at this scale -> random LHS point
                pt = qmc.scale(qmc.LatinHypercube(d=5, seed=seed * 1000 + iteration).random(1),
                                [b[0] for b in BOUNDS_LIST], [b[1] for b in BOUNDS_LIST])[0]
                recipe = {name: float(v) for name, v in zip(VAR_NAMES, pt)}
            else:
                sub = df_history[df_history["scale"] == chosen_scale]
                best_y_here = float(sub["Y"].max())
                X_hist = np.array([normalize([r[v] for v in VAR_NAMES]) for _, r in sub.iterrows()])
                next_norm = optimize_acquisition_generic(model, acquisition, best_y_here,
                                                           X_history=X_hist, kappa=kappa)
                recipe = recipe_dict_from_norm(next_norm)
        else:
            calib = fit_calibration(df_history)
            raw_noise = estimate_noise_variance(df_history)
            X, y_eq, var_eq, _ = to_pilot_equivalent(df_history, calib, raw_noise)
            if len(X) < 4:
                log("  Not enough data yet to fit surrogate, stopping.")
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = build_surrogate(surrogate, X, y_eq, var_eq)
            best_y_eq = float(np.max(y_eq))
            next_norm = optimize_acquisition_generic(model, acquisition, best_y_eq,
                                                       X_history=X, kappa=kappa)
            recipe = recipe_dict_from_norm(next_norm)

        cost = SCALE_COSTS[chosen_scale]
        log(f"[{acquisition}/{surrogate}/{scale_mode}] seed {seed} iter {iteration} "
            f"spent {budget_spent:.0f}/{budget:.0f} -> {chosen_scale.upper()} ({reason})")
        res = client.run(scale=chosen_scale, **recipe)
        y_val = res.get("Y", res.get("y"))
        df_history = pd.concat([df_history, pd.DataFrame(
            [{**recipe, "scale": chosen_scale, "Y": y_val, "cost_eur": cost}])], ignore_index=True)

        if chosen_scale == "pilot":
            pilot_evals += 1
            if y_val > best_pilot_y:
                best_pilot_y = y_val
                log(f"  *** NEW BEST PILOT Y: {best_pilot_y:.4f} ***")

        log_row(chosen_scale, recipe, y_val, cost, reason)
        iteration += 1

    log(f"===== [{acquisition}/{surrogate}/{scale_mode}] SEED {seed} DONE: "
        f"best pilot Y={best_pilot_y:.4f}, spent={budget_spent:.0f}, pilot_evals={pilot_evals} =====")
    return pd.DataFrame(trajectory), best_pilot_y, budget_spent, pilot_evals


# ==========================================
# GENERIC GROUPED COMPARISON PLOT (used by A1/A2/A3)
# ==========================================
def plot_grouped_comparison(summary: pd.DataFrame, group_col: str, filename: str, title: str):
    groups = list(pd.unique(summary[group_col]))
    means_y = [summary[summary[group_col] == g]["best_pilot_Y"].mean() for g in groups]
    std_y = [summary[summary[group_col] == g]["best_pilot_Y"].std() for g in groups]
    means_c = [summary[summary[group_col] == g]["total_cost"].mean() for g in groups]
    std_c = [summary[summary[group_col] == g]["total_cost"].std() for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    axes[0].bar(groups, means_y, yerr=std_y, capsize=6, color="#1f77b4")
    axes[0].set_title("Mean Best Pilot Yield (g/L)", fontweight="bold")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(groups, means_c, yerr=std_c, capsize=6, color="#ff7f0e")
    axes[1].set_title("Mean Total Cost (EUR)", fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f" -> Saved '{filename}'")


# ==========================================
# A1 DRIVER: Acquisition function comparison
# ==========================================
def task_a1_acquisition_comparison(n_seeds=DEFAULT_SEEDS, budget=DEFAULT_BUDGET):
    print("\n===== TASK A1: Acquisition function comparison (EI vs UCB) =====")
    client = BioreactorClient(BASE_URL)
    client.login(USER, PASSWORD)
    base_history = load_base_history()

    rows, all_traj = [], []
    for acq in ["EI", "UCB"]:
        for seed in range(n_seeds):
            traj, best_y, cost, pevals = run_generalized_campaign(
                client, seed, budget, base_history,
                acquisition=acq, surrogate="pooled_gp", scale_mode="multi_fidelity")
            all_traj.append(traj)
            rows.append({"acquisition": acq, "seed": seed, "best_pilot_Y": best_y,
                          "total_cost": cost, "pilot_evals": pevals})

    summary = pd.DataFrame(rows)
    summary.to_csv("a1_acquisition_comparison_summary.csv", index=False)
    pd.concat(all_traj, ignore_index=True).to_csv("a1_acquisition_comparison_trajectories.csv", index=False)
    print(summary.groupby("acquisition")[["best_pilot_Y", "total_cost"]].agg(["mean", "std"]))
    plot_grouped_comparison(summary, "acquisition", "a1_acquisition_comparison.png",
                             "Task A1: Expected Improvement vs Upper Confidence Bound")
    return summary


# ==========================================
# A2 DRIVER: Surrogate model class comparison
# ==========================================
def task_a2_surrogate_comparison(n_seeds=DEFAULT_SEEDS, budget=DEFAULT_BUDGET):
    print("\n===== TASK A2: Surrogate model class comparison =====")
    client = BioreactorClient(BASE_URL)
    client.login(USER, PASSWORD)
    base_history = load_base_history()

    rows, all_traj = [], []
    for surrogate in ["pooled_gp", "per_scale_gp", "rf"]:
        for seed in range(n_seeds):
            traj, best_y, cost, pevals = run_generalized_campaign(
                client, seed, budget, base_history,
                acquisition="EI", surrogate=surrogate, scale_mode="multi_fidelity")
            all_traj.append(traj)
            rows.append({"surrogate": surrogate, "seed": seed, "best_pilot_Y": best_y,
                          "total_cost": cost, "pilot_evals": pevals})

    summary = pd.DataFrame(rows)
    summary.to_csv("a2_surrogate_comparison_summary.csv", index=False)
    pd.concat(all_traj, ignore_index=True).to_csv("a2_surrogate_comparison_trajectories.csv", index=False)
    print(summary.groupby("surrogate")[["best_pilot_Y", "total_cost"]].agg(["mean", "std"]))
    plot_grouped_comparison(summary, "surrogate", "a2_surrogate_comparison.png",
                             "Task A2: Pooled Multi-Scale GP vs Independent Per-Scale GPs vs Random Forest")
    return summary


# ==========================================
# A3 DRIVER: Multi-fidelity vs pilot-only (equal budget)
# ==========================================
def task_a3_multifidelity_vs_pilot_only(n_seeds=DEFAULT_SEEDS, budget=PILOT_ONLY_BUDGET):
    print("\n===== TASK A3: Multi-fidelity vs pilot-only BO, equal total budget =====")
    client = BioreactorClient(BASE_URL)
    client.login(USER, PASSWORD)
    base_history = load_base_history()

    rows, all_traj = [], []
    modes = [
        ("multi_fidelity", "pooled_gp"),
        ("pilot_only", "per_scale_gp"),  # per_scale_gp on pilot-only data == a plain raw pilot GP
    ]
    for scale_mode, surrogate in modes:
        for seed in range(n_seeds):
            traj, best_y, cost, pevals = run_generalized_campaign(
                client, seed, budget, base_history,
                acquisition="EI", surrogate=surrogate, scale_mode=scale_mode)
            all_traj.append(traj)
            rows.append({"scale_mode": scale_mode, "seed": seed, "best_pilot_Y": best_y,
                          "total_cost": cost, "pilot_evals": pevals})

    summary = pd.DataFrame(rows)
    summary.to_csv("a3_multifidelity_vs_pilot_only_summary.csv", index=False)
    pd.concat(all_traj, ignore_index=True).to_csv("a3_multifidelity_vs_pilot_only_trajectories.csv", index=False)
    print(summary.groupby("scale_mode")[["best_pilot_Y", "total_cost", "pilot_evals"]].agg(["mean", "std"]))
    plot_grouped_comparison(summary, "scale_mode", "a3_multifidelity_vs_pilot_only.png",
                             "Task A3: Does Cheap-Scale Information Actually Help? (equal budget)")
    return summary


# ==========================================
# A4: Scale-selection pattern across ALL seeds (NO API CALLS)
# ==========================================
def task_a4_scale_selection_pattern(trajectories_csv="bo_all_trajectories.csv"):
    print("\n===== TASK A4: Full multi-seed scale-selection visualization (no API calls) =====")
    try:
        full_traj = pd.read_csv(trajectories_csv)
    except FileNotFoundError:
        print(f" Could not find '{trajectories_csv}'. Run phase3_bo.py first.")
        return None

    scale_to_int = {"micro": 0, "bench": 1, "pilot": 2}
    seeds = sorted(full_traj["seed"].unique())
    max_iter = int(full_traj["iteration"].max()) + 1
    grid = np.full((len(seeds), max_iter), np.nan)
    for si, seed in enumerate(seeds):
        g = full_traj[full_traj["seed"] == seed].sort_values("iteration")
        for _, row in g.iterrows():
            grid[si, int(row["iteration"])] = scale_to_int[row["scale"]]

    cmap = ListedColormap(["#2ca02c", "#ff7f0e", "#d62728"])
    fig, ax = plt.subplots(figsize=(13, 5), dpi=200)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f"seed {s}" for s in seeds])
    ax.set_xlabel("Iteration", fontweight="bold")
    ax.set_title("Task A4: Scale Selected per Iteration, All Seeds", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["MICRO", "BENCH", "PILOT"])
    plt.tight_layout()
    plt.savefig("a4_scale_selection_all_seeds.png", bbox_inches="tight")
    plt.close()

    # Transition matrix: what scale follows what, normalized to a probability per row
    trans_counts = {}
    for seed in seeds:
        g = full_traj[full_traj["seed"] == seed].sort_values("iteration")["scale"].tolist()
        for a, b in zip(g[:-1], g[1:]):
            trans_counts[(a, b)] = trans_counts.get((a, b), 0) + 1
    scales = ["micro", "bench", "pilot"]
    mat = np.zeros((3, 3))
    for i, a in enumerate(scales):
        for j, b in enumerate(scales):
            mat[i, j] = trans_counts.get((a, b), 0)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=200)
    im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([s.upper() for s in scales])
    ax.set_yticks(range(3)); ax.set_yticklabels([s.upper() for s in scales])
    ax.set_xlabel("Next scale", fontweight="bold")
    ax.set_ylabel("Current scale", fontweight="bold")
    ax.set_title("Task A4: Scale Transition Probabilities\n(across all seeds)",
                  fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat_norm[i, j] > 0.5 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("a4_scale_transition_matrix.png", bbox_inches="tight")
    plt.close()
    print(" -> Saved 'a4_scale_selection_all_seeds.png' and 'a4_scale_transition_matrix.png'")
    print(" Note: since the scale choice follows a fixed budget-based schedule ('bench until\n"
          " exactly 5 pilot runs remain, then pilot'), transitions are essentially deterministic\n"
          " and driven by REMAINING BUDGET, not by GP uncertainty. This is worth discussing in the\n"
          " report as a limitation/design choice versus a fully learned scale-selection policy.")
    return mat_norm


# ==========================================
# A5: Global sensitivity analysis (Sobol indices), NO API CALLS
# ==========================================
def sobol_sensitivity(model, n=2000, seed=0, d=5):
    """Monte-Carlo (Saltelli/Jansen) estimator of first-order (S1) and
    total-order (ST) Sobol indices, evaluated on the fitted surrogate
    instead of the real simulator (free, and the whole point of having a
    surrogate)."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, d))
    B = rng.random((n, d))

    def f(X):
        return np.asarray(model.predict(X)).ravel()

    yA, yB = f(A), f(B)
    var_y = np.var(np.concatenate([yA, yB]))
    S1 = np.zeros(d)
    ST = np.zeros(d)
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        yAB = f(AB)
        S1[i] = np.mean(yB * (yAB - yA)) / var_y
        ST[i] = 0.5 * np.mean((yA - yAB) ** 2) / var_y
    return S1, ST


def task_a5_global_sensitivity():
    print("\n===== TASK A5: Global sensitivity analysis (Sobol indices, no API calls) =====")
    base_history = load_base_history()
    try:
        bo_traj = pd.read_csv("bo_all_trajectories.csv")
        extra = bo_traj[["scale", "T", "pH", "F1", "F2", "F3", "Y", "cost_eur"]]
        combined = pd.concat([base_history, extra], ignore_index=True)
    except FileNotFoundError:
        print(" (No 'bo_all_trajectories.csv' found - using exploratory data only.)")
        combined = base_history

    calib = fit_calibration(combined)
    raw_noise = estimate_noise_variance(combined)
    X, y_eq, var_eq, _ = to_pilot_equivalent(combined, calib, raw_noise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr = build_gp(X, y_eq, var_eq)

    S1, ST = sobol_sensitivity(gpr)
    df = pd.DataFrame({"variable": VAR_NAMES, "S1_first_order": S1, "ST_total_order": ST})
    df.to_csv("a5_sobol_sensitivity.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["S1_first_order"], width, label="First-order S1", color="#1f77b4")
    ax.bar(x + width / 2, df["ST_total_order"], width, label="Total-order ST", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(df["variable"])
    ax.set_ylabel("Sobol index", fontweight="bold")
    ax.set_title("Task A5: Global Sensitivity Analysis of the Fitted GP Surrogate",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("a5_sobol_sensitivity.png", bbox_inches="tight")
    plt.close()
    print(" -> Saved 'a5_sobol_sensitivity.csv' and 'a5_sobol_sensitivity.png'")
    print(" Interpretation: ST >> S1 for a variable means it mostly matters through INTERACTIONS\n"
          " with other variables (not on its own) - worth calling out explicitly in the report.")
    return df


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Phase 4 - Additional Tasks runner")
    parser.add_argument("--task", choices=["a1", "a2", "a3", "a4", "a5", "all"], default="all")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                         help="seeds per comparison arm for a1/a2/a3 (default: %(default)s)")
    parser.add_argument("--budget", type=float, default=None,
                         help="EUR budget per comparison arm for a1/a2/a3 "
                              "(default: 8000 for a1/a2, 15000 for a3)")
    args = parser.parse_args()

    logger = Logger("phase4_output.txt")
    sys.stdout = logger
    try:
        if args.task in ("a1", "all"):
            task_a1_acquisition_comparison(args.seeds, args.budget or DEFAULT_BUDGET)
        if args.task in ("a2", "all"):
            task_a2_surrogate_comparison(args.seeds, args.budget or DEFAULT_BUDGET)
        if args.task in ("a3", "all"):
            task_a3_multifidelity_vs_pilot_only(args.seeds, args.budget or PILOT_ONLY_BUDGET)
        if args.task in ("a4", "all"):
            task_a4_scale_selection_pattern()
        if args.task in ("a5", "all"):
            task_a5_global_sensitivity()
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()