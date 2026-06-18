"""
Task A – Simulator Integration & Exploratory Experiments
=========================================================
Goals:
  1. Connect to the API and verify credentials.
  2. Inspect the raw API response (so we know the exact field names).
  3. Run 1D sweeps: vary each recipe variable while fixing others at midpoints,
     at each reactor scale.
  4. Visualise Y (product yield) vs. each variable, coloured by scale.
  5. Summarise per-scale costs and yield distributions.

Output:
  data/task_a_1d_sweeps.csv    – all raw run records
  plots/task_a_sweeps.png      – 5-panel yield sensitivity figure
  plots/task_a_cost.png        – mean cost per scale bar chart
  plots/task_a_yield_dist.png  – yield box-plot per scale

Usage:
  conda activate PAS
  pip install requests numpy pandas matplotlib
  python task_a_explore.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── making sure basic_client.py is importable 
sys.path.insert(0, str(Path(__file__).parent))
from basic_client import BioreactorClient

USER     = "group05"
PASSWORD = "higsq-wfcnq-cv9ym"
BASE_URL = "https://mlme26biosim.org"

# ─── Reactor scales (cheapest → most expensive / least noisy)
SCALES = ["micro", "bench", "pilot"]
SCALE_COLORS  = {"micro": "#e74c3c", "bench": "#f39c12", "pilot": "#2980b9"}
SCALE_MARKERS = {"micro": "o",       "bench": "s",        "pilot": "^"}

# ─── Recipe variable bounds
PARAM_BOUNDS = {
    "T":  (20.0, 60.0),   # reactor temperature °C
    "pH": (3.0,  9.5),    # pH
    "F1": (0.0,  2.0),    # glucose feed rate  g/L/h
    "F2": (0.0,  2.0),    # glutamine feed rate g/L/h
    "F3": (0.0,  2.0),    # rich-media feed rate g/L/h
}

PARAM_LABELS = {
    "T":  "Temperature T (°C)",
    "pH": "pH",
    "F1": "Glucose feed F₁ (g/L/h)",
    "F2": "Glutamine feed F₂ (g/L/h)",
    "F3": "Rich-media feed F₃ (g/L/h)",
}

# ─── Default / midpoint recipe used when a variable is NOT being swept
# (assumption to keep things neutral)
DEFAULTS = {
    "T":  40.0,
    "pH": 6.5,
    "F1": 1.0,
    "F2": 1.0,
    "F3": 1.0,
}

# ─── Number of sweep points per variable per scale 
# 6 pts × 5 vars × 3 scales = 90 runs total.
# Run the connection test first to see the cost of one micro run before
# committing to all 90!
N_SWEEP_PTS = 6

DATA_DIR  = Path("data")
PLOTS_DIR = Path("plots")


#  HELPERS
def recipe_with(**overrides) -> dict:
    """Return a copy of DEFAULTS with the given keys overridden."""
    r = dict(DEFAULTS)
    r.update(overrides)
    return r


def parse_result(raw: dict) -> tuple[float, float]:
    """
    Extract (Y, cost) from the API response.

    The actual field names depend on the server. The function tries several
    plausible names and falls back to NaN so the rest of the script still
    runs. Fix the key names below once you've run test_connection() and
    inspected the raw response printed to the console.
    """
    def try_keys(d, *keys):
        for k in keys:
            if k in d:
                try:
                    return float(d[k])
                except (TypeError, ValueError):
                    pass
        return float("nan")

    Y    = try_keys(raw, "Y", "y", "yield", "product_yield", "product")
    cost = try_keys(raw, "cost", "Cost", "price", "EUR", "cost_eur")
    return Y, cost
"""
# This can be used insted of the above parse_result, 
# if variable name is known in the server

def parse_result(raw):
    return raw["Y"], raw["cost_eur"]
"""

#  ---------------------
#  STEP 1 – CONNECTION TEST

def test_connection(client: BioreactorClient) -> dict:
    """
    Fire one micro-scale run at the default recipe.

    Purpose:
      • Confirm credentials and network access work.
      • Print the raw JSON so we can identify exact key names for Y and cost.
      • Gives an early cost estimate for one micro run.
    """
    print("\n" + "═" * 60)
    print("  STEP 1 - Connection & API response inspection")
    print("═" * 60)

    raw = client.run("micro",
                     T=DEFAULTS["T"], pH=DEFAULTS["pH"],
                     F1=DEFAULTS["F1"], F2=DEFAULTS["F2"], F3=DEFAULTS["F3"])

    print("\n  ┌── Raw API response (read the key names carefully!) ───")
    print("  │  " + json.dumps(raw, indent=2).replace("\n", "\n  │  "))
    print("  └─────────────────────────────────────────────────────")

    Y, cost = parse_result(raw)
    print(f"\n  Parsed → Y = {Y:.4f} g/L   |   cost = {cost:.2f} EUR")
    print(f"\n   If Y or cost shows 'nan', update parse_result() with")
    print(f"     the correct key names from the raw response above.\n")

    return raw

#  ---------------------
#  STEP 2 – 1D SWEEPS

def run_1d_sweeps(client: BioreactorClient, n_pts: int = N_SWEEP_PTS) -> pd.DataFrame:
    """
    For each recipe variable, sweep it linearly across its full range
    while holding all other variables at DEFAULTS.  Run at all three scales.

    Returns a tidy DataFrame with one row per experiment.
    """
    total_runs = len(PARAM_BOUNDS) * n_pts * len(SCALES)
    print("\n" + "═" * 60)
    print(f"  STEP 2 - 1D Sweeps")
    print(f"  {len(PARAM_BOUNDS)} variables × {n_pts} points × {len(SCALES)} scales"
          f" = {total_runs} runs")
    print("═" * 60)

    records: list[dict] = []

    for param, (lo, hi) in PARAM_BOUNDS.items():
        sweep_values = np.linspace(lo, hi, n_pts)
        print(f"\n  ── Sweeping {param:2s}  [{lo} → {hi}] ──────────────────")

        for scale in SCALES:
            print(f"     Scale: {scale}")
            for v in sweep_values:
                r = recipe_with(**{param: float(v)})

                time.sleep(2)

                raw = client.run(
                    scale,
                    T=r["T"], pH=r["pH"],
                    F1=r["F1"], F2=r["F2"], F3=r["F3"],
                )
                Y, cost = parse_result(raw)

                print(f"       {param}={v:6.3f}  →  Y={Y:7.3f} g/L   "
                      f"cost={cost:7.2f} EUR")

                records.append({
                    # sweep metadata
                    "sweep_var":   param,
                    "sweep_value": float(v),
                    "scale":       scale,
                    # outputs
                    "Y":           Y,
                    "cost":        cost,
                    # full recipe (useful for later regression / BO)
                    "T":   r["T"],
                    "pH":  r["pH"],
                    "F1":  r["F1"],
                    "F2":  r["F2"],
                    "F3":  r["F3"],
                    # raw response for reference
                    "raw_json": json.dumps(raw),
                })
            
    df = pd.DataFrame(records)

    # persist immediately so data isn't lost if plotting crashes
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "task_a_1d_sweeps.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Saved {len(df)} records → {csv_path}")
    return df


#  ---------------------
#  STEP 3 – COST SUMMARY

def print_cost_summary(df: pd.DataFrame) -> None:
    """
    Print per-scale cost statistics.
    Important for Task B (Noise & Bias Analysis) planning:
    knowing the cost ratio drives how often we can afford each scale.
    """
    print("\n" + "═" * 60)
    print("  STEP 3 – Cost summary by reactor scale")
    print("═" * 60)

    stats = (
        df.groupby("scale")["cost"]
          .agg(n_runs="count", mean="mean", std="std", min="min", max="max")
          .reindex(SCALES)
    )
    print("\n" + stats.to_string(float_format=lambda x: f"{x:.2f}"))

    total = df["cost"].sum()
    print(f"\n  Total spend for Task A exploratory runs: {total:.2f} EUR")

    # cost ratio relative to micro (useful for BO scale selection later)
    micro_mean = stats.loc["micro", "mean"]
    if not np.isnan(micro_mean) and micro_mean > 0:
        print("\n  Cost ratio relative to micro scale:")
        for s in SCALES:
            ratio = stats.loc[s, "mean"] / micro_mean
            print(f"    {s:8s}: {ratio:.2f}×")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 – VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_1d_sweeps(df: pd.DataFrame) -> None:
    """
    5-panel figure: one panel per recipe variable.
    Each panel shows Y vs. the swept variable for all three scales.

    This is the main Task A deliverable plot – it shows:
      • Which variables have the strongest effect on yield.
      • How scale-dependent bias shifts the curves up/down.
      • Where noise is larger (wider scatter) at cheaper scales.
    """
    PLOTS_DIR.mkdir(exist_ok=True)
    params = list(PARAM_BOUNDS.keys())

    fig, axes = plt.subplots(1, len(params), figsize=(20, 4.5), sharey=False)
    fig.suptitle(
        "1D Sensitivity: Product Yield Y vs. Recipe Variable at Each Reactor Scale\n"
        "(all other variables held at midpoint defaults)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for ax, param in zip(axes, params):
        sub = df[df["sweep_var"] == param]
        for scale in SCALES:
            sd = sub[sub["scale"] == scale].sort_values("sweep_value")
            ax.plot(
                sd["sweep_value"], sd["Y"],
                marker=SCALE_MARKERS[scale],
                color=SCALE_COLORS[scale],
                label=scale.capitalize(),
                linewidth=1.8, markersize=6, alpha=0.9,
            )

        ax.set_xlabel(PARAM_LABELS[param], fontsize=9)
        ax.set_ylabel("Y (g/L)", fontsize=9)
        ax.set_title(f"Sweep: {param}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = PLOTS_DIR / "task_a_sweeps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved sweep figure → {path}")
    plt.show()


def plot_cost_per_scale(df: pd.DataFrame) -> None:
    """
    Bar chart of mean cost per run at each scale.
    Directly answers: 'How much cheaper is micro vs. pilot?'
    """
    PLOTS_DIR.mkdir(exist_ok=True)

    cost_stats = (
        df.groupby("scale")["cost"]
          .agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)))
          .reindex(SCALES)
    )

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(
        [s.capitalize() for s in SCALES],
        cost_stats["mean"],
        yerr=cost_stats["sem"],
        color=[SCALE_COLORS[s] for s in SCALES],
        width=0.5, edgecolor="k", linewidth=0.8,
        capsize=5, error_kw={"elinewidth": 1.4},
    )
    ax.bar_label(bars, fmt="%.0f €", padding=6, fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean cost per run (EUR)", fontsize=11)
    ax.set_title("Experiment Cost by Reactor Scale\n(error bars = standard error)", fontsize=11)
    ax.set_ylim(0, cost_stats["mean"].max() * 1.35)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f} €"))

    plt.tight_layout()
    path = PLOTS_DIR / "task_a_cost.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved cost figure  → {path}")
    plt.show()


def plot_yield_distribution(df: pd.DataFrame) -> None:
    """
    Box-plot of Y distribution across all 1D sweep runs at each scale.

    This gives a first visual signal of:
      • Pilot vs. micro bias (median shift).
      • Measurement noise (box height / whisker length).
    """
    PLOTS_DIR.mkdir(exist_ok=True)

    data   = [df[df["scale"] == s]["Y"].dropna().values for s in SCALES]
    labels = [s.capitalize() for s in SCALES]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="x", markersize=5, alpha=0.5),
    )
    for patch, scale in zip(bp["boxes"], SCALES):
        patch.set_facecolor(SCALE_COLORS[scale])
        patch.set_alpha(0.75)

    ax.set_ylabel("Y (g/L)", fontsize=11)
    ax.set_title(
        "Yield Distribution across All 1D Sweep Runs\n"
        "(Box = IQR, line = median, whiskers = 1.5×IQR)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = PLOTS_DIR / "task_a_yield_dist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved distribution figure → {path}")
    plt.show()


def plot_heatmap_variable_vs_scale(df: pd.DataFrame) -> None:
    """
    Heatmap: rows = recipe variables, columns = scales,
    cell = range of Y across that sweep (max − min).

    Answers: 'Which variable matters most, and does it differ by scale?'
    """
    PLOTS_DIR.mkdir(exist_ok=True)

    records = []
    for param in PARAM_BOUNDS:
        for scale in SCALES:
            sub = df[(df["sweep_var"] == param) & (df["scale"] == scale)]["Y"]
            if len(sub) > 1:
                records.append({
                    "Variable": param,
                    "Scale": scale.capitalize(),
                    "Y range": sub.max() - sub.min(),
                })
    hm_df = pd.DataFrame(records).pivot(index="Variable", columns="Scale", values="Y range")
    hm_df = hm_df[[s.capitalize() for s in SCALES]]  # enforce column order

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(hm_df.values, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Y range (g/L)  [max – min across sweep]")

    ax.set_xticks(range(len(hm_df.columns)))
    ax.set_xticklabels(hm_df.columns, fontsize=11)
    ax.set_yticks(range(len(hm_df.index)))
    ax.set_yticklabels(hm_df.index, fontsize=11)

    # annotate cells
    for i in range(len(hm_df.index)):
        for j in range(len(hm_df.columns)):
            val = hm_df.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val > hm_df.values.max() * 0.6 else "black")

    ax.set_title("Effect Size per Variable per Scale\n(Y range = max − min over sweep)",
                 fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / "task_a_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved heatmap figure   → {path}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if USER == "your-username-here":
        print("\n⚠  ERROR: Please set USER and PASSWORD in the CONFIG section "
              "at the top of this script before running.\n")
        sys.exit(1)

    # ── Authenticate ──────────────────────────────────────────────────────
    print(f"\nConnecting to {BASE_URL} …")
    client = BioreactorClient(BASE_URL)
    client.login(USER, PASSWORD)
    print("  ✓ Logged in successfully")

    # ── Step 1 – Inspect the API ──────────────────────────────────────────
    # test_connection(client)

    # input("\n  ► Press Enter to start 1D sweeps (check the cost above first!) … ")

    # ── Step 2 – Run sweeps ───────────────────────────────────────────────
    df = run_1d_sweeps(client)

    # ── Step 3 – Cost analysis ────────────────────────────────────────────
    print_cost_summary(df)

    # ── Step 4 – Plots ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 4 – Generating plots")
    print("═" * 60)
    plot_1d_sweeps(df)
    plot_cost_per_scale(df)
    plot_yield_distribution(df)
    plot_heatmap_variable_vs_scale(df)

    print("\n" + "═" * 60)
    print("  ✓  Task A complete!")
    print(f"     Data  → {DATA_DIR}/task_a_1d_sweeps.csv")
    print(f"     Plots → {PLOTS_DIR}/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()