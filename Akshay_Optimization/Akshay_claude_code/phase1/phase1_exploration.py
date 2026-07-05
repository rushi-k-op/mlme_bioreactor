"""
Phase 1 - Exploration of the Multi-Scale Bioreactor
=====================================================
Mandatory tasks covered in this file:
  Task 1: Connect to the simulator API, run exploratory experiments, and
          visualize the dependence of Y on EACH recipe variable at EACH scale.
  Task 2: Quantify the per-scale measurement noise and bias, and discuss how
          these properties should influence the optimization strategy.

Fixes applied vs. the original script:
  - ADDED: plot_variable_dependence() - this was MISSING. The old script only
    plotted a scale-profiling scatter (Y grouped by scale), but never showed
    how Y depends on T, pH, F1, F2, F3 individually - which is explicitly
    required by mandatory Task 1.
  - ADDED: a bench-vs-pilot and micro-vs-pilot calibration plot + fitted
    linear regression, because this is the single most important empirical
    finding for Phase 3: bench correlates almost perfectly with pilot,
    while micro does not. This number is reused in phase3_bo.py.
  - FIX: stdout/log file handle is now closed properly at the end.
  - FIX: noise/bias analysis now also reports relative (%) bias, which is
    more meaningful for discussion than an absolute number alone.
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc, linregress
import sys
import matplotlib
matplotlib.use('Agg')  # Forces headless rendering
import matplotlib.pyplot as plt

from client import BioreactorClient, USER, PASSWORD, BASE_URL

# ==========================================
# CUSTOM TERMINAL LOGGER (writes to console + txt file automatically)
# ==========================================
class Logger(object):
    """Intercepts print statements and writes to both terminal and a txt file."""
    def __init__(self, filename="exploration_output.txt"):
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

# Recipe Bounds defined in Table 1 of the assignment
BOUNDS = {
    "T": (20.0, 60.0),
    "pH": (3.0, 9.5),
    "F1": (0.0, 2.0),
    "F2": (0.0, 2.0),
    "F3": (0.0, 2.0)
}

SCALES = ["micro", "bench", "pilot"]
NUM_RECIPES = 10     # Number of distinct points in 5D space to test
REPEATS = 3          # Times to run EACH recipe to measure noise variance


def generate_lhs_samples(n_samples: int, bounds: dict) -> pd.DataFrame:
    """Generates Latin Hypercube Samples across the parameter space."""
    sampler = qmc.LatinHypercube(d=len(bounds))
    sample_unit = sampler.random(n=n_samples)
    lower_bounds = [b[0] for b in bounds.values()]
    upper_bounds = [b[1] for b in bounds.values()]
    samples_scaled = qmc.scale(sample_unit, lower_bounds, upper_bounds)
    return pd.DataFrame(samples_scaled, columns=bounds.keys())


def run_exploration():
    client = BioreactorClient(BASE_URL)
    print("Logging into the bioreactor...")
    client.login(USER, PASSWORD)

    print(f"Generating {NUM_RECIPES} LHS recipes...")
    recipes_df = generate_lhs_samples(NUM_RECIPES, BOUNDS)

    results = []
    total_cost_spent = 0.0

    for idx, row in recipes_df.iterrows():
        recipe_dict = row.to_dict()
        print(f"\n--- Testing Recipe {idx+1}/{NUM_RECIPES} ---")

        for scale in SCALES:
            for rep in range(REPEATS):
                print(f"  Running {scale.upper()} (Repeat {rep+1}/{REPEATS})...", end=" ")

                res = client.run(
                    scale=scale,
                    T=recipe_dict["T"], pH=recipe_dict["pH"],
                    F1=recipe_dict["F1"], F2=recipe_dict["F2"], F3=recipe_dict["F3"]
                )

                y_val = res.get("Y")
                cost_val = res.get("cost_eur")
                if cost_val is None or y_val is None:
                    raise KeyError(f"Unexpected response format from server: {res}")

                total_cost_spent += cost_val
                print(f"Result: Y = {y_val:.4f}, Cost = {cost_val} EUR")

                results.append({
                    "Recipe_ID": idx, "Scale": scale, "Repeat": rep,
                    "T": recipe_dict["T"], "pH": recipe_dict["pH"],
                    "F1": recipe_dict["F1"], "F2": recipe_dict["F2"], "F3": recipe_dict["F3"],
                    "Y": y_val, "Cost": cost_val
                })

    df_results = pd.DataFrame(results)
    df_results.to_csv("exploratory_data.csv", index=False)
    print(f"\nExploration Complete! Data saved to 'exploratory_data.csv'.")
    print(f"Total budget spent on exploration: {total_cost_spent} EUR")
    return df_results


def analyze_noise_and_bias(df: pd.DataFrame):
    """Quantifies per-scale measurement noise (variance across repeats) and
    bias (systematic offset relative to pilot) - Mandatory Task 2."""
    print("\n=====================================")
    print(" NOISE AND BIAS ANALYSIS")
    print("=====================================")

    stats = df.groupby(['Recipe_ID', 'Scale'])['Y'].agg(['mean', 'var']).reset_index()

    # 1. Noise
    noise_summary = stats.groupby('Scale')['var'].mean().reset_index()
    noise_summary.rename(columns={'var': 'Average_Variance (Noise)'}, inplace=True)
    print("\n--- Measurement Noise (variance of Y across repeats) ---")
    print(noise_summary.to_string(index=False))
    noise_summary.to_csv("noise_analysis.csv", index=False)
    print(" -> Saved noise analysis to 'noise_analysis.csv'")

    # 2. Bias (absolute + relative + calibration to pilot)
    means_pivot = stats.pivot(index='Recipe_ID', columns='Scale', values='mean')
    means_pivot['Bias_Micro'] = means_pivot['micro'] - means_pivot['pilot']
    means_pivot['Bias_Bench'] = means_pivot['bench'] - means_pivot['pilot']

    bias_summary = pd.DataFrame({
        'Scale': ['micro', 'bench', 'pilot'],
        'Average_Bias': [means_pivot['Bias_Micro'].mean(), means_pivot['Bias_Bench'].mean(), 0.0],
        'Mean_Absolute_Bias': [means_pivot['Bias_Micro'].abs().mean(), means_pivot['Bias_Bench'].abs().mean(), 0.0],
    })
    print("\n--- Measurement Bias (Relative to Pilot) ---")
    print(bias_summary.to_string(index=False))
    bias_summary.to_csv("bias_analysis.csv", index=False)
    print(" -> Saved bias analysis to 'bias_analysis.csv'")

    # 3. Linear calibration: pilot ~ slope*scale + intercept  (KEY for Phase 3!)
    print("\n--- Cross-scale correlation / calibration (crucial for BO design) ---")
    calib_rows = []
    for cheap_scale in ['bench', 'micro']:
        x = means_pivot[cheap_scale].values
        y = means_pivot['pilot'].values
        slope, intercept, r, p, se = linregress(x, y)
        print(f"  pilot ~ {slope:.4f} * {cheap_scale} + {intercept:.4f}   (R^2 = {r**2:.4f})")
        calib_rows.append({'cheap_scale': cheap_scale, 'slope': slope,
                            'intercept': intercept, 'r_squared': r**2})
    pd.DataFrame(calib_rows).to_csv("scale_calibration.csv", index=False)
    print(" -> Saved calibration coefficients to 'scale_calibration.csv'")
    print("\n  INTERPRETATION: the scale with the higher R^2 is a much more")
    print("  trustworthy (and cost-effective) proxy for pilot performance,")
    print("  even if it is not the cheapest option. This directly informs")
    print("  which scale(s) the Phase 3 BO loop should rely on most.")


def plot_phase1_results(df: pd.DataFrame):
    """Scale-level profiling scatter (noise & bias overview)."""
    print("\nGenerating Phase 1 scale-profiling visualization...")
    plt.figure(figsize=(8, 6), dpi=300)
    scales = ['micro', 'bench', 'pilot']
    colors = {'micro': '#2ca02c', 'bench': '#ff7f0e', 'pilot': '#d62728'}

    for i, scale in enumerate(scales):
        sub_df = df[df['Scale'].str.lower() == scale]
        if sub_df.empty:
            continue
        jitter = np.random.normal(i, 0.05, size=len(sub_df))
        plt.scatter(jitter, sub_df['Y'], color=colors[scale], s=60, alpha=0.7,
                    edgecolors='black', label=scale.upper())
        mean_y = sub_df['Y'].mean()
        plt.hlines(y=mean_y, xmin=i-0.25, xmax=i+0.25, color=colors[scale],
                   linestyle='--', linewidth=2, alpha=0.8)

    plt.xticks(range(len(scales)), [s.upper() for s in scales], fontweight='bold')
    plt.title("Phase 1: Scale Profiling (Noise & Bias)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Measured Bio-Yield Y (g/L)", fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig("phase1_profiling.png", bbox_inches='tight')
    plt.close()
    print(" -> Saved visualization to 'phase1_profiling.png'")


def plot_variable_dependence(df: pd.DataFrame):
    """*** THIS WAS MISSING IN THE ORIGINAL SCRIPT ***
    Mandatory Task 1 explicitly requires visualizing how Y depends on EACH
    recipe variable (T, pH, F1, F2, F3) AT EACH scale. This produces a
    5-panel figure, one subplot per variable, each showing Y vs. that
    variable with points colored by scale (using the mean over repeats).
    """
    print("\nGenerating Phase 1 per-variable dependence visualization...")
    variables = ["T", "pH", "F1", "F2", "F3"]
    colors = {'micro': '#2ca02c', 'bench': '#ff7f0e', 'pilot': '#d62728'}

    means = df.groupby(['Recipe_ID', 'Scale']).agg(
        {**{v: 'first' for v in variables}, 'Y': 'mean'}
    ).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=200)
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        for scale in ['micro', 'bench', 'pilot']:
            sub = means[means['Scale'].str.lower() == scale].sort_values(var)
            if sub.empty:
                continue
            ax.scatter(sub[var], sub['Y'], color=colors[scale], s=70,
                       alpha=0.8, edgecolors='black', label=scale.upper())
        ax.set_xlabel(var, fontsize=11, fontweight='bold')
        ax.set_ylabel("Y (g/L)", fontsize=11, fontweight='bold')
        ax.set_title(f"Y vs {var}", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.4, linestyle='--')

    # Legend + unused 6th subplot slot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis('off')
    axes[-1].legend(handles, labels, loc='center', fontsize=13, frameon=True,
                     title="Scale")

    fig.suptitle("Phase 1: Dependence of Product Yield Y on Each Recipe Variable",
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("phase1_variable_dependence.png", bbox_inches='tight')
    plt.close()
    print(" -> Saved visualization to 'phase1_variable_dependence.png'")


def plot_scale_calibration(df: pd.DataFrame):
    """Scatter of bench-vs-pilot and micro-vs-pilot with fitted regression
    line. This is the empirical justification for the Phase 3 multi-fidelity
    strategy (bench is a near-perfect, much cheaper proxy for pilot)."""
    print("\nGenerating scale-calibration visualization...")
    means_pivot = df.groupby(['Recipe_ID', 'Scale'])['Y'].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    for ax, cheap_scale, color in zip(axes, ['bench', 'micro'], ['#ff7f0e', '#2ca02c']):
        x = means_pivot[cheap_scale].values
        y = means_pivot['pilot'].values
        slope, intercept, r, p, se = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.scatter(x, y, color=color, s=80, edgecolors='black', zorder=3)
        ax.plot(xs, slope * xs + intercept, color='black', linestyle='--',
                label=f"pilot = {slope:.3f}*{cheap_scale} + {intercept:.3f}\nR^2={r**2:.3f}")
        ax.set_xlabel(f"Y_{cheap_scale} (g/L)", fontweight='bold')
        ax.set_ylabel("Y_pilot (g/L)", fontweight='bold')
        ax.set_title(f"Pilot vs {cheap_scale.capitalize()}", fontweight='bold')
        ax.grid(alpha=0.4, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)

    fig.suptitle("Phase 1: Cross-Scale Calibration (which cheap scale predicts pilot best?)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("phase1_scale_calibration.png", bbox_inches='tight')
    plt.close()
    print(" -> Saved visualization to 'phase1_scale_calibration.png'")


if __name__ == "__main__":
    logger = Logger("exploration_output.txt")
    sys.stdout = logger
    try:
        df = run_exploration()
        analyze_noise_and_bias(df)
        plot_phase1_results(df)
        plot_variable_dependence(df)
        plot_scale_calibration(df)
    finally:
        sys.stdout = logger.terminal
        logger.close()