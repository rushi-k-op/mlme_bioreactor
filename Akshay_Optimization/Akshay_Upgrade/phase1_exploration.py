import numpy as np
import pandas as pd
from scipy.stats import qmc
import sys
import matplotlib
matplotlib.use('Agg')  # Forces headless rendering
import matplotlib.pyplot as plt

# --- Centralized Configuration Import ---
from client import BioreactorClient, USER, PASSWORD, BASE_URL

# ==========================================
# CUSTOM TERMINAL LOGGER
# ==========================================
class Logger(object):
    """Intercepts print statements and writes to both terminal and a txt file."""
    def __init__(self, filename="exploration_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes to the file immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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
    
    # Scale from [0, 1] to actual physical bounds
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

    # Run nested loop: For each recipe, run on all scales, multiple times
    for idx, row in recipes_df.iterrows():
        recipe_dict = row.to_dict()
        print(f"\n--- Testing Recipe {idx+1}/{NUM_RECIPES} ---")
        
        for scale in SCALES:
            for rep in range(REPEATS):
                print(f"  Running {scale.upper()} (Repeat {rep+1}/{REPEATS})...", end=" ")
                
                res = client.run(
                    scale=scale,
                    T=recipe_dict["T"],
                    pH=recipe_dict["pH"],
                    F1=recipe_dict["F1"],
                    F2=recipe_dict["F2"],
                    F3=recipe_dict["F3"]
                )
                
                # Extract values using the exact keys from the server
                y_val = res.get("Y")
                cost_val = res.get("cost_eur")
                
                if cost_val is None or y_val is None:
                    raise KeyError(f"Unexpected response format from server: {res}")
                
                total_cost_spent += cost_val
                
                print(f"Result: Y = {y_val:.4f}, Cost = {cost_val} EUR")
                
                results.append({
                    "Recipe_ID": idx,
                    "Scale": scale,
                    "Repeat": rep,
                    "T": recipe_dict["T"],
                    "pH": recipe_dict["pH"],
                    "F1": recipe_dict["F1"],
                    "F2": recipe_dict["F2"],
                    "F3": recipe_dict["F3"],
                    "Y": y_val,
                    "Cost": cost_val
                })

    df_results = pd.DataFrame(results)
    df_results.to_csv("exploratory_data.csv", index=False)
    print(f"\nExploration Complete! Data saved to 'exploratory_data.csv'.")
    print(f"Total budget spent on exploration: {total_cost_spent} EUR")
    
    return df_results

def analyze_noise_and_bias(df: pd.DataFrame):
    """Calculates per-scale measurement noise and bias."""
    print("\n=====================================")
    print(" NOISE AND BIAS ANALYSIS")
    print("=====================================")
    
    # Calculate Mean and Variance for each recipe at each scale
    stats = df.groupby(['Recipe_ID', 'Scale'])['Y'].agg(['mean', 'var']).reset_index()
    
    # 1. Quantify Noise
    noise_summary = stats.groupby('Scale')['var'].mean().reset_index()
    noise_summary.rename(columns={'var': 'Average_Variance (Noise)'}, inplace=True)
    print("\n--- Measurement Noise ---")
    print(noise_summary.to_string(index=False))
    
    # Save noise to CSV
    noise_summary.to_csv("noise_analysis.csv", index=False)
    print(" -> Saved noise analysis to 'noise_analysis.csv'")
    
    # 2. Quantify Bias 
    means_pivot = stats.pivot(index='Recipe_ID', columns='Scale', values='mean')
    means_pivot['Bias_Micro'] = means_pivot['micro'] - means_pivot['pilot']
    means_pivot['Bias_Bench'] = means_pivot['bench'] - means_pivot['pilot']
    
    bias_summary = pd.DataFrame({
        'Scale': ['micro', 'bench', 'pilot'],
        'Average_Bias': [
            means_pivot['Bias_Micro'].mean(),
            means_pivot['Bias_Bench'].mean(),
            0.0 
        ],
        'Mean_Absolute_Bias': [
            means_pivot['Bias_Micro'].abs().mean(),
            means_pivot['Bias_Bench'].abs().mean(),
            0.0
        ]
    })
    print("\n--- Measurement Bias (Relative to Pilot) ---")
    print(bias_summary.to_string(index=False))
    
    # Save bias to CSV
    bias_summary.to_csv("bias_analysis.csv", index=False)
    print(" -> Saved bias analysis to 'bias_analysis.csv'")

def plot_phase1_results(df: pd.DataFrame):
    """Generates a scatter plot of the exploration data to visualize noise and bias."""
    print("\nGenerating Phase 1 visualization...")
    plt.figure(figsize=(8, 6), dpi=300)

    scales = ['micro', 'bench', 'pilot']
    colors = {'micro': '#2ca02c', 'bench': '#ff7f0e', 'pilot': '#d62728'}

    for i, scale in enumerate(scales):
        sub_df = df[df['Scale'].str.lower() == scale]
        if sub_df.empty: continue

        # Jitter the X axis slightly so data points don't stack invisibly
        jitter = np.random.normal(i, 0.05, size=len(sub_df))
        plt.scatter(jitter, sub_df['Y'], color=colors[scale],
                   s=60, alpha=0.7, edgecolors='black', label=scale.upper())

        # Draw mean line for the scale
        mean_y = sub_df['Y'].mean()
        plt.hlines(y=mean_y, xmin=i-0.25, xmax=i+0.25, color=colors[scale], linestyle='--', linewidth=2, alpha=0.8)

    plt.xticks(range(len(scales)), [s.upper() for s in scales], fontweight='bold')
    plt.title("Phase 1: Scale Profiling (Noise & Bias)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Measured Bio-Yield Y (g/L)", fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Prevent duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True)

    plt.tight_layout()
    output_img = "phase1_profiling.png"
    plt.savefig(output_img, bbox_inches='tight')
    plt.close()
    print(f" -> Saved visualization to '{output_img}'")

if __name__ == "__main__":
    # Redirect standard output to our custom Logger
    sys.stdout = Logger("exploration_output.txt")
    
    df = run_exploration()
    analyze_noise_and_bias(df)
    plot_phase1_results(df)
