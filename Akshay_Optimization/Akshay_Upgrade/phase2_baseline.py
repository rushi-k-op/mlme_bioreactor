import pandas as pd
import numpy as np
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
    def __init__(self, filename="baseline_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes to the file immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Recipe Bounds defined in Table 1
BOUNDS = {
    "T": (20.0, 60.0),
    "pH": (3.0, 9.5),
    "F1": (0.0, 2.0),
    "F2": (0.0, 2.0),
    "F3": (0.0, 2.0)
}

# The target budget from Felix's benchmark
TOTAL_BUDGET = 15000  
PILOT_COST = 2000  # Cost per pilot run (based on your earlier server logs)

def generate_lhs_samples(n_samples: int, bounds: dict) -> pd.DataFrame:
    """Generates Latin Hypercube Samples across the parameter space."""
    sampler = qmc.LatinHypercube(d=len(bounds))
    sample_unit = sampler.random(n=n_samples)
    
    lower_bounds = [b[0] for b in bounds.values()]
    upper_bounds = [b[1] for b in bounds.values()]
    
    samples_scaled = qmc.scale(sample_unit, lower_bounds, upper_bounds)
    return pd.DataFrame(samples_scaled, columns=bounds.keys())

def run_baseline() -> pd.DataFrame:
    client = BioreactorClient(BASE_URL)
    print("Logging into the bioreactor...")
    client.login(USER, PASSWORD)
    
    # Calculate exactly how many pilot runs we can afford
    num_runs = TOTAL_BUDGET // PILOT_COST
    print(f"Budget: {TOTAL_BUDGET} EUR. Afforded Pilot runs: {num_runs}")
    
    recipes_df = generate_lhs_samples(num_runs, BOUNDS)
    
    best_y = 0.0
    total_cost = 0.0
    results = []

    print("\nStarting Baseline LHS Campaign (Pilot Only)...")
    for idx, row in recipes_df.iterrows():
        recipe_dict = row.to_dict()
        
        print(f"  Run {idx+1}/{num_runs}...", end=" ")
        
        res = client.run(
            scale="pilot",
            T=recipe_dict["T"],
            pH=recipe_dict["pH"],
            F1=recipe_dict["F1"],
            F2=recipe_dict["F2"],
            F3=recipe_dict["F3"]
        )
        
        # Extract values using the exact keys from the server
        y_val = res.get("Y")
        cost_val = res.get("cost_eur")
        
        if y_val is None or cost_val is None:
            raise KeyError(f"Unexpected response format: {res}")
            
        total_cost += cost_val
        
        # Track the best product amount found so far
        if y_val > best_y:
            best_y = y_val
            
        print(f"Result: Y = {y_val:.4f} (Best so far: {best_y:.4f})")
        
        # Store for history, ensuring 'scale' is present for plotting script compatibility later
        results.append({
            "Iteration": idx + 1,
            "scale": "pilot",
            "T": recipe_dict["T"],
            "pH": recipe_dict["pH"],
            "F1": recipe_dict["F1"],
            "F2": recipe_dict["F2"],
            "F3": recipe_dict["F3"],
            "Y": y_val,
            "Best_Y": best_y,
            "Cumulative_Cost": total_cost
        })

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("baseline_results.csv", index=False)
    
    print("\n=====================================")
    print(" BASELINE CAMPAIGN COMPLETE")
    print("=====================================")
    print(f"Total Cost Spent: {total_cost} EUR")
    print(f"Best Pilot Y Found: {best_y:.4f} g/L")
    print(f"Data saved to 'baseline_results.csv'")
    
    return df_results

def plot_baseline_results(df: pd.DataFrame):
    """Generates a step-plot trajectory of the baseline optimization campaign."""
    print("\nGenerating Phase 2 visualization...")
    plt.figure(figsize=(9, 6), dpi=300)

    # Insert a 0,0 starting point to ensure the step plot starts from the y-axis
    costs = [0] + df['Cumulative_Cost'].tolist()
    yields = [0] + df['Best_Y'].tolist()

    # Step plot for trajectory
    plt.step(costs, yields, where='post', color='#7f7f7f', linestyle='-', linewidth=2.5, label='Random Pilot Search')

    # Target Benchmark Line
    plt.axhline(y=14.0, color='#d62728', linestyle='--', alpha=0.9, linewidth=1.5, label="Grading Target (14.0 g/L)")

    # Aesthetics
    plt.title("Phase 2: Baseline Optimization Trajectory", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Cumulative Capital Expenditure (EUR)", fontsize=12, fontweight='bold')
    plt.ylabel("Maximum Verified Pilot Yield (g/L)", fontsize=12, fontweight='bold')
    
    plt.xlim(0, max(TOTAL_BUDGET, max(costs)))
    # Dynamically scale Y-axis up to at least 15, or slightly higher than max yield
    plt.ylim(0, max(15.0, max(yields) + 2.0)) 
    
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    output_img = "baseline_trajectory.png"
    plt.savefig(output_img, bbox_inches='tight')
    plt.close()
    print(f" -> Saved visualization to '{output_img}'")

if __name__ == "__main__":
    # Redirect standard output to our custom Logger
    sys.stdout = Logger("baseline_output.txt")
    
    df = run_baseline()
    plot_baseline_results(df)