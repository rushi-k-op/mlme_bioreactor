import numpy as np
import pandas as pd
from scipy.stats import qmc

# --- Centralized Configuration Import ---
from client import BioreactorClient, USER, PASSWORD, BASE_URL

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

if __name__ == "__main__":
    df = run_exploration()
    analyze_noise_and_bias(df)

    #akkkkkk