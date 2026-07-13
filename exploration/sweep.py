import numpy as np
import pandas as pd
import csv, sys, time
sys.path.insert(0, ".")
from basic_client import BioreactorClient, USER, PASSWORD
from scipy.stats import qmc

client = BioreactorClient()
client.login(USER, PASSWORD)
print("Logged in successfully.")

# Recipe Bounds
BOUNDS = {
    "T": (20.0, 60.0),
    "pH": (3.0, 9.5),
    "F1": (0.0, 2.0),
    "F2": (0.0, 2.0),
    "F3": (0.0, 2.0)
}

SCALES = ["micro", "bench", "pilot"]
NUM_RECIPES = 10     
REPEATS = 3          

def generate_lhs_samples(n_samples, bounds):
    # Setup the LHS sampler
    sampler = qmc.LatinHypercube(d=len(bounds))
    sample_unit = sampler.random(n=n_samples)
    
    # Get lower and upper bounds as simple lists
    lower_bounds = [bounds[k][0] for k in bounds]
    upper_bounds = [bounds[k][1] for k in bounds]
    
    # Scale samples to real physical ranges
    samples_scaled = qmc.scale(sample_unit, lower_bounds, upper_bounds)
    
    # Return as a simple dataframe
    return pd.DataFrame(samples_scaled, columns=bounds.keys())

def run_exploration():
    
    print(f"Generating {NUM_RECIPES} LHS recipes...")
    recipes_df = generate_lhs_samples(NUM_RECIPES, BOUNDS)
    
    results = []
    total_cost_spent = 0.0

    # Loop through each generated recipe row
    for idx, row in recipes_df.iterrows():
        print(f"\n--- Testing Recipe {idx+1}/{NUM_RECIPES} ---")
        
        # Test across all scales and repeats
        for scale in SCALES:
            for rep in range(REPEATS):
                print(f"Running {scale.upper()} (Repeat {rep+1}/{REPEATS})...")
                
                # Send the parameters directly from the row
                res = client.run(
                    scale=scale,
                    T=row["T"],
                    pH=row["pH"],
                    F1=row["F1"],
                    F2=row["F2"],
                    F3=row["F3"]
                )

                # Extract outputs from server response
                y_val = res["Y"]
                cost_val = res["cost_eur"]
                total_cost_spent += cost_val
                
                print(f"Result: Y = {y_val:.4f}, Cost = {cost_val} EUR")
                
                # Save data to list
                results.append({
                    "Recipe_ID": idx,
                    "Scale": scale,
                    "Repeat": rep,
                    "T": row["T"],
                    "pH": row["pH"],
                    "F1": row["F1"],
                    "F2": row["F2"],
                    "F3": row["F3"],
                    "Y": y_val,
                    "Cost": cost_val
                })
        time.sleep(1.0)

    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("exploratory_data.csv", index=False)
    print(f"\nExploration Complete! Saved to 'exploratory_data.csv'.")
    print(f"Total budget spent: {total_cost_spent} EUR")
    
    return df_results

def analyze_noise_and_bias(df, output_filename="noise_bias_results.txt"):
    # Create a list to store all the text we want to print and save
    output = []
    
    output.append("=====================================")
    output.append(" NOISE AND BIAS ANALYSIS")
    output.append("=====================================")
    
    # 1. Noise Analysis (Relative Noise / Coefficient of Variation)
    output.append("\n--- Measurement Noise (Coefficient of Variation) ---")
    for scale in SCALES:
        scale_df = df[df["Scale"] == scale]
        
        # Calculate mean and standard deviation for each recipe
        recipe_stats = scale_df.groupby("Recipe_ID")["Y"].agg(['mean', 'std'])
        recipe_stats['CV'] = recipe_stats['std'] / recipe_stats['mean']
        avg_cv = recipe_stats['CV'].mean()
        
        output.append(f"Scale: {scale:5s} | Relative Noise (CV): {avg_cv * 100:.2f}%")
        
    # 2. Bias Analysis (Relative to Pilot scale)
    output.append("\n--- Measurement Bias (Relative to Pilot) ---")
    
    # Extract mean yield for each recipe per scale
    mean_yields = df.groupby(["Recipe_ID", "Scale"])["Y"].mean().unstack()
    mean_yields["Bias_Micro"] = mean_yields["micro"] - mean_yields["pilot"]
    mean_yields["Bias_Bench"] = mean_yields["bench"] - mean_yields["pilot"]
    
    output.append(f"Micro Scale - Average Bias: {mean_yields['Bias_Micro'].mean():.4f} | Absolute Bias: {mean_yields['Bias_Micro'].abs().mean():.4f}")
    output.append(f"Bench Scale - Average Bias: {mean_yields['Bias_Bench'].mean():.4f} | Absolute Bias: {mean_yields['Bias_Bench'].abs().mean():.4f}")
    
    # Combine the list into a single text block separated by newlines
    final_report = "\n".join(output)
    
    # Print to the console for the user to see
    print(final_report)
    
    # Save the text block to the specified file
    with open(output_filename, "w") as file:
        file.write(final_report)
        
    print(f"\n[Info] Results successfully saved to: {output_filename}")

if __name__ == "__main__":
    df = run_exploration()
    analyze_noise_and_bias(df)