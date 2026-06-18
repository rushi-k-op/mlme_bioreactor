import pandas as pd
from scipy.stats import qmc

# --- Centralized Configuration Import ---
from client import BioreactorClient, USER, PASSWORD, BASE_URL

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

def run_baseline():
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
        
        # Store for history
        results.append({
            "Iteration": idx + 1,
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

if __name__ == "__main__":
    run_baseline()