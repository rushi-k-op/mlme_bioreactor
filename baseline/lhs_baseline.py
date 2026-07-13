import pandas as pd
from scipy.stats import qmc

# --- Configuration ---
import sys
sys.path.insert(0, ".")
from basic_client import BioreactorClient, USER, PASSWORD

TOTAL_BUDGET = 15000  
PILOT_COST = 2000
BOUNDS_LIST = [(20.0, 60.0), (3.0, 9.5), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
KEYS = ["T", "pH", "F1", "F2", "F3"]

def run_baseline_campaign():
    client = BioreactorClient()
    client.login(USER, PASSWORD)
    print("Logged in successfully.")
    
    print("\n=====================================")
    print(" CAMPAIGN 1: BASELINE (LHS PILOT ONLY)")
    print("=====================================")
    
    # Calculate how many pilot runs we can afford
    num_runs = TOTAL_BUDGET // PILOT_COST
    print(f"Budget: {TOTAL_BUDGET} EUR | Pilot Runs: {num_runs}")
    
    # Generate LHS samples
    sampler = qmc.LatinHypercube(d=5)
    sample_unit = sampler.random(n=num_runs)
    lower_bounds = [b[0] for b in BOUNDS_LIST]
    upper_bounds = [b[1] for b in BOUNDS_LIST]
    samples_scaled = qmc.scale(sample_unit, lower_bounds, upper_bounds)
    
    best_y = 0.0
    total_cost = 0.0
    results = []

    # Execute runs
    for idx, row in enumerate(samples_scaled):
        res = client.run(scale="pilot", T=row[0], pH=row[1], F1=row[2], F2=row[3], F3=row[4])
        
        y_val = res["Y"]
        total_cost += res["cost_eur"]
        best_y = max(best_y, y_val)
        
        print(f"Run {idx+1}/{num_runs} | Cost: {total_cost} EUR | Y: {y_val:.4f} (Best: {best_y:.4f})")
        
        results.append({
            "Iteration": idx+1, "Scale": "pilot", "Cost_Spent": total_cost, 
            "T": row[0], "pH": row[1], "F1": row[2], "F2": row[3], "F3": row[4],
            "Y": y_val, "Best_Y": best_y
        })
        
    # Save to CSV
    pd.DataFrame(results).to_csv("results_baseline.csv", index=False)
    print("Baseline complete. Data saved to 'results_baseline.csv'")

if __name__ == "__main__":
    run_baseline_campaign()