import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# --- Configuration ---
import sys
sys.path.insert(0, ".")
from basic_client import BioreactorClient, USER, PASSWORD

HISTORY_FILE = "exploratory_data.csv"
TOTAL_BUDGET = 15000  

BOUNDS_LIST = [(20.0, 60.0), (3.0, 9.5), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
FIDELITY_MAP = {"micro": 0.0, "bench": 0.5, "pilot": 1.0}
SCALE_COSTS = {"micro": 10, "bench": 500, "pilot": 2000}

def prepare_6d_data(df):
    X, Y = [], []
    for _, row in df.iterrows():
        if pd.isna(row.get('Y')): continue
        
        x_norm = [(row['T'] - BOUNDS_LIST[0][0]) / (BOUNDS_LIST[0][1] - BOUNDS_LIST[0][0]),
                  (row['pH'] - BOUNDS_LIST[1][0]) / (BOUNDS_LIST[1][1] - BOUNDS_LIST[1][0]),
                  (row['F1'] - BOUNDS_LIST[2][0]) / (BOUNDS_LIST[2][1] - BOUNDS_LIST[2][0]),
                  (row['F2'] - BOUNDS_LIST[3][0]) / (BOUNDS_LIST[3][1] - BOUNDS_LIST[3][0]),
                  (row['F3'] - BOUNDS_LIST[4][0]) / (BOUNDS_LIST[4][1] - BOUNDS_LIST[4][0])]
        
        x_norm.append(FIDELITY_MAP.get(str(row['Scale']).lower(), 0.0))
        X.append(x_norm)
        Y.append(row['Y'])
    return np.array(X), np.array(Y).reshape(-1, 1)

def denormalize_5d_recipe(x_norm):
    return np.array([BOUNDS_LIST[i][0] + x_norm[i] * (BOUNDS_LIST[i][1] - BOUNDS_LIST[i][0]) for i in range(5)])

def expected_improvement_pilot(X_5D_norm, Y_sample, gpr, xi=0.01):
    fidelity_col = np.ones((X_5D_norm.shape[0], 1))
    X_6D = np.hstack([X_5D_norm, fidelity_col])
    
    mu, sigma = gpr.predict(X_6D, return_std=True)
    mu, sigma = mu.reshape(-1, 1), sigma.reshape(-1, 1)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='ignore', invalid='ignore'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
    return ei

def optimize_acquisition(Y_sample, gpr):
    bounds = [(0.0, 1.0)] * 5
    def objective(x):
        return -expected_improvement_pilot(x.reshape(1, -1), Y_sample, gpr)[0][0]
    result = differential_evolution(objective, bounds, maxiter=30, popsize=20)
    return result.x

def run_bo_campaign():
    client = BioreactorClient()
    client.login(USER, PASSWORD)
    print("Logged in successfully.")
    
    print("\n=====================================")
    print(" CAMPAIGN 2: COST-AWARE BO LOOP")
    print("=====================================")
    
    df_history = pd.read_csv(HISTORY_FILE)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*6, length_scale_bounds=(1e-3, 1e4)) + WhiteKernel(noise_level=1e-3)
    
    budget_spent = 0.0
    iteration = 1
    best_pilot_y = df_history[df_history['Scale'] == 'pilot']['Y'].max() if 'pilot' in df_history['Scale'].values else 0.0
    results = []

    while budget_spent < TOTAL_BUDGET:
        X_train, Y_train = prepare_6d_data(df_history)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X_train, Y_train)

        next_5d_norm = optimize_acquisition(Y_train, gpr)
        next_5d_raw = denormalize_5d_recipe(next_5d_norm)
        
        X_eval_6D = np.append(next_5d_norm, 1.0).reshape(1, -1)
        mu_val, std_val = gpr.predict(X_eval_6D, return_std=True)
        std_val = std_val.item()

        # Dynamic Scale Selection
        if std_val > 0.8 and (budget_spent + SCALE_COSTS["micro"] <= TOTAL_BUDGET):
            chosen_scale = "micro"
        elif std_val > 0.3 and (budget_spent + SCALE_COSTS["bench"] <= TOTAL_BUDGET):
            chosen_scale = "bench"
        elif budget_spent + SCALE_COSTS["pilot"] <= TOTAL_BUDGET:
            chosen_scale = "pilot"
        else:
            chosen_scale = "micro" if budget_spent + SCALE_COSTS["micro"] <= TOTAL_BUDGET else None
            
        if not chosen_scale:
            break

        cost_of_run = SCALE_COSTS[chosen_scale]
        budget_spent += cost_of_run
        
        print(f"Iter {iteration} | Spent: {budget_spent}/{TOTAL_BUDGET} | Scale: {chosen_scale.upper()} (Uncertainty: {std_val:.2f})")
        
        res = client.run(
            scale=chosen_scale, 
            T=next_5d_raw[0], pH=next_5d_raw[1], F1=next_5d_raw[2], F2=next_5d_raw[3], F3=next_5d_raw[4]
        )
        y_val = res["Y"]
        
        if chosen_scale == "pilot":
            best_pilot_y = max(best_pilot_y, y_val)
            
        print(f"  -> Y: {y_val:.4f} | Best Pilot: {best_pilot_y:.4f}")
        
        new_row = pd.DataFrame([{
            'Scale': chosen_scale, 'T': next_5d_raw[0], 'pH': next_5d_raw[1],
            'F1': next_5d_raw[2], 'F2': next_5d_raw[3], 'F3': next_5d_raw[4], 'Y': y_val
        }])
        df_history = pd.concat([df_history, new_row], ignore_index=True)
        
        results.append({"Iteration": iteration, "Scale": chosen_scale, "Cost_Spent": budget_spent, "Y": y_val, "Best_Y": best_pilot_y})
        iteration += 1

    pd.DataFrame(results).to_csv("results_bo.csv", index=False)
    print("\nBO Campaign complete. Data saved to 'results_bo.csv'")

if __name__ == "__main__":
    run_bo_campaign()