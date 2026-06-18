import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# --- Centralized Configuration Import ---
from client import BioreactorClient, USER, PASSWORD, BASE_URL

HISTORY_FILE = "phase3test4.csv"  # The clean Phase 1 LHS data
BO_BUDGET = 15000  

BOUNDS_LIST = [(20.0, 60.0), (3.0, 9.5), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
FIDELITY_MAP = {"micro": 0.0, "bench": 0.5, "pilot": 1.0}
SCALE_COSTS = {"micro": 10, "bench": 500, "pilot": 2000}

# ==========================================
# 6D Multi-Fidelity Data Preparation
# ==========================================
def prepare_6d_data(df):
    """Normalizes 5D inputs and appends the 6th Fidelity dimension."""
    X, Y = [], []
    for _, row in df.iterrows():
        if pd.isna(row.get('Y')): continue
        
        # Normalize the 5 physical parameters to [0, 1]
        x_norm = [(row['T'] - BOUNDS_LIST[0][0]) / (BOUNDS_LIST[0][1] - BOUNDS_LIST[0][0]),
                  (row['pH'] - BOUNDS_LIST[1][0]) / (BOUNDS_LIST[1][1] - BOUNDS_LIST[1][0]),
                  (row['F1'] - BOUNDS_LIST[2][0]) / (BOUNDS_LIST[2][1] - BOUNDS_LIST[2][0]),
                  (row['F2'] - BOUNDS_LIST[3][0]) / (BOUNDS_LIST[3][1] - BOUNDS_LIST[3][0]),
                  (row['F3'] - BOUNDS_LIST[4][0]) / (BOUNDS_LIST[4][1] - BOUNDS_LIST[4][0])]
        
        # Add the 6th dimension: Fidelity
        scale_name = str(row['scale']).lower()
        x_norm.append(FIDELITY_MAP.get(scale_name, 0.0))
        
        X.append(x_norm)
        Y.append(row['Y'])
        
    return np.array(X), np.array(Y).reshape(-1, 1)

def denormalize_5d_recipe(x_norm):
    """Maps normalized [0, 1] points back to physical reactor limits."""
    x_raw = []
    for i in range(5):
        x_val = BOUNDS_LIST[i][0] + x_norm[i] * (BOUNDS_LIST[i][1] - BOUNDS_LIST[i][0])
        x_raw.append(x_val)
    return np.array(x_raw)

# ==========================================
# Core Bayesian Optimization Logic
# ==========================================
def expected_improvement_pilot(X_5D_norm, Y_sample, gpr, xi=0.01):
    """Calculates EI explicitly targeting the Pilot scale (Fidelity = 1.0)."""
    fidelity_col = np.ones((X_5D_norm.shape[0], 1))
    X_6D = np.hstack([X_5D_norm, fidelity_col])
    
    mu, sigma = gpr.predict(X_6D, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='ignore', invalid='ignore'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
    return ei

def optimize_acquisition(Y_sample, gpr):
    """Searches the normalized 5D space for the best Pilot prediction."""
    bounds = [(0.0, 1.0)] * 5
    def objective(x):
        return -expected_improvement_pilot(x.reshape(1, -1), Y_sample, gpr)[0][0]

    result = differential_evolution(objective, bounds, maxiter=30, popsize=20)
    return result.x

def run_bayesian_optimization():
    client = BioreactorClient(BASE_URL)
    print("Logging into the bioreactor...")
    client.login(USER, PASSWORD)
    
    try:
        df_history = pd.read_csv(HISTORY_FILE)
        if 'Scale' in df_history.columns:
            df_history = df_history.rename(columns={'Scale': 'scale'})
    except FileNotFoundError:
        print(f"Error: Could not find '{HISTORY_FILE}'.")
        return

    # THE FIX: Expanded length_scale bounds to allow the RBF to fit smoothly
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*6, length_scale_bounds=(1e-3, 1e4)) + WhiteKernel(noise_level=1e-3)
    
    budget_spent = 0.0
    iteration = 1
    best_pilot_y = 0.0

    print("\n=====================================")
    print(" STARTING FLAWLESS 6D BO LOOP")
    print("=====================================")

    while budget_spent < BO_BUDGET:
        X_train, Y_train = prepare_6d_data(df_history)
        
        pilot_runs = df_history[df_history['scale'] == 'pilot']
        if not pilot_runs.empty:
            best_pilot_y = pilot_runs['Y'].max()

        print(f"\n--- Iteration {iteration} | Spent: {budget_spent}/{BO_BUDGET} EUR | Best Pilot Y: {best_pilot_y:.4f} ---")
        
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        # Suppress generic sklearn warnings for a clean terminal output
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X_train, Y_train)

        next_5d_norm = optimize_acquisition(Y_train, gpr)
        next_5d_raw = denormalize_5d_recipe(next_5d_norm)
        
        X_eval_6D = np.append(next_5d_norm, 1.0).reshape(1, -1)
        mu_val, std_val = gpr.predict(X_eval_6D, return_std=True)
        mu_val, std_val = mu_val.item(), std_val.item()
        
        # The Phased Budget Strategy
        if budget_spent < 1500:
            chosen_scale = "micro"
        elif std_val > 1.5 and (budget_spent + SCALE_COSTS["micro"] <= BO_BUDGET):
            chosen_scale = "micro"
        elif budget_spent + SCALE_COSTS["pilot"] <= BO_BUDGET:
            chosen_scale = "pilot"
        elif budget_spent + SCALE_COSTS["bench"] <= BO_BUDGET:
            chosen_scale = "bench"
        elif budget_spent + SCALE_COSTS["micro"] <= BO_BUDGET:
            chosen_scale = "micro"
        else:
            break

        cost_of_run = SCALE_COSTS[chosen_scale]
        budget_spent += cost_of_run

        print(f"  Target -> T={next_5d_raw[0]:.1f}, pH={next_5d_raw[1]:.2f}, Fs=[{next_5d_raw[2]:.2f}, {next_5d_raw[3]:.2f}, {next_5d_raw[4]:.2f}]")
        print(f"  Scale  -> {chosen_scale.upper()} (Pilot Prediction: {mu_val:.2f} +/- {std_val:.2f})")
        
        res = client.run(
            scale=chosen_scale,
            T=float(next_5d_raw[0]), pH=float(next_5d_raw[1]),
            F1=float(next_5d_raw[2]), F2=float(next_5d_raw[3]), F3=float(next_5d_raw[4])
        )
        
        y_val = res.get("Y", res.get("y"))
        print(f"  Result -> Actual Output Y: {y_val:.4f}")
        
        new_row = pd.DataFrame([{
            'scale': chosen_scale,
            'T': next_5d_raw[0], 'pH': next_5d_raw[1],
            'F1': next_5d_raw[2], 'F2': next_5d_raw[3], 'F3': next_5d_raw[4],
            'Y': y_val, 'cost_eur': cost_of_run
        }])
        df_history = pd.concat([df_history, new_row], ignore_index=True)
        iteration += 1

    print("\n=====================================")
    print(" BO CAMPAIGN COMPLETE")
    print(f" Final Maximum Pilot Yield: {best_pilot_y:.4f} g/L")
    print("=====================================")
    df_history.to_csv("bo_final_results_Flawless.csv", index=False)

if __name__ == "__main__":
    run_bayesian_optimization()