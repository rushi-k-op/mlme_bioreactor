import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# -------- Path and diagnostic ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)          # mlme_bioreactor/
csv_path = os.path.join(project_root, 'data', 'experiments.csv')

print(f"Looking for file: {csv_path}")
print(f"File exists? {os.path.exists(csv_path)}")
if os.path.exists(csv_path):
    print(f"File size: {os.path.getsize(csv_path)} bytes")
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        print(f"Number of lines: {len(lines)}")
        if len(lines) > 0:
            print("First 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"  {i+1}: {line.strip()}")
        else:
            print("The file is completely empty.")
            raise SystemExit("Cannot proceed with empty CSV file.")
else:
    raise FileNotFoundError(f"File not found at {csv_path}")

# Now read
df = pd.read_csv(csv_path)
print(f"Loaded DataFrame with shape {df.shape}")

# -------- Plotting ----------
baseline = {'T': 35, 'pH': 7, 'F1': 2, 'F2': 2, 'F3': 2}
sweeps = [
    {'var': 'T',  'fixed': {'pH': 7, 'F1': 2, 'F2': 2, 'F3': 2}},
    {'var': 'pH', 'fixed': {'T': 35, 'F1': 2, 'F2': 2, 'F3': 2}},
    {'var': 'F1', 'fixed': {'T': 35, 'pH': 7, 'F2': 2, 'F3': 2}},
    {'var': 'F2', 'fixed': {'T': 35, 'pH': 7, 'F1': 2, 'F3': 2}},
    {'var': 'F3', 'fixed': {'T': 35, 'pH': 7, 'F1': 2, 'F2': 2}},
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, sweep in enumerate(sweeps):
    var = sweep['var']
    fixed = sweep['fixed']
    mask = pd.Series([True] * len(df))
    for key, val in fixed.items():
        mask &= (df[key] == val)
    subset = df[mask]
    grouped = subset.groupby(var)['Y'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.sort_values(var)
    ax = axes[idx]
    ax.errorbar(grouped[var], grouped['mean'], yerr=grouped['std'],
                fmt='o-', capsize=5, capthick=2, ecolor='gray',
                label='Mean ± std')
    if var in baseline:
        ax.axvline(x=baseline[var], color='r', linestyle='--', alpha=0.7,
                   label=f'baseline {var} = {baseline[var]}')
    ax.set_xlabel(var)
    ax.set_ylabel('Yield Y')
    ax.set_title(f'Effect of {var} (others fixed)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()