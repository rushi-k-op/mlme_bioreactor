import pandas as pd
import matplotlib.pyplot as plt

def visualize_dependencies(csv_filepath="exploratory_data.csv"):
    print(f"Reading data from {csv_filepath}...")
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filepath}.")
        return

    print("Generating dependency visualizations...")

    variables = ['T', 'pH', 'F1', 'F2', 'F3']
    scales = ['micro', 'bench', 'pilot']
    colors = {'micro': 'tab:blue', 'bench': 'tab:orange', 'pilot': 'tab:green'}

    # One row of 5 plots
    fig, axes = plt.subplots(5, 1, figsize=(6, 20))
    fig.suptitle('Dependence of Yield (Y) on Recipe Variables across Scales', fontsize=16)

    for i, var in enumerate(variables):
        for scale in scales:
            scale_data = df[df['Scale'] == scale].sort_values(by=var)

            # Connect points with lines
            axes[i].plot(
                scale_data[var],
                scale_data['Y'],
                '-o',                      # line + circle markers
                color=colors[scale],
                label=scale,
                linewidth=1.5,
                markersize=5,
                alpha=0.8
            )

        axes[i].set_title(f'Yield vs {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Yield (Y)')
        axes[i].grid(True, linestyle='--', alpha=0.5)

        if i == 0:
            axes[i].legend(title="Scale")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('dependency_visualization.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    visualize_dependencies()