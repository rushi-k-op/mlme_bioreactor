import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Load Data
# =====================================================

df = pd.read_csv("exploratory_data.csv")

# Desired order of scales
order = ["micro", "bench", "pilot"]

# =====================================================
# Mean Response by Scale
# =====================================================

mean_scale = (
    df.groupby("Scale")["Y"]
      .mean()
      .reindex(order)
)

# =====================================================
# Variance for Each Recipe
# =====================================================

var_stats = (
    df.groupby(["Recipe_ID", "Scale"])["Y"]
      .var()
      .reset_index(name="Variance")
)

# Set plotting order
var_stats["Scale"] = pd.Categorical(
    var_stats["Scale"],
    categories=order,
    ordered=True
)

# =====================================================
# Coefficient of Variation (CV)
# =====================================================

cv_stats = (
    df.groupby(["Recipe_ID", "Scale"])["Y"]
      .agg(["mean", "std"])
      .reset_index()
)

cv_stats["CV"] = (cv_stats["std"] / cv_stats["mean"]) * 100

# Remove any NaN values (good practice)
cv_stats = cv_stats.dropna(subset=["CV"])

# Set plotting order
cv_stats["Scale"] = pd.Categorical(
    cv_stats["Scale"],
    categories=order,
    ordered=True
)

# =====================================================
# Create One Figure with Four Plots
# =====================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# =====================================================
# Plot 1 : Mean Response
# =====================================================

axes[0,0].bar(mean_scale.index, mean_scale.values)

axes[0,0].set_title("Average Measurement by Scale")
axes[0,0].set_xlabel("Scale")
axes[0,0].set_ylabel("Average Response (Y)")
axes[0,0].grid(axis="y", alpha=0.3)

# =====================================================
# Plot 2 : Variance Boxplot
# =====================================================

var_stats.boxplot(
    column="Variance",
    by="Scale",
    ax=axes[0,1]
)

axes[0,1].set_title("Measurement Variance")
axes[0,1].set_xlabel("Scale")
axes[0,1].set_ylabel("Variance")

# =====================================================
# Plot 3 : CV Boxplot
# =====================================================

cv_stats.boxplot(
    column="CV",
    by="Scale",
    ax=axes[1,0]
)

axes[1,0].set_title("Coefficient of Variation")
axes[1,0].set_xlabel("Scale")
axes[1,0].set_ylabel("CV (%)")

# =====================================================
# Plot 4 : Mean vs Standard Deviation
# =====================================================

for scale in order:

    subset = cv_stats[cv_stats["Scale"] == scale]

    axes[1,1].scatter(
        subset["mean"],
        subset["std"],
        s=40,
        label=scale
    )

axes[1,1].set_title("Standard Deviation vs Mean")
axes[1,1].set_xlabel("Mean Measurement")
axes[1,1].set_ylabel("Standard Deviation")
axes[1,1].grid(alpha=0.3)
axes[1,1].legend(title="Scale")

# Remove the automatic pandas title
plt.suptitle("")

plt.tight_layout()

plt.show()