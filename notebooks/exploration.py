"""
Exploratory Data Analysis — Supply Chain System
================================================
Generates descriptive statistics and visualizations for the synthetic dataset
used across all three modules.

Run from the project root:
    python notebooks/exploration.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import generate_data
from src.ml_models import generate_risk_dataset

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted")

# ── Load data ─────────────────────────────────────────────────────────────────
warehouses, demand_points, cost_matrix, demand_history = generate_data(seed=SEED)
df_risk = generate_risk_dataset(seed=SEED)


# ===========================================================================
# SECTION 1 · DESCRIPTIVE STATISTICS
# ===========================================================================
print("=" * 60)
print("  SECTION 1 · DESCRIPTIVE STATISTICS")
print("=" * 60)

print("\n--- Warehouses ---")
print(warehouses.describe().round(1).to_string())

print("\n--- Demand Points ---")
print(demand_points["base_demand"].describe().round(1).to_string())

print("\n--- Cost Matrix (per unit, $/unit) ---")
cost_flat = cost_matrix.values.flatten()
print(pd.Series(cost_flat, name="cost").describe().round(2).to_string())

print("\n--- Risk Dataset ---")
print(df_risk.describe().round(2).to_string())
print(f"\nRisk class distribution:\n{df_risk['risk'].value_counts().to_string()}")
print(f"Class imbalance ratio: {df_risk['risk'].value_counts()[0] / df_risk['risk'].value_counts()[1]:.2f}:1")


# ===========================================================================
# SECTION 2 · DEMAND DISTRIBUTION
# ===========================================================================
print("\n" + "=" * 60)
print("  SECTION 2 · DEMAND DISTRIBUTION")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Module 1 — Demand & Capacity Distribution", fontsize=13, fontweight="bold")

# 2.1 Base demand per point
axes[0].bar(
    demand_points["point_id"],
    demand_points["base_demand"],
    color=sns.color_palette("muted")[0],
    edgecolor="white"
)
axes[0].axhline(demand_points["base_demand"].mean(), color="red", linestyle="--", label="Mean")
axes[0].set_title("Base Demand per Point")
axes[0].set_xlabel("Demand Point")
axes[0].set_ylabel("Units")
axes[0].legend()

# 2.2 Warehouse capacity
axes[1].bar(
    warehouses["warehouse_id"],
    warehouses["capacity"],
    color=sns.color_palette("muted")[2],
    edgecolor="white"
)
axes[1].axhline(
    demand_points["base_demand"].sum() / len(warehouses),
    color="red", linestyle="--", label="Avg demand share"
)
axes[1].set_title("Warehouse Capacity")
axes[1].set_xlabel("Warehouse")
axes[1].set_ylabel("Units")
axes[1].legend()

# 2.3 Cost distribution
axes[2].hist(cost_flat, bins=15, color=sns.color_palette("muted")[4], edgecolor="white")
axes[2].axvline(cost_flat.mean(), color="red", linestyle="--", label=f"Mean = {cost_flat.mean():.1f}")
axes[2].set_title("Transport Cost Distribution")
axes[2].set_xlabel("Cost ($/unit)")
axes[2].set_ylabel("Frequency")
axes[2].legend()

plt.tight_layout()
plt.savefig("notebooks/fig1_demand_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
print("  -> Saved: notebooks/fig1_demand_distribution.png")


# ===========================================================================
# SECTION 3 · DEMAND TIME SERIES (SELECTED POINTS)
# ===========================================================================
print("\n" + "=" * 60)
print("  SECTION 3 · DEMAND TIME SERIES")
print("=" * 60)

selected_points = [1, 5, 10, 15, 20]
fig, axes = plt.subplots(len(selected_points), 1, figsize=(14, 10), sharex=True)
fig.suptitle("Module 1 — Weekly Demand History (Selected Points)", fontsize=13, fontweight="bold")

for ax, pid in zip(axes, selected_points):
    series = demand_history[demand_history["point_id"] == pid].sort_values("week")
    ax.plot(series["week"], series["demand"], linewidth=1.5, label=f"P{pid}")
    ax.fill_between(series["week"], series["demand"], alpha=0.15)
    ax.set_ylabel("Units")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)

axes[-1].set_xlabel("Week")
plt.tight_layout()
plt.savefig("notebooks/fig2_demand_time_series.png", dpi=120, bbox_inches="tight")
plt.show()
print("  -> Saved: notebooks/fig2_demand_time_series.png")


# ===========================================================================
# SECTION 4 · COST MATRIX HEATMAP
# ===========================================================================
print("\n" + "=" * 60)
print("  SECTION 4 · COST MATRIX HEATMAP")
print("=" * 60)

fig, ax = plt.subplots(figsize=(16, 4))
fig.suptitle("Module 1 — Transport Cost Matrix (Warehouse x Demand Point)",
             fontsize=13, fontweight="bold")

sns.heatmap(
    cost_matrix,
    annot=True, fmt="d",
    cmap="YlOrRd",
    linewidths=0.4,
    ax=ax,
    cbar_kws={"label": "Cost ($/unit)"}
)
ax.set_xlabel("Demand Points")
ax.set_ylabel("Warehouses")
plt.tight_layout()
plt.savefig("notebooks/fig3_cost_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print("  -> Saved: notebooks/fig3_cost_matrix.png")


# ===========================================================================
# SECTION 5 · RISK DATASET EXPLORATION
# ===========================================================================
print("\n" + "=" * 60)
print("  SECTION 5 · RISK DATASET (Module 2B)")
print("=" * 60)

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Module 2B — Risk Classification: Feature Analysis", fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(2, 4, figure=fig)

features = ["stock", "lead_time", "demand", "distance"]
colors   = {0: "#4C72B0", 1: "#DD8452"}
labels   = {0: "Low Risk", 1: "High Risk"}

# 5.1 Distribution per feature by class
for i, feat in enumerate(features):
    ax = fig.add_subplot(gs[0, i])
    for cls in [0, 1]:
        subset = df_risk[df_risk["risk"] == cls][feat]
        ax.hist(subset, bins=15, alpha=0.65, color=colors[cls], label=labels[cls], edgecolor="white")
    ax.set_title(feat.replace("_", " ").title())
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    if i == 0:
        ax.legend(fontsize=8)

# 5.2 Correlation matrix
ax_corr = fig.add_subplot(gs[1, :2])
corr = df_risk.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=ax_corr, square=True)
ax_corr.set_title("Feature Correlation Matrix")

# 5.3 Class balance
ax_bar = fig.add_subplot(gs[1, 2])
counts = df_risk["risk"].value_counts().sort_index()
ax_bar.bar(["Low Risk", "High Risk"], counts.values,
           color=[colors[0], colors[1]], edgecolor="white")
ax_bar.set_title("Class Distribution")
ax_bar.set_ylabel("Samples")
for j, v in enumerate(counts.values):
    ax_bar.text(j, v + 1, str(v), ha="center", fontsize=10)

# 5.4 Coverage ratio vs risk
ax_cov = fig.add_subplot(gs[1, 3])
df_risk["coverage"] = df_risk["stock"] / (df_risk["demand"] * df_risk["lead_time"] * 0.15)
for cls in [0, 1]:
    subset = df_risk[df_risk["risk"] == cls]["coverage"].clip(upper=5)
    ax_cov.hist(subset, bins=15, alpha=0.65, color=colors[cls],
                label=labels[cls], edgecolor="white")
ax_cov.set_title("Coverage Ratio by Class")
ax_cov.set_xlabel("stock / (demand x lead_time x 0.15)")
ax_cov.set_ylabel("Count")
ax_cov.legend(fontsize=8)

plt.tight_layout()
plt.savefig("notebooks/fig4_risk_features.png", dpi=120, bbox_inches="tight")
plt.show()
print("  -> Saved: notebooks/fig4_risk_features.png")


# ===========================================================================
# SECTION 6 · KEY INSIGHTS SUMMARY
# ===========================================================================
print("\n" + "=" * 60)
print("  SECTION 6 · KEY INSIGHTS")
print("=" * 60)

total_demand   = demand_points["base_demand"].sum()
total_capacity = warehouses["capacity"].sum()
slack          = total_capacity - total_demand

print(f"""
Network overview
  Warehouses          : {len(warehouses)}
  Demand points       : {len(demand_points)}
  Total demand        : {total_demand:,} units/week
  Total capacity      : {total_capacity:,} units/week
  Capacity slack      : {slack:,} units ({slack/total_demand:.1%} above demand)

Cost matrix
  Min cost route      : {cost_flat.min()} $/unit
  Max cost route      : {cost_flat.max()} $/unit
  Mean cost           : {cost_flat.mean():.1f} $/unit
  Std deviation       : {cost_flat.std():.1f} $/unit

Demand variability (52-week history)
  Mean weekly demand  : {demand_history['demand'].mean():.1f} units
  Std deviation       : {demand_history['demand'].std():.1f} units
  Coefficient of var  : {demand_history['demand'].std()/demand_history['demand'].mean():.2%}

Risk dataset
  Total samples       : {len(df_risk)}
  High-risk samples   : {df_risk['risk'].sum()} ({df_risk['risk'].mean():.1%})
  Low-risk samples    : {(df_risk['risk']==0).sum()} ({(df_risk['risk']==0).mean():.1%})
  Most correlated     : stock vs risk ({corr.loc['stock','risk']:.3f})
""")

print("Exploration complete. Figures saved to notebooks/")
