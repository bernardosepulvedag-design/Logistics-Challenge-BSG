import matplotlib.pyplot as plt
import seaborn as sns


def plot_results(warehouses, demand_points, solution, base_cost, stress_cost):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # -----------------------
    # FLOW HEATMAP
    # -----------------------
    if solution.empty:
        print("No flows to display")
        return

    pivot = solution.pivot_table(
        index="warehouse",
        columns="point",
        values="flow",
        fill_value=0
    )

    sns.heatmap(pivot, cmap="Blues", ax=axes[0])

    axes[0].set_title("Optimal Flow (Warehouse → Demand)")
    axes[0].set_xlabel("Demand Points")
    axes[0].set_ylabel("Warehouses")

    # -----------------------
    # COST COMPARISON
    # -----------------------
    axes[1].bar(
        ["Base", "Stress +20%"],
        [base_cost, stress_cost]
    )

    axes[1].set_title("Cost Comparison")
    axes[1].set_ylabel("Total Cost")

    plt.tight_layout()
    plt.show()