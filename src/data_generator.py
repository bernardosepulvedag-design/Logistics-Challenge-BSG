import numpy as np
import pandas as pd


def generate_data(seed: int = 42):
    np.random.seed(seed)

    # -----------------------
    # 1. DEMAND POINTS (1–20)
    # -----------------------
    n_points = 20

    base_demand = np.random.randint(80, 250, size=n_points)

    demand_points = pd.DataFrame({
        "point_id": range(1, n_points + 1),
        "base_demand": base_demand
    })

    total_demand = base_demand.sum()

    # -----------------------
    # 2. WAREHOUSES (1–5)
    # -----------------------
    n_warehouses = 5

    raw_caps = np.random.uniform(0.8, 1.2, size=n_warehouses)

    capacities = (raw_caps / raw_caps.sum()) * total_demand * 1.3

    warehouses = pd.DataFrame({
        "warehouse_id": range(1, n_warehouses + 1),
        "capacity": capacities.astype(int)
    })

    # sanity check
    assert warehouses["capacity"].sum() >= demand_points["base_demand"].sum()

    # -----------------------
    # 3. COST MATRIX (W1–W5 x P1–P20)
    # -----------------------
    cost_matrix = pd.DataFrame(
        np.random.randint(5, 50, size=(n_warehouses, n_points)),
        index=[f"W{i}" for i in range(1, n_warehouses + 1)],
        columns=[f"P{j}" for j in range(1, n_points + 1)]
    )

    # -----------------------
    # 4. DEMAND TIME SERIES (52 weeks)
    # -----------------------
    weeks = 52

    rows = []

    for p in range(1, n_points + 1):
        base = demand_points.loc[p - 1, "base_demand"]

        for w in range(weeks):
            seasonal = 15 * np.sin(2 * np.pi * w / 52)
            noise = np.random.normal(0, 8)

            demand = max(0, base + seasonal + noise)

            rows.append([p, w, demand])

    demand_history = pd.DataFrame(rows, columns=["point_id", "week", "demand"])

    return warehouses, demand_points, cost_matrix, demand_history