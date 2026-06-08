import pulp
import pandas as pd


def solve_transport(warehouses, demand_points, cost_matrix):

    model = pulp.LpProblem("transport_problem", pulp.LpMinimize)

    w_ids = warehouses["warehouse_id"].tolist()
    p_ids = demand_points["point_id"].tolist()

    x = pulp.LpVariable.dicts(
        "ship",
        ((i, j) for i in w_ids for j in p_ids),
        lowBound=0,
        cat="Continuous"
    )

    model += pulp.lpSum(
        cost_matrix.loc[f"W{i}", f"P{j}"] * x[(i, j)]
        for i in w_ids for j in p_ids
    )

    for j in p_ids:
        model += pulp.lpSum(x[(i, j)] for i in w_ids) >= demand_points.loc[j - 1, "base_demand"]

    for i in w_ids:
        model += pulp.lpSum(x[(i, j)] for j in p_ids) <= warehouses.loc[i - 1, "capacity"]

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    results = []

    for i in w_ids:
        for j in p_ids:
            val = x[(i, j)].varValue or 0
            if val > 0:
                results.append({
                    "warehouse": i,
                    "point": j,
                    "flow": val
                })

    df = pd.DataFrame(results)
    total_cost = pulp.value(model.objective)

    return df, total_cost


def solve_transport_sensitivity(warehouses, demand_points, cost_matrix, increase=0.2):

    demand_points_mod = demand_points.copy()
    demand_points_mod["base_demand"] *= (1 + increase)

    model = pulp.LpProblem("transport_sensitivity", pulp.LpMinimize)

    w_ids = warehouses["warehouse_id"].tolist()
    p_ids = demand_points_mod["point_id"].tolist()

    x = pulp.LpVariable.dicts(
        "ship",
        ((i, j) for i in w_ids for j in p_ids),
        lowBound=0,
        cat="Continuous"
    )

    model += pulp.lpSum(
        cost_matrix.loc[f"W{i}", f"P{j}"] * x[(i, j)]
        for i in w_ids for j in p_ids
    )

    for j in p_ids:
        model += pulp.lpSum(x[(i, j)] for i in w_ids) >= demand_points_mod.loc[j - 1, "base_demand"]

    for i in w_ids:
        model += pulp.lpSum(x[(i, j)] for j in p_ids) <= warehouses.loc[i - 1, "capacity"]

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    results = []

    for i in w_ids:
        for j in p_ids:
            val = x[(i, j)].varValue or 0
            if val > 0:
                results.append({
                    "warehouse": i,
                    "point": j,
                    "flow": val
                })

    df = pd.DataFrame(results)
    total_cost = pulp.value(model.objective)

    return df, total_cost