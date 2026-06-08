"""
Tests for Module 1 — transport optimization solver.
"""

import pytest
from src.data_generator import generate_data
from src.optimization import solve_transport, solve_transport_sensitivity


@pytest.fixture
def base_data():
    return generate_data(seed=42)


def test_solution_has_positive_cost(base_data):
    """Optimal cost must be greater than zero."""
    warehouses, demand_points, cost_matrix, _ = base_data
    _, total_cost = solve_transport(warehouses, demand_points, cost_matrix)
    assert total_cost > 0


def test_demand_fully_satisfied(base_data):
    """Every demand point must receive at least its required units."""
    warehouses, demand_points, cost_matrix, _ = base_data
    flow_df, _ = solve_transport(warehouses, demand_points, cost_matrix)

    for _, row in demand_points.iterrows():
        pid = int(row["point_id"])
        supplied = flow_df[flow_df["point"] == pid]["flow"].sum()
        assert supplied >= row["base_demand"] - 0.01, (
            f"Point {pid} undersupplied: got {supplied:.1f}, needed {row['base_demand']}"
        )


def test_stress_cost_higher_than_base(base_data):
    """Distributing 20% more demand must cost more than the base scenario."""
    warehouses, demand_points, cost_matrix, _ = base_data
    _, base_cost = solve_transport(warehouses, demand_points, cost_matrix)
    _, stress_cost = solve_transport_sensitivity(
        warehouses, demand_points, cost_matrix, increase=0.2
    )
    assert stress_cost > base_cost
