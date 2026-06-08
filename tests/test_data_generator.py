"""
Tests for Module 1 — data generation and reproducibility.
"""

import pytest
import numpy as np
from src.data_generator import generate_data


def test_correct_dimensions():
    """Data must have exactly 5 warehouses and 20 demand points."""
    warehouses, demand_points, cost_matrix, demand_history = generate_data(seed=42)
    assert len(warehouses) == 5
    assert len(demand_points) == 20
    assert cost_matrix.shape == (5, 20)


def test_reproducibility():
    """Same seed must always produce identical results."""
    _, dp1, cm1, _ = generate_data(seed=42)
    _, dp2, cm2, _ = generate_data(seed=42)
    assert dp1["base_demand"].tolist() == dp2["base_demand"].tolist()
    assert cm1.values.tolist() == cm2.values.tolist()


def test_capacity_covers_demand():
    """Total warehouse capacity must be >= total demand (feasible problem)."""
    warehouses, demand_points, _, _ = generate_data(seed=42)
    assert warehouses["capacity"].sum() >= demand_points["base_demand"].sum()


def test_demand_history_length():
    """Demand history must have 52 weeks for each of the 20 points."""
    _, _, _, demand_history = generate_data(seed=42)
    assert len(demand_history) == 52 * 20
