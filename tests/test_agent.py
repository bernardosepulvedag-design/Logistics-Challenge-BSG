"""
Tests for Module 3 — intelligent logistics agent.
"""

import warnings
import pytest
warnings.filterwarnings("ignore")

from src.agent import LogisticsAgent


@pytest.fixture(scope="module")
def agent():
    return LogisticsAgent(seed=42)


def test_agent_detects_high_risk_points(agent):
    """With seed=42 the agent must identify at least one high-risk demand point."""
    summary = agent.run()
    assert len(summary["high_risk_points_attended"]) >= 1


def test_agent_issues_alerts(agent):
    """Every detected high-risk point must receive exactly one alert."""
    summary = agent.run()
    attended = summary["high_risk_points_attended"]
    alerted = [a["point_id"] for a in summary["alerts"]]
    for pid in attended:
        assert pid in alerted, f"Point {pid} was attended but received no alert"


def test_stock_status_returns_valid_fields(agent):
    """get_stock_status must return fill_rate between 0 and 1 and a known status."""
    result = agent.get_stock_status(1)
    assert 0.0 <= result["fill_rate"] <= 1.0
    assert result["status"] in ("OK", "LOW", "CRITICAL")


def test_demand_forecast_returns_correct_weeks(agent):
    """get_demand_forecast must return exactly the requested number of forecast values."""
    result = agent.get_demand_forecast(point_id=1, weeks=4)
    assert len(result["forecasted_demand"]) == 4
    assert result["risk_label"] in ("LOW", "HIGH")


def test_optimization_returns_positive_cost(agent):
    """run_optimization must return a cost greater than zero for both scenarios."""
    base = agent.run_optimization(scenario="base")
    stress = agent.run_optimization(scenario="stress")
    assert base["total_cost"] > 0
    assert stress["total_cost"] > base["total_cost"]
