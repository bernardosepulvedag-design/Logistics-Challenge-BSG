"""
Module 3 — Intelligent Logistics Agent (ReAct Pattern)

Loop: OBSERVE → REASON → ACT → OBSERVE → … → STOP
Stop condition: every detected high-risk demand point has been attended.

No external LLM required — the reasoner uses deterministic business logic
so the system is fully reproducible and requires no API keys.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict, List, Set
from src.data_generator import generate_data
from src.optimization import solve_transport
from src.ml_models import build_forecast_model, build_risk_classifier


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    step: int
    phase: str                   # OBSERVE | REASON | ACT | STOP
    tool: Optional[str]
    input: dict
    output: Any
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Alert:
    point_id: int
    message: str
    severity: str                # LOW | MEDIUM | HIGH | CRITICAL
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ──────────────────────────────────────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────────────────────────────────────

class LogisticsAgent:
    """
    ReAct agent that monitors warehouse inventory, forecasts demand,
    runs transport optimization, and emits proactive client alerts.

    Tools
    -----
    get_stock_status(warehouse_id)         → inventory level and fill-rate
    get_demand_forecast(point_id, weeks)   → ML forecast + risk classification
    run_optimization(scenario)             → PuLP transport solver (Module 1)
    send_alert(point_id, message, severity)→ simulated notification

    Reasoning
    ---------
    Deterministic rule-based logic: a demand point is flagged HIGH-RISK when
    the XGBoost risk model returns probability >= RISK_THRESHOLD.
    A warehouse is CRITICAL when its fill-rate drops below STOCK_ALERT_RATIO.
    """

    RISK_THRESHOLD   = 0.50   # XGB probability cutoff for high-risk
    STOCK_ALERT_RATIO = 0.15  # fill-rate below this → CRITICAL
    MAX_ITERATIONS   = 10     # safety cap on the ReAct loop

    # ──────────────────────────────────────────────────────────────────────────
    # INIT
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, seed: int = 42):
        self._seed = seed
        np.random.seed(seed)

        # Load base data (same seed as Modules 1 & 2 for consistency)
        (
            self.warehouses,
            self.demand_points,
            self.cost_matrix,
            self.demand_history,
        ) = generate_data(seed=seed)

        # Simulate current warehouse inventory: 80-95% of capacity
        self.current_stock: Dict[int, int] = {
            int(row.warehouse_id): int(row.capacity * np.random.uniform(0.80, 0.95))
            for _, row in self.warehouses.iterrows()
        }

        # Simulate on-hand stock at each demand point (in-distribution: 20-180 units).
        # Uses a secondary seed so point_stock is reproducible independently of the
        # global RNG state left by generate_data().
        rng = np.random.RandomState(seed + 1000)
        self.point_stock: Dict[int, int] = {}
        for _, row in self.demand_points.iterrows():
            pid = int(row.point_id)
            # ~35% of points receive critically low stock to produce realistic alerts
            if rng.random() < 0.35:
                self.point_stock[pid] = int(rng.randint(20, 45))
            else:
                self.point_stock[pid] = int(rng.randint(70, 180))

        # Train ML models once at startup
        self._train_models()

        # Agent internal state
        self.trace:    List[AgentStep] = []
        self.alerts:   List[Alert]     = []
        self.attended: Set[int]        = set()   # point_ids fully addressed
        self._step_counter = 0

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL TRAINING
    # ──────────────────────────────────────────────────────────────────────────

    def _train_models(self):
        """
        Loads production models built in ml_models.py (Module 2).

        - Forecast model: RF trained on full per-point demand history
          (walk-forward validation in Module 2 confirmed RF is the best model;
          here we train it on all available data for production use).
        - Risk classifier: XGBoost trained with the same data and
          hyperparameters validated in Module 2B.
        """
        self._forecast_model = build_forecast_model(
            self.demand_history, seed=self._seed
        )
        self._risk_model = build_risk_classifier(seed=self._seed)

    # ──────────────────────────────────────────────────────────────────────────
    # TOOLS  (Module 3A)
    # ──────────────────────────────────────────────────────────────────────────

    def get_stock_status(self, warehouse_id: int) -> dict:
        """Returns current inventory level and fill-rate for a warehouse."""
        capacity = int(
            self.warehouses.loc[
                self.warehouses["warehouse_id"] == warehouse_id, "capacity"
            ].values[0]
        )
        stock    = self.current_stock.get(warehouse_id, 0)
        fill_rate = round(stock / capacity, 3) if capacity > 0 else 0.0
        status   = (
            "CRITICAL" if fill_rate < self.STOCK_ALERT_RATIO
            else "LOW"  if fill_rate < 0.40
            else "OK"
        )
        return {
            "warehouse_id": warehouse_id,
            "current_stock": stock,
            "capacity": capacity,
            "fill_rate": fill_rate,
            "status": status,
        }

    def get_demand_forecast(self, point_id: int, weeks: int = 4) -> dict:
        """
        Forecasts demand for the next N weeks using the trained RF model,
        then classifies the supply-risk using the XGBoost classifier.
        """
        # ── Multi-step forecast seeded by last 3 weeks of history
        series = (
            self.demand_history[self.demand_history["point_id"] == point_id]
            .sort_values("week")["demand"]
            .values
        )
        lags = list(series[-3:])
        forecasts = []
        for _ in range(weeks):
            x = pd.DataFrame(
                [[lags[-1], lags[-2], lags[-3]]], columns=["lag_1", "lag_2", "lag_3"]
            )
            pred = float(self._forecast_model.predict(x)[0])
            forecasts.append(round(pred, 1))
            lags.append(pred)

        # ── Build feature vector for risk classification
        base_demand = float(
            self.demand_points.loc[
                self.demand_points["point_id"] == point_id, "base_demand"
            ].values[0]
        )
        # Average transport cost across all warehouses as distance proxy (cost*10 ≈ km).
        # Using the average (not minimum) gives meaningful variability across points.
        avg_cost       = float(self.cost_matrix[f"P{point_id}"].mean())
        distance_proxy = int(avg_cost * 10)
        stock_proxy    = self.point_stock.get(point_id, 100)
        lead_time      = max(1, distance_proxy // 50)

        risk_prob = float(
            self._risk_model.predict_proba(
                [[stock_proxy, lead_time, base_demand, distance_proxy]]
            )[0][1]
        )
        risk_label = "HIGH" if risk_prob >= self.RISK_THRESHOLD else "LOW"

        return {
            "point_id":         point_id,
            "forecast_weeks":   weeks,
            "forecasted_demand": forecasts,
            "total_forecast":   round(sum(forecasts), 1),
            "risk_probability": round(risk_prob, 3),
            "risk_label":       risk_label,
        }

    def run_optimization(self, scenario: str = "base") -> dict:
        """
        Runs the PuLP transport model from Module 1.
        scenario='stress' applies a 20 % demand increase.
        """
        demand_mod = self.demand_points.copy()
        if scenario == "stress":
            demand_mod["base_demand"] = (demand_mod["base_demand"] * 1.2).astype(int)

        flow_df, total_cost = solve_transport(
            self.warehouses, demand_mod, self.cost_matrix
        )
        return {
            "scenario":        scenario,
            "total_cost":      round(total_cost, 2),
            "n_active_routes": len(flow_df),
            "allocations":     flow_df.to_dict(orient="records"),
        }

    def send_alert(self, point_id: int, message: str, severity: str) -> dict:
        """Simulates dispatching a proactive notification to a demand point."""
        self.alerts.append(Alert(point_id=point_id, message=message, severity=severity))
        self.attended.add(point_id)
        return {
            "status":    "SENT",
            "point_id":  point_id,
            "severity":  severity,
            "message":   message,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL LOGGING
    # ──────────────────────────────────────────────────────────────────────────

    def _log(
        self,
        phase: str,
        tool: Optional[str],
        inp: dict,
        out: Any,
        reasoning: str,
    ):
        self._step_counter += 1
        step = AgentStep(
            step=self._step_counter,
            phase=phase,
            tool=tool,
            input=inp,
            output=out,
            reasoning=reasoning,
        )
        self.trace.append(step)
        self._print_step(step)

    @staticmethod
    def _print_step(step: AgentStep):
        bar = "-" * 60
        tool_tag = f" -> {step.tool}" if step.tool else ""
        print(f"\n{bar}")
        print(f"[Step {step.step:02d}] {step.phase}{tool_tag}")
        print(f"Reasoning : {step.reasoning}")
        if step.tool:
            print(f"Input     : {json.dumps(step.input, ensure_ascii=False)}")
            out_str = json.dumps(step.output, ensure_ascii=False, default=str)
            print(f"Output    : {out_str[:300]}{'...' if len(out_str) > 300 else ''}")
        print(f"Timestamp : {step.timestamp}")

    # ──────────────────────────────────────────────────────────────────────────
    # REACT LOOP  (Module 3B)
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Executes the ReAct loop until all high-risk demand points are attended
        or MAX_ITERATIONS is reached.

        Returns a summary dict with full results.
        """
        print("\n" + "=" * 60)
        print("   LOGISTICS AGENT - ReAct Loop")
        print("=" * 60)

        iteration         = 0
        all_high_risk: Set[int] = set()

        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            print(f"\n{'*' * 60}")
            print(f"  ITERATION {iteration}")
            print(f"{'*' * 60}")

            # ── OBSERVE: warehouse inventory ──────────────────────────────
            stock_snapshot: Dict[int, dict] = {}
            for wh_id in self.warehouses["warehouse_id"].tolist():
                result = self.get_stock_status(int(wh_id))
                stock_snapshot[int(wh_id)] = result
                self._log(
                    phase="OBSERVE",
                    tool="get_stock_status",
                    inp={"warehouse_id": int(wh_id)},
                    out=result,
                    reasoning=(
                        f"Polling inventory for warehouse W{wh_id}. "
                        f"Fill-rate = {result['fill_rate']:.1%} ({result['status']})."
                    ),
                )

            # ── OBSERVE: demand forecast & risk for all points ────────────
            forecast_snapshot: Dict[int, dict] = {}
            for pid in self.demand_points["point_id"].tolist():
                result = self.get_demand_forecast(int(pid), weeks=4)
                forecast_snapshot[int(pid)] = result
                self._log(
                    phase="OBSERVE",
                    tool="get_demand_forecast",
                    inp={"point_id": int(pid), "weeks": 4},
                    out=result,
                    reasoning=(
                        f"4-week forecast for P{pid}: "
                        f"{result['total_forecast']} units. "
                        f"Risk = {result['risk_label']} "
                        f"(p={result['risk_probability']:.1%})."
                    ),
                )

            # ── REASON ────────────────────────────────────────────────────
            newly_detected = {
                pid
                for pid, fc in forecast_snapshot.items()
                if fc["risk_label"] == "HIGH" and pid not in self.attended
            }
            all_high_risk.update(newly_detected)

            critical_warehouses = [
                wid
                for wid, s in stock_snapshot.items()
                if s["status"] == "CRITICAL"
            ]

            # Stop condition: nothing new to address
            if not newly_detected and not critical_warehouses:
                self._log(
                    phase="REASON",
                    tool=None,
                    inp={},
                    out={
                        "unattended_high_risk": 0,
                        "critical_warehouses": 0,
                    },
                    reasoning=(
                        "No new high-risk demand points and no critical warehouses. "
                        "Stop condition reached — all points in acceptable state."
                    ),
                )
                break

            self._log(
                phase="REASON",
                tool=None,
                inp={},
                out={
                    "newly_detected_high_risk": sorted(newly_detected),
                    "critical_warehouses": critical_warehouses,
                },
                reasoning=(
                    f"Detected {len(newly_detected)} new high-risk point(s) "
                    f"{sorted(newly_detected)} and "
                    f"{len(critical_warehouses)} critical warehouse(s) "
                    f"{critical_warehouses}. Triggering action phase."
                ),
            )

            # ── ACT: reoptimize if any warehouse is critical ───────────────
            if critical_warehouses:
                opt_result = self.run_optimization(scenario="stress")
                self._log(
                    phase="ACT",
                    tool="run_optimization",
                    inp={"scenario": "stress"},
                    out={
                        "total_cost":      opt_result["total_cost"],
                        "n_active_routes": opt_result["n_active_routes"],
                    },
                    reasoning=(
                        f"Warehouse(s) {critical_warehouses} are critically low. "
                        "Running stress-scenario optimization (+20 % demand) to "
                        "recompute allocation plan and identify cost impact."
                    ),
                )

            # ── ACT: send alert for each newly detected high-risk point ────
            for pid in sorted(newly_detected):
                fc       = forecast_snapshot[pid]
                severity = "CRITICAL" if fc["risk_probability"] > 0.75 else "HIGH"
                message  = (
                    f"Demand point P{pid} - projected 4-week demand: "
                    f"{fc['total_forecast']} units "
                    f"(risk probability: {fc['risk_probability']:.1%}). "
                    f"Immediate replenishment review recommended."
                )
                alert_result = self.send_alert(pid, message, severity)
                self._log(
                    phase="ACT",
                    tool="send_alert",
                    inp={"point_id": pid, "severity": severity},
                    out=alert_result,
                    reasoning=(
                        f"P{pid} classified as {severity} "
                        f"(p={fc['risk_probability']:.1%}). "
                        f"Alert dispatched to client."
                    ),
                )

            # ── OBSERVE (closing): verify stop condition ───────────────────
            unattended = all_high_risk - self.attended
            self._log(
                phase="OBSERVE",
                tool=None,
                inp={},
                out={"unattended_high_risk_points": sorted(unattended)},
                reasoning=(
                    f"{len(unattended)} high-risk point(s) still unattended: "
                    f"{sorted(unattended)}. Looping back."
                    if unattended
                    else "All high-risk points attended. Evaluating stop condition."
                ),
            )

            if not unattended:
                self._log(
                    phase="STOP",
                    tool=None,
                    inp={},
                    out={
                        "attended_points": sorted(self.attended),
                        "total_alerts":    len(self.alerts),
                    },
                    reasoning=(
                        "Stop condition satisfied: every detected high-risk demand "
                        "point has received a proactive alert and the optimization "
                        "model has been re-run where required."
                    ),
                )
                break

        # ── Build and print summary ────────────────────────────────────────
        summary = self._build_summary(iteration)
        self._print_summary(summary)
        return summary

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY & EXPORT
    # ──────────────────────────────────────────────────────────────────────────

    def _build_summary(self, iterations: int) -> dict:
        return {
            "total_steps":                self._step_counter,
            "iterations_used":            iterations,
            "high_risk_points_attended":  sorted(self.attended),
            "total_alerts_issued":        len(self.alerts),
            "alerts": [asdict(a) for a in self.alerts],
            "final_warehouse_status": {
                wid: self.get_stock_status(wid)
                for wid in self.warehouses["warehouse_id"].tolist()
            },
        }

    @staticmethod
    def _print_summary(summary: dict):
        print("\n" + "=" * 60)
        print("   AGENT EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total reasoning steps  : {summary['total_steps']}")
        print(f"ReAct iterations used  : {summary['iterations_used']}")
        print(f"High-risk points       : {len(summary['high_risk_points_attended'])} attended")
        print(f"Alerts issued          : {summary['total_alerts_issued']}")
        print("\nALERT LOG:")
        for a in summary["alerts"]:
            msg_short = a["message"][:70] + "..." if len(a["message"]) > 70 else a["message"]
            print(f"  [{a['severity']:8s}] P{a['point_id']:02d} - {msg_short}")

    def export_trace(self, path: str = "data/agent_trace.json"):
        """Serializes the full reasoning trace to JSON for auditability."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(s) for s in self.trace],
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        print(f"\nReasoning trace exported → {path}")
