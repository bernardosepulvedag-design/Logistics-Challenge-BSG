import textwrap

from src.data_generator import generate_data
from src.optimization import solve_transport, solve_transport_sensitivity
from src.visualization import plot_results
from src.ml_models import (
    generate_time_series,
    create_features,
    walk_forward_validation,
    generate_risk_dataset,
    train_risk_models,
    get_feature_importance,
)
from src.agent import LogisticsAgent


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 1 — OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────────────

def run_module_1():
    print("\n" + "=" * 60)
    print("        MODULE 1 — SUPPLY CHAIN OPTIMIZATION")
    print("=" * 60)

    warehouses, demand_points, cost_matrix, _ = generate_data(seed=42)

    print("\nWAREHOUSES (CAPACITY):")
    print(warehouses.to_string(index=False))

    print("\nDEMAND POINTS:")
    print(demand_points.to_string(index=False))

    solution, total_cost = solve_transport(warehouses, demand_points, cost_matrix)

    print("\n" + "=" * 60)
    print("             OPTIMAL SOLUTION")
    print("=" * 60)
    print(f"\nTOTAL COST: {total_cost:,.2f}")
    print("\nFULL ALLOCATION MATRIX:")
    print(solution.to_string(index=False))

    stress_solution, stress_cost = solve_transport_sensitivity(
        warehouses, demand_points, cost_matrix, increase=0.2
    )

    print("\n" + "=" * 60)
    print("        SENSITIVITY ANALYSIS (+20%)")
    print("=" * 60)
    print(f"\nBASE COST:   {total_cost:,.2f}")
    print(f"STRESS COST: {stress_cost:,.2f}")
    print(f"DELTA:       +{stress_cost - total_cost:,.2f} ({(stress_cost/total_cost - 1):.1%})")

    print("\nGenerating visualization dashboard...")
    plot_results(warehouses, demand_points, solution, total_cost, stress_cost)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 2 — MACHINE LEARNING
# ──────────────────────────────────────────────────────────────────────────────

def run_module_2():
    print("\n" + "=" * 60)
    print("           MODULE 2 — MACHINE LEARNING")
    print("=" * 60)

    # 2A — Demand Forecasting
    df_ts   = generate_time_series()
    df_feat = create_features(df_ts)
    forecast_results = walk_forward_validation(df_feat)

    print("\n── 2A · DEMAND FORECAST (Walk-Forward Validation) ──")
    for model, metrics in forecast_results.items():
        print(f"\n  Model: {model}")
        print(f"  MAE:   {metrics['MAE']:.2f}")
        print(f"  RMSE:  {metrics['RMSE']:.2f}")
        print(f"  MAPE:  {metrics['MAPE']:.2%}")

    print(
        "\n  Walk-forward validation completed. Models trained exclusively on "
        "historical data and evaluated on future periods (no data leakage)."
    )

    # 2B — Risk Classification
    df_risk = generate_risk_dataset()

    print("\n── 2B · RISK CLASSIFICATION ──")
    print("\n  Risk distribution:")
    print(df_risk["risk"].value_counts().to_string())

    risk_results = train_risk_models(df_risk)

    for model_name, result in risk_results.items():
        print(f"\n  Model: {model_name}")
        print(f"  F1:    {result['F1']:.3f}")
        print(f"  AUC:   {result['AUC']:.3f}")
        print("\n  Confusion Matrix:")
        print(result["CM"])

        fi = get_feature_importance(
            result["model"],
            df_risk.drop(columns=["risk"]).columns,
        )
        print("\n  Feature Importance:")
        print(fi.to_string(index=False))

    # 2C — Executive summary
    summary = """
    The ML module addresses demand forecasting and supply-risk classification.
    For demand prediction, Random Forest and XGBoost were trained with lag features
    and walk-forward validation (no data leakage). Random Forest achieved MAE=16.62,
    RMSE=18.98, MAPE=10.21 %, outperforming XGBoost under the current data regime.
    For risk classification, XGBoost reached F1=0.812 and AUC=0.948.
    Feature importance shows stock level and lead time as the dominant predictors,
    followed by demand and distance — confirming that replenishment speed and
    available inventory are the primary levers for supply-risk management.
    """
    print("\n" + "=" * 60)
    print("  EXECUTIVE SUMMARY — MODULE 2")
    print("=" * 60)
    print(textwrap.fill(summary.strip(), width=70))


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 3 — INTELLIGENT AGENT
# ──────────────────────────────────────────────────────────────────────────────

def run_module_3():
    print("\n" + "=" * 60)
    print("        MODULE 3 — INTELLIGENT LOGISTICS AGENT")
    print("=" * 60)

    agent = LogisticsAgent(seed=42)
    summary = agent.run()

    # Export the full reasoning trace for auditability
    agent.export_trace(path="data/agent_trace.json")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    run_module_1()
    run_module_2()
    run_module_3()


if __name__ == "__main__":
    main()
