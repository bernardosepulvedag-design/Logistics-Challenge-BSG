import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier


# =====================================================
# 2A - TIME SERIES DATA
# =====================================================

def generate_time_series(n_points=52, seed=42):
    np.random.seed(seed)

    t = np.arange(n_points)

    demand = (
        100 +
        2 * t +
        15 * np.sin(2 * np.pi * t / 12) +
        np.random.normal(0, 5, n_points)
    )

    return pd.DataFrame({
        "week": t,
        "demand": demand
    })


def create_features(df, lags=3):
    df_feat = df.copy()

    for i in range(1, lags + 1):
        df_feat[f"lag_{i}"] = df_feat["demand"].shift(i)

    return df_feat.dropna()


# =====================================================
# 2A - WALK FORWARD EVALUATION
# =====================================================

def walk_forward_validation(df_feat):

    X = df_feat.drop(columns=["demand"])
    y = df_feat["demand"]

    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "RF": RandomForestRegressor(n_estimators=120, random_state=42),
        "XGB": XGBRegressor(
            n_estimators=120,
            learning_rate=0.1,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        maes, rmses, mapes = [], [], []

        for train_idx, test_idx in tscv.split(X):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            maes.append(mean_absolute_error(y_test, preds))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
            mapes.append(np.mean(np.abs((y_test - preds) / y_test)))

        results[name] = {
            "MAE": np.mean(maes),
            "RMSE": np.mean(rmses),
            "MAPE": np.mean(mapes),
    }

    return results

# =====================================================
# 2B - RISK DATA (SIN PESOS MANUALES)
# =====================================================

def generate_risk_dataset(n=200, seed=42):

    np.random.seed(seed)

    df = pd.DataFrame({
        "stock": np.random.randint(20, 200, n),
        "lead_time": np.random.randint(1, 10, n),
        "demand": np.random.randint(30, 300, n),
        "distance": np.random.randint(10, 500, n)
    })

    # ==========================================
    # BUSINESS-BASED RISK GENERATION
    # ==========================================

    coverage_ratio = (
        df["stock"] /
        (df["demand"] * df["lead_time"] * 0.15)
    )

    risk_score = (
        3.0 * (1 / coverage_ratio)
        + 0.015 * df["distance"]
        + np.random.normal(0, 1.5, n)
    )

    df["risk"] = (
        risk_score >
        np.percentile(risk_score, 75)
    ).astype(int)

    return df


# =====================================================
# 2B - CLASSIFICATION MODELS
# =====================================================

def train_risk_models(df):

    X = df.drop(columns=["risk"])
    y = df["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    models = {
        "RF": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

        "XGB": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "F1": f1_score(y_test, preds),
            "AUC": roc_auc_score(y_test, probs),
            "CM": confusion_matrix(y_test, preds),
            "model": model
        }

    return results

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def get_feature_importance(model, features):

    return pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)


# =====================================================
# PRODUCTION MODELS  (used by the agent)
# =====================================================

def build_forecast_model(demand_history: pd.DataFrame, seed: int = 42) -> RandomForestRegressor:
    """
    Trains the final Random Forest forecast model on the full per-point
    demand history from data_generator.  Walk-forward validation (above)
    confirmed RF is the best model; this function produces the production
    version trained on all available data.

    Parameters
    ----------
    demand_history : DataFrame with columns [point_id, week, demand]
    seed           : random seed for reproducibility

    Returns
    -------
    Fitted RandomForestRegressor ready for multi-step forecasting.
    """
    rows = []
    for pid in demand_history["point_id"].unique():
        series = (
            demand_history[demand_history["point_id"] == pid]
            .sort_values("week")["demand"]
            .values
        )
        for i in range(3, len(series)):
            rows.append({
                "lag_1": series[i - 1],
                "lag_2": series[i - 2],
                "lag_3": series[i - 3],
                "target": series[i],
            })

    df = pd.DataFrame(rows)
    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    model.fit(df[["lag_1", "lag_2", "lag_3"]], df["target"])
    return model


def build_risk_classifier(seed: int = 42):
    """
    Trains and returns the production XGBoost risk classifier using the
    same data and hyperparameters validated in Module 2B.

    Returns
    -------
    Fitted XGBClassifier ready for risk probability inference.
    """
    df_risk = generate_risk_dataset(seed=seed)
    results = train_risk_models(df_risk)
    return results["XGB"]["model"]