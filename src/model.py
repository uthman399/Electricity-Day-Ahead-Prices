"""
model.py
--------
Trains and evaluates models that forecast Germany's day-ahead electricity
price 24 hours in advance.

Baseline : "naive seasonal" -> tomorrow's price = same hour, last week
Model 1  : Linear Regression on engineered features
Model 2  : Random Forest Regressor
Model 3  : Gradient-boosted trees (XGBoost)

Evaluation uses a chronological (not random) train/test split, since this
is a time series -- the last 8 weeks of 2022 are held out as the test set.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from data_loader import load_clean
from features import build_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNTRY = "germany"
TEST_WEEKS = 8


def mape(y_true, y_pred, eps=20.0):
    """Mean absolute percentage error, floored to avoid blow-ups near zero.
    Day-ahead prices can sit near 0 EUR/MWh or go negative (esp. late Dec
    2022), where plain MAPE explodes -- a floor keeps it interpretable, but
    RMSE/MAE (in EUR/MWh) are the primary metrics reported for this reason."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def evaluate(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAPE_%": round(mape(y_true, y_pred), 1),
    }


def naive_seasonal_baseline(df, test_idx):
    """Predict price(t) = price(t - 168h), i.e. same hour last week."""
    lag = df[TARGET_COUNTRY].shift(168)
    return lag.loc[test_idx]


def run():
    df = load_clean()
    fm = build_feature_matrix(df, target=TARGET_COUNTRY, exog=("france", "belgium"))

    cutoff = fm.index.max() - pd.Timedelta(weeks=TEST_WEEKS)
    train, test = fm[fm.index <= cutoff], fm[fm.index > cutoff]

    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    results = {}
    predictions = {"actual": y_test}

    # --- Baseline: naive seasonal (same hour, last week) ---
    baseline_pred = naive_seasonal_baseline(df, y_test.index)
    baseline_pred = baseline_pred.reindex(y_test.index)
    valid = baseline_pred.notna()
    results["Naive Seasonal (t-168h)"] = evaluate(y_test[valid], baseline_pred[valid])
    predictions["baseline"] = baseline_pred

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = pd.Series(lr.predict(X_test), index=y_test.index)
    results["Linear Regression"] = evaluate(y_test, lr_pred)
    predictions["linear_regression"] = lr_pred

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = pd.Series(rf.predict(X_test), index=y_test.index)
    results["Random Forest"] = evaluate(y_test, rf_pred)
    predictions["random_forest"] = rf_pred

    # --- XGBoost ---
    xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = pd.Series(xgb.predict(X_test), index=y_test.index)
    results["XGBoost"] = evaluate(y_test, xgb_pred)
    predictions["xgboost"] = xgb_pred

    # --- Save results ---
    results_df = pd.DataFrame(results).T
    results_df.index.name = "model"
    results_df.to_csv(ROOT / "outputs" / "model_results.csv")
    print(results_df)

    joblib.dump(xgb, MODEL_DIR / "xgboost_germany_24h.joblib")
    joblib.dump(list(X_train.columns), MODEL_DIR / "feature_columns.joblib")

    with open(ROOT / "outputs" / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    _plot_predictions(predictions)
    _plot_feature_importance(xgb, X_train.columns)
    _plot_model_comparison(results_df)

    return results_df, predictions


def _plot_predictions(predictions):
    fig, ax = plt.subplots(figsize=(16, 7))
    window = predictions["actual"].index[:24 * 14]  # first 2 weeks of the test period
    ax.plot(window, predictions["actual"].loc[window], label="Actual", color="black", linewidth=2)
    ax.plot(window, predictions["baseline"].loc[window], label="Naive Seasonal", linestyle="--", color="grey")
    ax.plot(window, predictions["xgboost"].loc[window], label="XGBoost", color="#C43E3E")
    ax.set_title("Germany Day-Ahead Price: 24h-Ahead Forecast vs. Actual (Test Period, first 2 weeks)")
    ax.set_ylabel("EUR / MWh")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_forecast_vs_actual.png", dpi=150)
    plt.close(fig)


def _plot_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    importances.plot(kind="barh", ax=ax, color="#274690")
    ax.set_title("XGBoost Feature Importance (Germany 24h-Ahead Forecast)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_feature_importance.png", dpi=150)
    plt.close(fig)


def _plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df["RMSE"].sort_values().plot(kind="barh", ax=ax, color="#2E933C")
    ax.set_title("Model Comparison: RMSE on Held-Out Test Period (EUR/MWh)")
    ax.set_xlabel("RMSE (EUR/MWh, lower is better)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
