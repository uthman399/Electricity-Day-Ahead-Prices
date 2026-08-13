"""
features.py
------------
Feature engineering for day-ahead electricity price forecasting.

Given a cleaned, hourly-indexed price DataFrame (see data_loader.py), builds
a supervised-learning feature matrix for forecasting a single target
country's price using calendar features, lags, and rolling statistics
(optionally enriched with other countries' prices as exogenous signals).
"""

import pandas as pd

CALENDAR_FEATURES = ["hour", "dayofweek", "month", "is_weekend"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str, lags=(1, 24, 168)) -> pd.DataFrame:
    """Add lagged values of the target column (1h, 24h/1-day, 168h/1-week back)."""
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str, windows=(24, 168)) -> pd.DataFrame:
    """Add rolling mean/std of the target, computed on lagged data only
    (shift(1) first) so no future information leaks into the features."""
    df = df.copy()
    shifted = df[target].shift(1)
    for w in windows:
        df[f"{target}_rollmean{w}"] = shifted.rolling(w).mean()
        df[f"{target}_rollstd{w}"] = shifted.rolling(w).std()
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target: str = "germany",
    exog: tuple = ("france", "belgium"),
) -> pd.DataFrame:
    """
    Build the full feature matrix + target for forecasting `target`
    24 hours ahead. `exog` countries are included at their 24h-lagged
    value only (the most recent value we'd actually have at forecast time).
    """
    out = add_calendar_features(df)
    out = add_lag_features(out, target, lags=(24, 25, 48, 168))
    out = add_rolling_features(out, target, windows=(24, 168))

    for c in exog:
        out[f"{c}_lag24"] = df[c].shift(24)

    out["target"] = df[target].shift(-24)  # what we're predicting: price 24h ahead
    out = out.drop(columns=list(df.columns))  # drop all raw price columns; keep engineered features only

    return out.dropna()
