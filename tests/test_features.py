"""
Basic sanity tests for data loading and feature engineering.
Run with:  pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_clean, COUNTRIES  # noqa: E402
from features import build_feature_matrix, add_calendar_features  # noqa: E402


def test_load_clean_has_expected_columns():
    df = load_clean()
    assert list(df.columns) == COUNTRIES


def test_load_clean_hourly_index_no_gaps():
    df = load_clean()
    inferred = pd.infer_freq(df.index)
    assert inferred in ("h", "H")


def test_calendar_features_ranges():
    df = load_clean()
    out = add_calendar_features(df)
    assert out["hour"].between(0, 23).all()
    assert out["dayofweek"].between(0, 6).all()
    assert set(out["is_weekend"].unique()) <= {0, 1}


def test_feature_matrix_no_nans():
    df = load_clean()
    fm = build_feature_matrix(df)
    assert fm.isna().sum().sum() == 0


def test_feature_matrix_target_is_shifted_price():
    df = load_clean()
    fm = build_feature_matrix(df, target="germany")
    # target at time t should equal germany's actual price 24h later
    sample_ts = fm.index[100]
    expected = df.loc[sample_ts + pd.Timedelta(hours=24), "germany"]
    assert abs(fm.loc[sample_ts, "target"] - expected) < 1e-9
