"""
Streamlit app: European Day-Ahead Electricity Price Forecasting

Run locally with:  streamlit run app.py
Deployed via Streamlit Community Cloud (see README for the deploy guide).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

st.set_page_config(page_title="EU Electricity Price Forecasting", layout="wide")
sns.set_theme(style="whitegrid")

COUNTRIES = ["france", "italy", "belgium", "spain", "uk", "germany"]
PALETTE = {
    "france": "#274690", "italy": "#2E933C", "belgium": "#F2A007",
    "spain": "#C43E3E", "uk": "#7A3E9D", "germany": "#1A1A1A",
}


@st.cache_data
def load_data():
    from data_loader import load_clean
    return load_clean()


@st.cache_data
def get_features(_df):
    from features import build_feature_matrix
    return build_feature_matrix(_df, target="germany", exog=("france", "belgium"))


@st.cache_resource
def train_models(_feature_matrix):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    fm = _feature_matrix
    cutoff = fm.index.max() - pd.Timedelta(weeks=8)
    train, test = fm[fm.index <= cutoff], fm[fm.index > cutoff]
    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    def evaluate(y_true, y_pred):
        return {
            "MAE": round(mean_absolute_error(y_true, y_pred), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        }

    results, preds = {}, {"actual": y_test}

    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = pd.Series(lr.predict(X_test), index=y_test.index)
    results["Linear Regression"] = evaluate(y_test, lr_pred)
    preds["Linear Regression"] = lr_pred

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = pd.Series(rf.predict(X_test), index=y_test.index)
    results["Random Forest"] = evaluate(y_test, rf_pred)
    preds["Random Forest"] = rf_pred

    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb.fit(X_train, y_train)
        xgb_pred = pd.Series(xgb.predict(X_test), index=y_test.index)
        results["XGBoost"] = evaluate(y_test, xgb_pred)
        preds["XGBoost"] = xgb_pred
    except ImportError:
        pass

    return results, preds


# ---- UI --------------------------------------------------------------

st.title("European Day-Ahead Electricity Price Forecasting")
st.caption(
    "Forecasting Germany's day-ahead electricity price 24 hours ahead, using 2022 hourly data "
    "across six European markets. [View the full project on GitHub](https://github.com/uthman399)."
)

df = load_data()

tab1, tab2, tab3 = st.tabs(["Market Overview", "Forecast Model", "About This Project"])

with tab1:
    st.subheader("2022 European Day-Ahead Prices")
    selected = st.multiselect("Markets to show", COUNTRIES, default=COUNTRIES)

    daily = df.resample("D").mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in selected:
        ax.plot(daily.index, daily[c], label=c.upper(), color=PALETTE[c], linewidth=1.4)
    ax.set_ylabel("EUR / MWh")
    ax.legend(ncol=len(selected) or 1, frameon=False)
    ax.set_title("Daily Average Price")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak daily average (any market)", f"{daily[COUNTRIES].max().max():.0f} EUR/MWh")
    with col2:
        neg_hours = (df[COUNTRIES] < 0).sum().sum()
        st.metric("Total negative-price hours (2022, all markets)", f"{neg_hours:,}")

with tab2:
    st.subheader("24-Hour-Ahead Forecast: Germany")
    with st.spinner("Training models (cached after first run)..."):
        feature_matrix = get_features(df)
        results, preds = train_models(feature_matrix)

    results_df = pd.DataFrame(results).T
    st.dataframe(results_df, width='stretch')

    model_choice = st.selectbox("Model to plot", list(preds.keys())[1:])
    weeks_to_show = st.slider("Weeks of test period to show", 2, 8, 4)

    actual = preds["actual"]
    pred = preds[model_choice]
    window = actual.index[: 24 * 7 * weeks_to_show]

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(window, actual.loc[window], label="Actual", color="black", linewidth=2)
    ax2.plot(window, pred.loc[window], label=model_choice, color="#C43E3E")
    ax2.set_ylabel("EUR / MWh")
    ax2.legend(frameon=False)
    ax2.set_title(f"Germany: Actual vs. {model_choice} Forecast")
    st.pyplot(fig2)

with tab3:
    st.markdown("""
    **Methodology**
    - Leak-free feature engineering: calendar features, autoregressive lags (24h/48h/168h),
      rolling statistics computed only on already-lagged data
    - Chronological train/test split (last 8 weeks of 2022 held out), no shuffled cross-validation
    - Naive seasonal baseline benchmarked against Linear Regression, Random Forest, and XGBoost

    **Stack:** Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn, Streamlit

    Full source, tests, and the executed analysis notebook are in the GitHub repository linked above.
    """)
