"""
eda.py
------
Generates the exploratory data analysis figures used in the writeup /
notebook, saved to outputs/figures/.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

from data_loader import load_clean, COUNTRIES

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {
    "france": "#274690", "italy": "#2E933C", "belgium": "#F2A007",
    "spain": "#C43E3E", "uk": "#7A3E9D", "germany": "#1A1A1A",
}


def plot_full_year(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    daily = df.resample("D").mean()
    for c in COUNTRIES:
        ax.plot(daily.index, daily[c], label=c.upper(), color=PALETTE[c], linewidth=1.4)
    ax.set_title("2022 European Day-Ahead Electricity Prices (Daily Average)")
    ax.set_ylabel("EUR / MWh")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(ncol=6, loc="upper left", frameon=False)
    ax.axvspan(pd.Timestamp("2022-02-24"), pd.Timestamp("2022-02-25"), color="grey", alpha=0.3)
    ax.text(pd.Timestamp("2022-03-01"), ax.get_ylim()[1]*0.92, "Russia invades Ukraine", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_full_year_trend.png", dpi=150)
    plt.close(fig)


def plot_monthly_boxplot(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    for ax, c in zip(axes.flat, COUNTRIES):
        monthly = df[[c]].copy()
        monthly["month"] = monthly.index.month
        sns.boxplot(data=monthly, x="month", y=c, ax=ax, color=PALETTE[c], fliersize=1.5)
        ax.set_title(c.upper())
        ax.set_xlabel("")
        ax.set_ylabel("EUR/MWh" if ax in axes[:, 0] else "")
    fig.suptitle("Monthly Price Distribution by Country (2022)", y=1.02, fontsize=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_monthly_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hourly_profile(df):
    fig, ax = plt.subplots(figsize=(12, 7))
    hourly = df.copy()
    hourly["hour"] = hourly.index.hour
    profile = hourly.groupby("hour")[COUNTRIES].mean()
    for c in COUNTRIES:
        ax.plot(profile.index, profile[c], marker="o", markersize=3, label=c.upper(), color=PALETTE[c])
    ax.set_title("Average Intraday Price Profile (2022)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("EUR / MWh")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_hourly_profile.png", dpi=150)
    plt.close(fig)


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = df[COUNTRIES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=0, vmax=1, ax=ax,
                square=True, cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Cross-Country Price Correlation (hourly, 2022)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_negative_prices(df):
    neg_counts = (df[COUNTRIES] < 0).sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(neg_counts.index.str.upper(), neg_counts.values,
                   color=[PALETTE[c] for c in neg_counts.index])
    for b, v in zip(bars, neg_counts.values):
        ax.text(b.get_x() + b.get_width()/2, v + 1, str(v), ha="center", fontsize=12)
    ax.set_title("Hours with Negative Day-Ahead Prices (2022)")
    ax.set_ylabel("Number of hours")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_negative_price_hours.png", dpi=150)
    plt.close(fig)
    return neg_counts


def plot_volatility(df):
    weekly_std = df[COUNTRIES].resample("W").std()
    fig, ax = plt.subplots(figsize=(16, 7))
    for c in COUNTRIES:
        ax.plot(weekly_std.index, weekly_std[c], label=c.upper(), color=PALETTE[c], linewidth=1.4)
    ax.set_title("Weekly Price Volatility (Std. Dev. of Hourly Prices)")
    ax.set_ylabel("EUR / MWh (std. dev.)")
    ax.legend(ncol=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_weekly_volatility.png", dpi=150)
    plt.close(fig)


def summary_stats(df):
    return df[COUNTRIES].describe().T


def run_all():
    df = load_clean()
    plot_full_year(df)
    plot_monthly_boxplot(df)
    plot_hourly_profile(df)
    plot_correlation(df)
    neg_counts = plot_negative_prices(df)
    plot_volatility(df)
    stats = summary_stats(df)
    print("Summary stats:\n", stats)
    print("\nNegative price hour counts:\n", neg_counts)
    print(f"\nFigures written to {FIG_DIR}")
    return df, stats, neg_counts


if __name__ == "__main__":
    run_all()
