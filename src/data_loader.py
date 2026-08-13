"""
data_loader.py
---------------
Load and clean the European Day-Ahead electricity price dataset.

The raw CSV has one row per hour, with a `date` column, an `hour` column
formatted as "HH:00 - HH:00", and one price column (EUR/MWh) per country.
"""

from pathlib import Path
import pandas as pd

COUNTRIES = ["france", "italy", "belgium", "spain", "uk", "germany"]

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "electricity_dah_prices.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV as-is."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps, sort chronologically, de-duplicate, and index by time.

    The `hour` column looks like '00:00 - 01:00'; we take the starting hour
    and combine it with `date` to build a proper datetime index.
    """
    df = df.copy()

    start_hour = df["hour"].str.split(" - ").str[0]
    df["timestamp"] = pd.to_datetime(df["date"] + " " + start_hour, format="%Y/%m/%d %H:%M")

    df = df.drop(columns=["date", "hour"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")

    # Reindex to a complete hourly range so any missing hours become explicit
    # NaNs rather than silent gaps, then forward-fill short gaps (<=3h).
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "timestamp"
    df[COUNTRIES] = df[COUNTRIES].ffill(limit=3)

    return df


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    return clean(load_raw(path))


if __name__ == "__main__":
    data = load_clean()
    print(data.shape)
    print(data.head())
    print(data.isna().sum())
