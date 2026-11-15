"""
Data preprocessing script for Beijing Multi-Site Air Quality dataset.
Combines 12 monitoring stations, handles missing values, and creates derived features.

Run: python Preprocess.py
Outputs:
  - Data/processed/beijing_air_clean.parquet
  - Data/processed/missing_report.csv
"""

import numpy as np
import pandas as pd

CSV_FILES = [
    "Data/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "Data/PRSA_Data_Changping_20130301-20170228.csv",
    "Data/PRSA_Data_Dingling_20130301-20170228.csv",
    "Data/PRSA_Data_Dongsi_20130301-20170228.csv",
    "Data/PRSA_Data_Guanyuan_20130301-20170228.csv",
    "Data/PRSA_Data_Gucheng_20130301-20170228.csv",
    "Data/PRSA_Data_Huairou_20130301-20170228.csv",
    "Data/PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    "Data/PRSA_Data_Shunyi_20130301-20170228.csv",
    "Data/PRSA_Data_Tiantan_20130301-20170228.csv",
    "Data/PRSA_Data_Wanliu_20130301-20170228.csv",
    "Data/PRSA_Data_Wanshouxigong_20130301-20170228.csv",
]

CLEAN_DATA_PATH = "Data/processed/beijing_air_clean.parquet"
MISSING_REPORT_PATH = "Data/processed/missing_report.csv"

# Columns to process
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC = POLLUTANTS + WEATHER

# Compass directions for 16 wind points (each point is 22.5 degrees)
COMPASS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

# Month-to-season mapping
SEASONS = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

# Environmental station classifications (Beijing Municipal Environmental Monitoring Center)
SUBURBAN = {"Changping", "Huairou", "Shunyi"}
BACKGROUND = {"Dingling"}


def classify_site(station):
    if station in BACKGROUND:
        return "Regional Background"
    elif station in SUBURBAN:
        return "Suburban"
    return "Urban"


def main():
    print(f"Reading {len(CSV_FILES)} station CSV files...")
    df_list = []
    for filepath in CSV_FILES:
        station_df = pd.read_csv(filepath)
        df_list.append(station_df)

    df = pd.concat(df_list, ignore_index=True)

    # Combine year, month, day, hour into a single datetime column
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns="No").sort_values(["station", "datetime"]).reset_index(drop=True)
    print("Combined dataset shape:", df.shape)

    # Count missing values before cleaning
    before = df[NUMERIC + ["wd"]].isna().sum()

    # Pre-imputation flags (computed before filling any values)
    df["imputed_any"] = df[NUMERIC + ["wd"]].isna().any(axis=1)
    df["pm_inconsistent"] = df["PM2.5"] > df["PM10"]
    df["extreme_pm25"] = df["PM2.5"] > 500

    # Step 1: Linear interpolation for short gaps (up to 6 hours) per station
    df[NUMERIC] = df.groupby("station")[NUMERIC].transform(
        lambda s: s.interpolate(limit=6, limit_area="inside")
    )

    # Step 2: Fill remaining gaps with the station's median for that specific month and hour
    grouped = df.groupby(["station", "month", "hour"])
    for col in NUMERIC:
        df[col] = df[col].fillna(grouped[col].transform("median"))

    # Step 3: Impute wind direction (forward-fill up to 6 hours, then station-month mode)
    df["wd"] = df.groupby("station")["wd"].transform(lambda s: s.ffill(limit=6))
    mode_wd = df.groupby(["station", "month"])["wd"].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan
    )
    df["wd"] = df["wd"].fillna(mode_wd)

    # Count missing values after cleaning (should be 0)
    after = df[NUMERIC + ["wd"]].isna().sum()

    # Step 4: Derived columns
    df["date"] = df["datetime"].dt.normalize()
    df["season"] = df["month"].map(SEASONS)
    df["weekday"] = df["datetime"].dt.day_name()
    df["wd_deg"] = df["wd"].map({d: i * 22.5 for i, d in enumerate(COMPASS)})
    df["site_type"] = df["station"].map(classify_site)

    # Rolling 24-hour PM2.5 mean and China HJ 633-2012 air quality grades
    df["pm25_24h"] = df.groupby("station")["PM2.5"].transform(
        lambda s: s.rolling(24, min_periods=18).mean()
    )
    df["pm25_category"] = pd.cut(
        df["pm25_24h"],
        bins=[0, 35, 75, 115, 150, 250, np.inf],
        labels=["Excellent", "Good", "Light", "Moderate", "Heavy", "Severe"],
    )

    # Save missing report and cleaned parquet file
    report = pd.DataFrame({"before": before, "after": after})
    report["before_pct"] = (report["before"] / len(df) * 100).round(2)
    report.to_csv(MISSING_REPORT_PATH, index_label="column")
    df.to_parquet(CLEAN_DATA_PATH, index=False)

    print("\nMissing values report:")
    print(report)
    print(f"\nCleaned dataset saved directly to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
