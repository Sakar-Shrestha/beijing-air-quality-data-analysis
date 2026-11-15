"""Combine the 12 Beijing station CSVs and clean them.


Made by using the initial ipynb file process. 

Run by  python preprocess.py
Outputs (in Data/processed/):
    beijing_air_clean.parquet   cleaned dataset used by the dashboard
    missing_report.csv          missing values before vs after (evidence for the report)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "Data"
OUT_DIR = os.path.join(DATA_DIR, "processed")

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC = POLLUTANTS + WEATHER
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
# grouping by location of the monitoring site
SUBURBAN = {"Changping", "Dingling", "Huairou", "Shunyi"}


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "PRSA_Data_*.csv")))
    print(f"{len(files)} files found")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns="No").sort_values(["station", "datetime"]).reset_index(drop=True)
    print("combined:", df.shape, "| duplicate station+time:", df.duplicated(["station", "datetime"]).sum())

    before = df[NUMERIC + ["wd"]].isna().sum()

    # Flags, computed on raw values before any filling
    df["imputed_any"] = df[NUMERIC + ["wd"]].isna().any(axis=1)
    df["pm_inconsistent"] = df["PM2.5"] > df["PM10"]
    df["extreme_pm25"] = df["PM2.5"] > 500

    # 1) short gaps (<= 6 h): linear interpolation inside each station
    df[NUMERIC] = df.groupby("station")[NUMERIC].transform(
        lambda s: s.interpolate(limit=6, limit_area="inside"))
    # 2) remaining gaps: median for that station / month / hour
    grp = df.groupby(["station", "month", "hour"])
    for c in NUMERIC:
        df[c] = df[c].fillna(grp[c].transform("median"))
    # 3) wind direction: ffill short gaps, then station-month mode
    df["wd"] = df.groupby("station")["wd"].transform(lambda s: s.ffill(limit=6))
    mode_wd = df.groupby(["station", "month"])["wd"].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    df["wd"] = df["wd"].fillna(mode_wd)

    after = df[NUMERIC + ["wd"]].isna().sum()

    # 4) derived columns
    df["date"] = df["datetime"].dt.normalize()
    df["season"] = df["month"].map(SEASONS)
    df["weekday"] = df["datetime"].dt.day_name()
    df["wd_deg"] = df["wd"].map({d: i * 22.5 for i, d in enumerate(COMPASS)})
    df["site_type"] = np.where(df["station"].isin(SUBURBAN), "Suburban/rural", "Urban")
    # China's PM2.5 grades (HJ 633-2012) are defined on 24-hour averages, so grade
    # the rolling 24 h mean rather than the single hourly reading.
    df["pm25_24h"] = df.groupby("station")["PM2.5"].transform(
        lambda s: s.rolling(24, min_periods=18).mean())
    df["pm25_category"] = pd.cut(
        df["pm25_24h"], [0, 35, 75, 115, 150, 250, np.inf],
        labels=["Excellent", "Good", "Light", "Moderate", "Heavy", "Severe"])

    os.makedirs(OUT_DIR, exist_ok=True)
    report = pd.DataFrame({"before": before, "after": after})
    report["before_pct"] = (report["before"] / len(df) * 100).round(2)
    report.to_csv(os.path.join(OUT_DIR, "missing_report.csv"), index_label="column")
    df.to_parquet(os.path.join(OUT_DIR, "beijing_air_clean.parquet"), index=False)
    print(report)
    print("saved ->", OUT_DIR, "| rows:", len(df))


if __name__ == "__main__":
    main()