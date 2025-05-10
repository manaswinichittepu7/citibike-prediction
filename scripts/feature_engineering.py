# scripts/feature_engineering.py

import pandas as pd
import os

# Step 1: Load hourly trip demand data (replace with correct CSV path if needed)
df = pd.read_csv("data/hourly_trip_demand.csv", parse_dates=["hour"])
df = df.sort_values(by=["start_station_name", "hour"])

# Step 2: Create lag features
for lag in range(1, 29):
    df[f"lag_{lag}"] = df.groupby("start_station_name")["trip_count"].shift(lag)

# Step 3: Drop rows with missing values
df = df.dropna().reset_index(drop=True)

# Step 4: Save to CSV
os.makedirs("data", exist_ok=True)
df.to_csv("data/hourly_trip_features_with_28lags.csv", index=False)
print("✅ Saved to data/hourly_trip_features_with_28lags.csv")
