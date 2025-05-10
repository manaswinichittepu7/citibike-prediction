import pandas as pd
import joblib

# Load data
data_path = 'data/hourly_trip_features_with_28lags.csv'
df = pd.read_csv(data_path)

# Drop unused columns and ensure correct types
df = df.drop(columns=['start_station_name'], errors='ignore')
df['hour'] = pd.to_numeric(df['hour'], errors='coerce')  # convert hour to numeric
df = df.dropna()  # drop rows with any NaNs after conversion

# Separate features and load model
X = df.drop(columns=['trip_count'])
model = joblib.load('models/lgbm_model.pkl')

# Check if X is valid
if X.empty or len(X.shape) != 2:
    raise ValueError("Input feature data X is invalid. It must be non-empty and 2D.")

# Make predictions
y_pred = model.predict(X)

# Save predictions
df['predicted_trip_count'] = y_pred
df.to_csv('data/predictions.csv', index=False)
