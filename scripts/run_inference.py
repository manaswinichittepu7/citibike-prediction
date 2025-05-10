import pandas as pd
import joblib

# Load data
df = pd.read_csv('data/hourly_trip_features_with_28lags.csv')

# Drop non-numeric features not used by the model
df = df.drop(columns=['start_station_name'])

# Fix 'hour' column if it’s object due to bad parsing
df['hour'] = pd.to_numeric(df['hour'], errors='coerce')

# Drop any rows with NaNs caused by conversion errors
df = df.dropna()

# Separate features
X = df.drop(columns=['trip_count'])

# Load model
model = joblib.load('models/lgbm_model.pkl')

# Predict
y_pred = model.predict(X)

# Save predictions
df['predicted_trip_count'] = y_pred
df[['start_station_id', 'hour', 'predicted_trip_count']].to_csv('data/predictions.csv', index=False)
