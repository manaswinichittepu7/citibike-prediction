import pandas as pd
import joblib
import os

# Load feature data
data_path = 'data/hourly_trip_features_with_28lags.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Feature data not found at: {data_path}")

df = pd.read_csv(data_path)

# Drop any non-numeric or irrelevant columns
if 'trip_count' not in df.columns:
    raise ValueError("Expected column 'trip_count' missing in the data.")

X = df.drop(columns=['trip_count'])

# OPTIONAL: drop columns like station names if still present
X = X.select_dtypes(include=['int64', 'float64'])  # keep only numeric

# Debug
print(f"Shape of X: {X.shape}")
print(f"Columns in X: {X.columns.tolist()}")

# Final check
if X.empty or len(X.shape) != 2:
    raise ValueError("Input feature data X is invalid. It must be non-empty and 2D.")

# Load model
model = joblib.load('models/lgbm_model.pkl')

# Predict
y_pred = model.predict(X)

# Save predictions
df['predicted_trip_count'] = y_pred
df.to_csv('data/predictions.csv', index=False)
print("✅ Inference completed and predictions saved.")
