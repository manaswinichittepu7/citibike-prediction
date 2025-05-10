import pandas as pd
import joblib
import os

# Ensure the models directory and file exist
model_path = "models/lgbm_model.pkl"
data_path = "data/hourly_trip_features_with_28lags.csv"
output_path = "data/predictions.csv"

# Check for model and data
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Feature data not found at: {data_path}")

# Load model
model = joblib.load(model_path)

# Load feature data
df = pd.read_csv(data_path)

# Split features and target
if "trip_count" not in df.columns:
    raise ValueError("'trip_count' column missing in input data.")

X = df.drop(columns=["trip_count"])
y_true = df["trip_count"]

# Predict
y_pred = model.predict(X)

# Save predictions
predictions_df = pd.DataFrame({
    "actual": y_true,
    "predicted": y_pred
})

# Save to CSV
os.makedirs("data", exist_ok=True)
predictions_df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to {output_path}")
