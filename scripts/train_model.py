
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

df = pd.read_csv('data/hourly_trip_features_with_28lags.csv')
X = df.drop(columns=['trip_count'])
y = df['trip_count']

split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/lgbm_model.pkl')
