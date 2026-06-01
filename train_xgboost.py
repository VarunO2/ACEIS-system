import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("aceis_training_dataset.csv")

# Feature engineering
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.dayofweek

# Features
X = df[[
    "hour",
    "day",
    "temperature",
    "occupancy",
    "hvac_load",
    "solar_output",
    "tariff"
]]

# Target
y = df["load"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, pred)

print("XGBoost MAE:", mae)

# Save
joblib.dump(model, "xgboost_model.pkl")

print("XGBoost model saved")