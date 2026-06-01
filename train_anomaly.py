import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("aceis_training_dataset.csv")

features = [
    "temperature",
    "occupancy",
    "hvac_load",
    "solar_output",
    "tariff",
    "load"
]

X = df[features]

# Scale
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(X_scaled)

joblib.dump(model, "anomaly_model.pkl")
joblib.dump(scaler, "anomaly_scaler.pkl")

print("Anomaly model saved")