import pandas as pd
import joblib

from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("aceis_training_dataset.csv")

# Use load column
series = df["load"]

print("Training ARIMA...")

# Train model
model = ARIMA(series, order=(5,1,2))

model_fit = model.fit()

# Save model
joblib.dump(model_fit, "arima_model.pkl")

print("ARIMA model saved successfully")