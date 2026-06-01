import joblib

# Load ARIMA model
model = joblib.load("arima_model.pkl")

# Prediction function
def predict_arima(steps=24):

    forecast = model.forecast(steps=steps)

    return forecast