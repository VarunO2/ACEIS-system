import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("xgboost_model.pkl")

# Prediction function
def predict_xgb(temp=35, occupancy=80):

    hours = np.arange(24)

    df = pd.DataFrame({

        "hour": hours,

        "day": [1]*24,

        "temperature": [temp]*24,

        "occupancy": [occupancy]*24,

        "hvac_load": [(temp - 20) * 12]*24,

        "solar_output": [50]*24,

        "tariff": [8]*24

    })

    prediction = model.predict(df)

    return prediction