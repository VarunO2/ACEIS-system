import joblib
import pandas as pd

model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("anomaly_scaler.pkl")


def detect_anomaly(data):

    df = pd.DataFrame([data])

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)

    score = model.decision_function(scaled)

    return prediction[0], score[0]