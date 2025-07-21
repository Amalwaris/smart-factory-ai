from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from app.model_utils import load_models
import joblib

app = FastAPI()

clf, anomaly_model, label_encoder = load_models()

feature_columns = [
    'temperature',
    'vibration',
    'power_usage',
    'production_speed',
    'noise_level'
]

class SensorData(BaseModel):
    temperature: float
    vibration: float
    power_usage: float
    production_speed: float
    noise_level: float

@app.post("/predict")
def predict(data: SensorData):
    input_data = np.array([[getattr(data, col) for col in feature_columns]])

    class_pred = clf.predict(input_data)[0]
    class_label = label_encoder.inverse_transform([class_pred])[0]

    anomaly_pred = anomaly_model.predict(input_data)[0]
    anomaly_status = "Anomaly" if anomaly_pred == -1 else "Normal"

    return {
        "maintenance_prediction": class_label,
        "anomaly_status": anomaly_status
    }
