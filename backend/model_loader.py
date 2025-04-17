from fastapi import APIRouter
from backend.schemas import SensorData
from ml_model.inference import predict_rul, detect_anomaly

router = APIRouter()

@router.post("/predict")
def predict(sensor_data: SensorData):
    rul = predict_rul(sensor_data.data)
    anomaly_score = detect_anomaly(sensor_data.data)
    return {
        "predicted_RUL": round(rul, 2),
        "anomaly_score": round(anomaly_score, 6)
    }
