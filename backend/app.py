from fastapi import FastAPI
from backend.routes import predict

app = FastAPI(
    title="Aircraft Predictive Maintenance API",
    version="1.0",
    description="Returns RUL and Anomaly Score from sensor input"
)

# Include routes
app.include_router(predict.router)
