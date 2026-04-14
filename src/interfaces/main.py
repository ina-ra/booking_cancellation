from fastapi import FastAPI

from src.infrastructure.ml.model_loader import model_registry
from src.interfaces.api.routes import router

app = FastAPI(
    title="Booking Cancellation Prediction API",
    version="1.0.0",
    description="API for predicting hotel booking cancellation probability",
)


@app.on_event("startup")
def startup_event():
    model_registry.load()


app.include_router(router)

