from fastapi import FastAPI

from app.api.routes import router
from app.core.model_loader import model_registry

app = FastAPI(
    title="Booking Cancellation Prediction API",
    version="1.0.0",
    description="API for predicting hotel booking cancellation probability",
)


@app.on_event("startup")
def startup_event():
    model_registry.load()


app.include_router(router)