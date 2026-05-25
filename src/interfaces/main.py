from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.infrastructure.ml.model_loader import model_registry
from src.interfaces.api.frontend_static import build_frontend_static_router
from src.interfaces.api.routes import frontend_router, router

app = FastAPI(
    title="Booking Cancellation Prediction API",
    version="1.0.0",
    description="API for predicting hotel booking cancellation probability",
)


@app.on_event("startup")
def startup_event():
    model_registry.load()


app.include_router(router)
app.include_router(frontend_router)

frontend_static_router = build_frontend_static_router()
if frontend_static_router is not None:
    app.include_router(frontend_static_router)
    assets_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist" / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            name="frontend-assets",
        )

