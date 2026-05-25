from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


def build_frontend_static_router() -> APIRouter | None:
    dist_dir = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    index_path = dist_dir / "index.html"

    if not index_path.exists():
        return None

    router = APIRouter(include_in_schema=False)

    @router.get("/")
    def frontend_index():
        return FileResponse(index_path)

    return router
