"""FastAPI application — Experience layer (REST + static UI + WebSocket)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.data_routes import router as data_router
from src.api.models import router as models_router
from src.api.realtime import router as realtime_router
from src.api.scoring import router as scoring_router
from src.api.system import router as system_router
from src.api.deps import get_runtime
from src.architecture.layers import AIWorldStack

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(
    title="Fraud Detection AI Platform",
    description="Score transactions manually, from file path, or upload — plus real-time AI",
    version="2.1.0",
)
app.include_router(scoring_router)
app.include_router(data_router)
app.include_router(realtime_router)
app.include_router(models_router)
app.include_router(system_router)

if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/")
def root() -> RedirectResponse:
    if FRONTEND_DIR.is_dir():
        return RedirectResponse(url="/ui/index.html")
    return RedirectResponse(url="/docs")


@app.get("/architecture")
def architecture() -> dict[str, Any]:
    return {
        "description": AIWorldStack.describe(),
        "layers": [
            {"id": s.layer.value, "name": s.name, "responsibility": s.responsibility}
            for s in AIWorldStack.layers()
        ],
        "scoring": {
            "manual_json": "POST /v1/score",
            "file_path": "POST /v1/score/from-path",
            "file_upload": "POST /v1/score/from-upload",
            "ui": "/ui/index.html",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return get_runtime().health().to_dict()
