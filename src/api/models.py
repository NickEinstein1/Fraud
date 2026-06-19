"""Model registry & model card API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from src.registry.model_card import generate_model_cards
from src.registry.model_registry import ModelRegistry
from src.utils.config import load_config

router = APIRouter(prefix="/v1/models", tags=["Models"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@router.get("")
def list_models() -> dict[str, Any]:
    config = load_config()
    registry = ModelRegistry(config["paths"]["registry"])
    prod = registry.get_production()
    return {
        "versions": registry.list_versions(),
        "production": prod,
    }


@router.get("/cards")
def list_model_cards() -> dict[str, Any]:
    cards_dir = PROJECT_ROOT / "artifacts" / "model_cards"
    if not cards_dir.exists():
        raise HTTPException(status_code=404, detail="No model cards found. Run: python main.py")
    files = sorted(cards_dir.glob("MODEL_CARD_*.md"))
    return {
        "directory": str(cards_dir.relative_to(PROJECT_ROOT)),
        "cards": [{"name": f.name, "path": f"/v1/models/cards/{f.name}"} for f in files],
    }


@router.get("/cards/{filename}", response_class=PlainTextResponse)
def get_model_card(filename: str) -> str:
    if ".." in filename or not filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Invalid card filename")
    path = PROJECT_ROOT / "artifacts" / "model_cards" / filename
    if not path.exists():
        path = PROJECT_ROOT / "artifacts" / "registry" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model card not found")
    return path.read_text()


@router.post("/cards/regenerate")
def regenerate_cards() -> dict[str, Any]:
    """Regenerate model cards (local dev only — not for public deployments)."""
    import os

    if os.environ.get("FRAUD_API_ALLOW_ADMIN", "").lower() not in ("1", "true", "yes"):
        raise HTTPException(
            status_code=403,
            detail="Admin endpoints disabled. Set FRAUD_API_ALLOW_ADMIN=1 for local use.",
        )
    config = load_config()
    registry = ModelRegistry(config["paths"]["registry"])
    prod = registry.get_production()
    if not prod:
        raise HTTPException(status_code=404, detail="No production model in registry")
    import json

    summary_path = PROJECT_ROOT / "artifacts" / "run_summary.json"
    drift_alerts: list = []
    run_id = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        drift_alerts = summary.get("drift_alerts", [])
        run_id = summary.get("run_id")

    paths = generate_model_cards(
        config,
        version=prod["version"],
        metrics=prod.get("metrics", {}),
        drift_alerts=drift_alerts,
        governance_action="continue",
        run_id=run_id,
    )
    return {"regenerated": [str(p.relative_to(PROJECT_ROOT)) for p in paths]}
