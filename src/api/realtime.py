"""Real-time AI routes — WebSocket stream + status (L2 Experience)."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.realtime.ai_engine import RealtimeFraudAIEngine
from src.utils.paths import resolve_csv_path

router = APIRouter(prefix="/v1/realtime", tags=["Real-Time AI"])

_engine: RealtimeFraudAIEngine | None = None


def get_realtime_engine() -> RealtimeFraudAIEngine:
    global _engine
    if _engine is None:
        _engine = RealtimeFraudAIEngine()
        _engine.load()
    return _engine


class StreamControl(BaseModel):
    max_batches: int | None = Field(default=100, ge=1, le=10_000)
    source_path: str | None = None


@router.get("/status")
def realtime_status() -> dict[str, Any]:
    return get_realtime_engine().status()


@router.get("/history")
def realtime_history(n: int = 50) -> dict[str, Any]:
    n = min(max(n, 1), 500)
    return {"snapshots": get_realtime_engine().history(n)}


@router.post("/reset")
def realtime_reset() -> dict[str, str]:
    import os

    if os.environ.get("FRAUD_API_ALLOW_ADMIN", "").lower() not in ("1", "true", "yes"):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403,
            detail="Admin endpoints disabled. Set FRAUD_API_ALLOW_ADMIN=1 for local use.",
        )
    global _engine
    _engine = None
    return {"status": "engine_reset"}


@router.websocket("/ws")
async def realtime_websocket(websocket: WebSocket) -> None:
    """
    Stream RealtimeSnapshot JSON ticks.
    Client may send: {"max_batches": 200} or {"action": "stop"}
    """
    await websocket.accept()
    engine = get_realtime_engine()
    max_batches = 500
    source_path = None

    try:
        init = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
        max_batches = min(int(init.get("max_batches", max_batches)), 10_000)
        raw_path = init.get("source_path")
        if raw_path:
            source_path = str(resolve_csv_path(str(raw_path)))
    except (asyncio.TimeoutError, WebSocketDisconnect):
        pass
    except Exception:
        pass

    try:
        count = 0
        async for snap in engine.stream_async(source_path=source_path, max_batches=max_batches):
            await websocket.send_json(snap.to_dict())
            count += 1
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                if msg.get("action") == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            if snap.blocked:
                await websocket.send_json(
                    {"event": "governance_block", "action": snap.governance_action}
                )
                break
    except WebSocketDisconnect:
        return
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
