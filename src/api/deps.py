"""Shared API dependencies."""

from __future__ import annotations

from src.serving.runtime import FraudServingRuntime

_runtime: FraudServingRuntime | None = None


def get_runtime() -> FraudServingRuntime:
    global _runtime
    if _runtime is None:
        _runtime = FraudServingRuntime()
        _runtime.load()
    return _runtime
