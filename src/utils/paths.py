"""Safe path resolution for API file access (project-root jail)."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str, *, must_exist: bool = True) -> Path:
    """
    Resolve a user-supplied path relative to the project root.
    Rejects path traversal outside the repository.
    """
    if not path or not str(path).strip():
        raise HTTPException(status_code=400, detail="Path is required")
    candidate = (PROJECT_ROOT / path).resolve()
    root = PROJECT_ROOT.resolve()
    if not str(candidate).startswith(str(root)):
        raise HTTPException(status_code=400, detail="Path must stay inside the project directory")
    if must_exist and not candidate.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return candidate


def resolve_csv_path(path: str) -> Path:
    candidate = resolve_project_path(path)
    if candidate.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    return candidate
