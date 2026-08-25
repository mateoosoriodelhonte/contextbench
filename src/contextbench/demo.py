"""Deterministic demo corpus entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .api import AppState, _create_demo


def setup_demo(db_path: str | Path, vector_path: str | Path) -> dict[str, Any]:
    return _create_demo(AppState(db_path, vector_path))
