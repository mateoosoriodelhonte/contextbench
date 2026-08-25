"""Local storage defaults shared by the API and CLI."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIRECTORY = Path(os.environ.get("CONTEXTBENCH_DATA_DIR", ".contextbench")).expanduser()
DEFAULT_DATABASE_PATH = DATA_DIRECTORY / "contextbench.sqlite3"
DEFAULT_VECTOR_PATH = DATA_DIRECTORY / "qdrant"
