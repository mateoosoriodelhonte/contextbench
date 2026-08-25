"""Safe local document ingestion."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .chunking import normalize_text
from .schemas import SourceType

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 1000
MAX_EXTRACTED_CHARACTERS = 20_000_000
ALLOWED_SUFFIXES = {".txt": SourceType.TXT, ".md": SourceType.MD, ".markdown": SourceType.MD}
SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".sql",
    ".json",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    filename: str
    source_type: SourceType
    text: str
    sha256: str
    metadata: dict[str, Any]


class IngestionError(ValueError):
    pass


def _decode_utf8(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise IngestionError("text document is not valid UTF-8") from exc


def ingest_bytes(
    filename: str, data: bytes, *, max_bytes: int = MAX_DOCUMENT_BYTES
) -> IngestedDocument:
    if (
        not filename
        or Path(filename).name != filename
        or any(ord(character) < 32 or ord(character) == 127 for character in filename)
    ):
        raise IngestionError("filename must be a plain file name")
    if len(data) > max_bytes:
        raise IngestionError("document exceeds the maximum size")
    suffix = Path(filename).suffix.lower()
    page_offsets: list[dict[str, int]] = []
    if suffix in ALLOWED_SUFFIXES:
        source_type = ALLOWED_SUFFIXES[suffix]
        text = _decode_utf8(data)
    elif suffix == ".pdf":
        source_type = SourceType.PDF
        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
            if len(reader.pages) > MAX_PDF_PAGES:
                raise IngestionError("PDF exceeds the maximum page count")
            extracted_pages: list[str] = []
            extracted_characters = 0
            for page in reader.pages:
                page_text = normalize_text(page.extract_text() or "")
                extracted_characters += len(page_text)
                if extracted_characters > MAX_EXTRACTED_CHARACTERS:
                    raise IngestionError("PDF extracted text exceeds the maximum size")
                extracted_pages.append(page_text)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError("invalid PDF document") from exc
        pieces: list[str] = []
        cursor = 0
        for page_number, page_text in enumerate(extracted_pages, 1):
            if not page_text:
                continue
            if pieces:
                cursor += 2
            start = cursor
            pieces.append(page_text)
            cursor += len(page_text)
            page_offsets.append({"page": page_number, "startChar": start, "endChar": cursor})
        text = "\n\n".join(pieces)
    elif suffix in SOURCE_SUFFIXES:
        source_type = SourceType.SOURCE
        text = _decode_utf8(data)
    else:
        raise IngestionError("unsupported document type")
    text = normalize_text(text)
    if not text:
        raise IngestionError("document has no extractable text")
    metadata = {"pages": page_offsets} if source_type is SourceType.PDF else {}
    return IngestedDocument(filename, source_type, text, hashlib.sha256(data).hexdigest(), metadata)


def ingest_path(path: str | Path, *, allowed_root: str | Path | None = None) -> IngestedDocument:
    candidate = Path(path).resolve()
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        if candidate != root and root not in candidate.parents:
            raise IngestionError("path is outside the allowed root")
    if not candidate.is_file():
        raise IngestionError("document path is not a file")
    return ingest_bytes(candidate.name, candidate.read_bytes())
