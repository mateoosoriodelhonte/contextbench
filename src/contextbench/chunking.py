"""Text normalization and deterministic chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import ChunkingConfig, ChunkStrategy


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    ordinal: int
    start_char: int
    end_char: int
    token_count: int
    heading: str | None = None
    metadata: dict[str, object] | None = None


def normalize_text(text: str) -> str:
    """Normalize line endings, remove control characters, and collapse excess whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _char_offsets(text: str, pieces: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        start = text.find(piece, cursor)
        if start < 0:
            start = cursor
        end = start + len(piece)
        offsets.append((start, end))
        cursor = end
    return offsets


def chunk_text(text: str, config: ChunkingConfig) -> list[Chunk]:
    text = normalize_text(text)
    if not text:
        return []
    if config.strategy is ChunkStrategy.PARAGRAPH:
        pieces = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        offsets = _char_offsets(text, pieces)
        return [
            Chunk(p, i, s, e, len(p.split()))
            for i, (p, (s, e)) in enumerate(zip(pieces, offsets, strict=True))
        ]
    if config.strategy is ChunkStrategy.HEADING:
        sections: list[tuple[str | None, str]] = []
        heading: str | None = None
        body: list[str] = []
        for line in text.splitlines():
            if re.match(r"^#{1,6}\s+", line):
                if body:
                    sections.append((heading, "\n".join(body).strip()))
                    body = []
                heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            else:
                body.append(line)
        if body:
            sections.append((heading, "\n".join(body).strip()))
        pieces = [body for _, body in sections if body]
        offsets = _char_offsets(text, pieces)
        return [
            Chunk(p, i, s, e, len(p.split()), sections[i][0], {"heading": sections[i][0]})
            for i, (p, (s, e)) in enumerate(zip(pieces, offsets, strict=True))
        ]
    words = text.split()
    size = config.chunk_size
    step = size - config.overlap
    chunks: list[Chunk] = []
    for ordinal, start_word in enumerate(range(0, len(words), step)):
        piece_words = words[start_word : start_word + size]
        if not piece_words:
            break
        piece = " ".join(piece_words)
        start = text.find(piece, 0 if not chunks else chunks[-1].start_char)
        if start < 0:
            start = 0 if not chunks else chunks[-1].end_char
        chunks.append(Chunk(piece, ordinal, start, start + len(piece), len(piece_words)))
        if start_word + size >= len(words):
            break
    return chunks
