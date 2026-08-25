"""Embedding providers with deterministic offline defaults."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any, Protocol


class EmbeddingProviderProtocol(Protocol):
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Stable feature-hash embeddings. Useful for local tests and demos."""

    def __init__(self, dimension: int = 64, normalize: bool = True) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.normalize = normalize

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            if self.normalize:
                norm = math.sqrt(sum(value * value for value in vector))
                if norm:
                    vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors


class ModelDownloadRequired(RuntimeError):
    code = "MODEL_DOWNLOAD_REQUIRED"


class SentenceTransformersEmbeddingProvider:
    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        normalize: bool = True,
        allow_model_download: bool = False,
    ) -> None:
        self.model_name = model
        self.normalize = normalize
        self.allow_model_download = allow_model_download
        self._model: object | None = None
        self.dimension = 0

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; install the ml extra"
            ) from exc
        kwargs: dict[str, object] = {}
        if not self.allow_model_download:
            kwargs["local_files_only"] = True
        try:
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except Exception as exc:
            if not self.allow_model_download:
                raise ModelDownloadRequired(self.model_name) from exc
            raise
        model = self._model
        self.dimension = int(model.get_sentence_embedding_dimension())  # type: ignore[union-attr]
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model: Any = self._load()
        result = model.encode(list(texts), normalize_embeddings=self.normalize)
        return [list(map(float, row)) for row in result]
