"""Embedding providers with deterministic offline defaults."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any, Protocol


class EmbeddingProviderProtocol(Protocol):
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Stable feature-hash embeddings. Useful for local tests and demos."""

    def __init__(self, dimension: int = 64, normalize: bool = True) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.normalize = normalize

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
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

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class ModelDownloadRequired(RuntimeError):
    code = "MODEL_DOWNLOAD_REQUIRED"


class ModelRuntimeUnavailable(RuntimeError):
    code = "MODEL_RUNTIME_UNAVAILABLE"


class SentenceTransformersEmbeddingProvider:
    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        revision: str | None = None,
        normalize: bool = True,
        allow_model_download: bool = False,
    ) -> None:
        self.model_name = model
        self.revision = revision
        self.normalize = normalize
        self.allow_model_download = allow_model_download
        self._model: object | None = None
        self.dimension = 0
        self.resolved_revision: str | None = None

    @staticmethod
    def _commit_hash(model: Any) -> str | None:
        candidates = [model, getattr(model, "model", None)]
        try:
            first_module = model[0]
        except (KeyError, TypeError, IndexError):
            first_module = None
        candidates.extend([first_module, getattr(first_module, "auto_model", None)])
        for candidate in candidates:
            config = getattr(candidate, "config", None)
            commit = getattr(config, "_commit_hash", None)
            if isinstance(commit, str) and commit:
                return commit
        return None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ModelRuntimeUnavailable(
                "sentence-transformers is not installed; install the ml extra"
            ) from exc
        kwargs: dict[str, object] = {}
        if self.revision:
            kwargs["revision"] = self.revision
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
        self.resolved_revision = self._commit_hash(model)
        if self.resolved_revision is None:
            self._model = None
            raise ModelRuntimeUnavailable("the embedding model revision could not be resolved")
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model: Any = self._load()
        encoder = getattr(model, "encode_document", None) or model.encode
        result = encoder(list(texts), normalize_embeddings=self.normalize)
        return [list(map(float, row)) for row in result]

    def embed_query(self, text: str) -> list[float]:
        model: Any = self._load()
        encoder = getattr(model, "encode_query", None) or model.encode
        result = encoder([text], normalize_embeddings=self.normalize)
        return list(map(float, result[0]))


class CrossEncoderReranker:
    """Lazy local cross-encoder. Model downloads require explicit consent."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        *,
        revision: str | None = None,
        allow_model_download: bool = False,
    ) -> None:
        self.model_name = model
        self.revision = revision
        self.allow_model_download = allow_model_download
        self._model: object | None = None
        self.resolved_revision: str | None = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ModelRuntimeUnavailable(
                "sentence-transformers is not installed; install the ml extra"
            ) from exc
        try:
            self._model = CrossEncoder(
                self.model_name,
                revision=self.revision,
                local_files_only=not self.allow_model_download,
            )
        except Exception as exc:
            if not self.allow_model_download:
                raise ModelDownloadRequired(self.model_name) from exc
            raise
        model = self._model
        self.resolved_revision = SentenceTransformersEmbeddingProvider._commit_hash(model)
        if self.resolved_revision is None:
            self._model = None
            raise ModelRuntimeUnavailable("the reranker model revision could not be resolved")
        return model

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        model: Any = self._load()
        scores = model.predict(list(pairs))
        return [float(score) for score in scores]
