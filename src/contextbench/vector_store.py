"""Persisted Qdrant local vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


class LocalVectorStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=self.path)

    @staticmethod
    def collection_name(index_id: str) -> str:
        return "index_" + index_id.replace("-", "")

    def create_collection(self, index_id: str, dimension: int) -> None:
        name = self.collection_name(index_id)
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    def upsert(
        self,
        index_id: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        if not (len(vectors) == len(payloads) == len(ids)):
            raise ValueError("vectors, payloads, and ids must have the same length")
        self.client.upsert(
            self.collection_name(index_id),
            points=[
                models.PointStruct(id=point_id, vector=vector, payload=payload)
                for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
            ],
        )

    def search(
        self,
        index_id: str,
        vector: list[float],
        limit: int = 10,
        *,
        document_ids: list[str] | None = None,
        source_types: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        must: list[models.FieldCondition] = []
        if document_ids:
            must.append(
                models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))
            )
        if source_types:
            must.append(
                models.FieldCondition(key="source_type", match=models.MatchAny(any=source_types))
            )
        if tags:
            for tag in tags:
                must.append(models.FieldCondition(key="tags", match=models.MatchValue(value=tag)))
        query_filter = models.Filter(must=must) if must else None
        result = self.client.query_points(
            self.collection_name(index_id),
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
        return [(str(point.id), float(point.score), dict(point.payload or {})) for point in result]
