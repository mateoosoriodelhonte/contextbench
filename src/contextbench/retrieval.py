"""Vector, BM25, hybrid, and optional reranked retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .bm25 import BM25, reciprocal_rank_fusion
from .embedding import EmbeddingProviderProtocol
from .models import ChunkRecord, Document
from .schemas import RetrievalFilters, RetrievalMethod
from .vector_store import LocalVectorStore


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    native_score: float
    rank: int
    method: RetrievalMethod
    rrf_score: float | None = None
    cross_encoder_score: float | None = None
    reranked_rank: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    rankings: dict[str, list[RetrievedChunk]]
    latencies_ms: dict[str, float]
    context: str


def build_context(chunks: list[RetrievedChunk], max_tokens: int) -> str:
    """Keep ranking order and whole chunks; trim only at deterministic token boundaries."""
    selected: list[str] = []
    used = 0
    for chunk in chunks:
        words = chunk.text.split()
        if used + len(words) > max_tokens:
            remaining = max_tokens - used
            if remaining > 0:
                selected.append(" ".join(words[:remaining]))
            break
        selected.append(chunk.text)
        used += len(words)
    return "\n\n".join(selected)


class RetrievalEngine:
    def __init__(
        self,
        store: LocalVectorStore,
        embedder: EmbeddingProviderProtocol,
        *,
        reranker: Any | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    @staticmethod
    def _allowed(document: Document, filters: RetrievalFilters) -> bool:
        if filters.document_ids and document.id not in {
            str(value) for value in filters.document_ids
        }:
            return False
        if filters.source_types and document.source_type not in {
            value.value for value in filters.source_types
        }:
            return False
        return not filters.tags or all(tag in (document.tags or []) for tag in filters.tags)

    def index(self, index_id: str, chunks: list[tuple[Document, ChunkRecord]]) -> int:
        self.store.create_collection(index_id, self.embedder.dimension)
        vectors = self.embedder.embed([chunk.text for _, chunk in chunks])
        payloads = [
            {
                "chunk_id": chunk.id,
                "document_id": doc.id,
                "source_type": doc.source_type,
                "tags": doc.tags or [],
                "text": chunk.text,
            }
            for doc, chunk in chunks
        ]
        self.store.upsert(index_id, vectors, payloads, [chunk.id for _, chunk in chunks])
        return len(chunks)

    def retrieve(
        self,
        index_id: str,
        query: str,
        chunks: list[tuple[Document, ChunkRecord]],
        *,
        methods: list[RetrievalMethod],
        top_k: int = 5,
        candidate_k: int = 20,
        filters: RetrievalFilters | None = None,
        max_context_tokens: int = 1200,
    ) -> RetrievalResult:
        filters = filters or RetrievalFilters()
        allowed = [(doc, chunk) for doc, chunk in chunks if self._allowed(doc, filters)]
        by_id = {chunk.id: (doc, chunk) for doc, chunk in allowed}
        rankings: dict[str, list[RetrievedChunk]] = {}
        latencies: dict[str, float] = {}
        vector_rank_ids: list[str] = []
        bm25_rank_ids: list[str] = []
        if (
            RetrievalMethod.VECTOR in methods
            or RetrievalMethod.HYBRID in methods
            or RetrievalMethod.RERANKED in methods
        ):
            started = time.perf_counter()
            vector = self.embedder.embed([query])[0]
            results = (
                self.store.search(
                    index_id, vector, candidate_k, document_ids=[doc.id for doc, _ in allowed]
                )
                if allowed
                else []
            )
            vector_rank_ids = [identifier for identifier, _, _ in results if identifier in by_id]
            rankings[RetrievalMethod.VECTOR.value] = [
                self._item(identifier, score, by_id, RetrievalMethod.VECTOR, i)
                for i, (identifier, score, _) in enumerate(results, 1)
                if identifier in by_id
            ][:top_k]
            latencies["vector"] = (time.perf_counter() - started) * 1000
        if RetrievalMethod.BM25 in methods or RetrievalMethod.HYBRID in methods:
            started = time.perf_counter()
            bm25 = BM25((chunk.id, chunk.text) for _, chunk in allowed)
            pairs = bm25.search(query, candidate_k)
            bm25_rank_ids = [identifier for identifier, _ in pairs]
            rankings[RetrievalMethod.BM25.value] = [
                self._item(identifier, score, by_id, RetrievalMethod.BM25, i)
                for i, (identifier, score) in enumerate(pairs, 1)
            ][:top_k]
            latencies["bm25"] = (time.perf_counter() - started) * 1000
        if RetrievalMethod.HYBRID in methods or RetrievalMethod.RERANKED in methods:
            started = time.perf_counter()
            fused = reciprocal_rank_fusion(
                {"vector": vector_rank_ids, "bm25": bm25_rank_ids}, limit=candidate_k
            )
            items = [
                self._item(identifier, score, by_id, RetrievalMethod.HYBRID, i, rrf_score=score)
                for i, (identifier, score) in enumerate(fused, 1)
            ]
            rankings[RetrievalMethod.HYBRID.value] = items[:top_k]
            latencies["hybrid"] = (time.perf_counter() - started) * 1000
        if RetrievalMethod.RERANKED in methods:
            started = time.perf_counter()
            candidates = rankings.get(RetrievalMethod.HYBRID.value, [])[:candidate_k]
            if self.reranker is not None and candidates:
                scores = self.reranker.predict([(query, item.text) for item in candidates])
                reranked = sorted(
                    zip(candidates, scores, strict=True),
                    key=lambda pair: (-float(pair[1]), pair[0].chunk_id),
                )
                rankings[RetrievalMethod.RERANKED.value] = [
                    RetrievedChunk(
                        item.chunk_id,
                        item.document_id,
                        item.text,
                        item.native_score,
                        item.rank,
                        RetrievalMethod.RERANKED,
                        item.rrf_score,
                        float(score),
                        i,
                    )
                    for i, (item, score) in enumerate(reranked[:top_k], 1)
                ]
            else:
                rankings[RetrievalMethod.RERANKED.value] = candidates[:top_k]
            latencies["reranked"] = (time.perf_counter() - started) * 1000
        context_method = (
            RetrievalMethod.RERANKED.value
            if RetrievalMethod.RERANKED.value in rankings
            else next(iter(rankings), RetrievalMethod.BM25.value)
        )
        return RetrievalResult(
            rankings, latencies, build_context(rankings.get(context_method, []), max_context_tokens)
        )

    @staticmethod
    def _item(
        identifier: str,
        score: float,
        by_id: dict[str, tuple[Document, ChunkRecord]],
        method: RetrievalMethod,
        rank: int,
        *,
        rrf_score: float | None = None,
    ) -> RetrievedChunk:
        doc, chunk = by_id[identifier]
        return RetrievedChunk(identifier, doc.id, chunk.text, float(score), rank, method, rrf_score)
