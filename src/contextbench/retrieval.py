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
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    cross_encoder_score: float | None = None
    candidate_rank: int | None = None
    reranked_rank: int | None = None
    document_name: str = "unknown"
    ordinal: int = 0
    page: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    rankings: dict[str, list[RetrievedChunk]]
    latencies_ms: dict[str, float]
    context: str
    context_method: RetrievalMethod


def build_context(chunks: list[RetrievedChunk], max_tokens: int) -> str:
    """Keep ranking order and quote untrusted text under generated citation headers."""
    selected: list[str] = []
    used = 0
    for citation, chunk in enumerate(chunks, 1):
        safe_document_name = " ".join(chunk.document_name.split()) or "unknown"
        location = f" · page {chunk.page}" if chunk.page else ""
        header = f"[{citation}] {safe_document_name} · chunk {chunk.ordinal}{location}"
        header_tokens = len(header.split())
        text_words = chunk.text.split()
        quote_tokens = 1
        remaining = max_tokens - used
        if remaining <= header_tokens + quote_tokens:
            break
        if header_tokens + quote_tokens + len(text_words) > remaining:
            keep = remaining - header_tokens - quote_tokens
            selected.append(f"{header}\n> {' '.join(text_words[:keep])}")
            break
        selected.append(f"{header}\n> {' '.join(text_words)}")
        used += header_tokens + quote_tokens + len(text_words)
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
        vectors = self.embedder.embed_documents([chunk.text for _, chunk in chunks])
        self.store.create_collection(index_id, self.embedder.dimension)
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
        total_started = time.perf_counter()
        filters = filters or RetrievalFilters()
        allowed = [(doc, chunk) for doc, chunk in chunks if self._allowed(doc, filters)]
        by_id = {chunk.id: (doc, chunk) for doc, chunk in allowed}
        rankings: dict[str, list[RetrievedChunk]] = {}
        latencies: dict[str, float] = {}
        vector_rank_ids: list[str] = []
        bm25_rank_ids: list[str] = []
        needs_vector = (
            RetrievalMethod.VECTOR in methods
            or RetrievalMethod.HYBRID in methods
            or RetrievalMethod.RERANKED in methods
        )
        needs_bm25 = (
            RetrievalMethod.BM25 in methods
            or RetrievalMethod.HYBRID in methods
            or RetrievalMethod.RERANKED in methods
        )
        vector_items: list[RetrievedChunk] = []
        bm25_items: list[RetrievedChunk] = []
        hybrid_candidates: list[RetrievedChunk] = []
        if needs_vector:
            started = time.perf_counter()
            vector = self.embedder.embed_query(query)
            results = (
                self.store.search(
                    index_id, vector, candidate_k, document_ids=[doc.id for doc, _ in allowed]
                )
                if allowed
                else []
            )
            vector_rank_ids = [identifier for identifier, _, _ in results if identifier in by_id]
            vector_items = [
                self._item(
                    identifier,
                    score,
                    by_id,
                    RetrievalMethod.VECTOR,
                    i,
                    vector_score=score,
                )
                for i, (identifier, score, _) in enumerate(results, 1)
                if identifier in by_id
            ]
            if RetrievalMethod.VECTOR in methods:
                rankings[RetrievalMethod.VECTOR.value] = vector_items[:top_k]
            latencies["vector"] = (time.perf_counter() - started) * 1000
        if needs_bm25:
            started = time.perf_counter()
            bm25 = BM25((chunk.id, chunk.text) for _, chunk in allowed)
            pairs = bm25.search(query, candidate_k)
            bm25_rank_ids = [identifier for identifier, _ in pairs]
            bm25_items = [
                self._item(
                    identifier,
                    score,
                    by_id,
                    RetrievalMethod.BM25,
                    i,
                    bm25_score=score,
                )
                for i, (identifier, score) in enumerate(pairs, 1)
            ]
            if RetrievalMethod.BM25 in methods:
                rankings[RetrievalMethod.BM25.value] = bm25_items[:top_k]
            latencies["bm25"] = (time.perf_counter() - started) * 1000
        if RetrievalMethod.HYBRID in methods or RetrievalMethod.RERANKED in methods:
            started = time.perf_counter()
            fused = reciprocal_rank_fusion(
                {"vector": vector_rank_ids, "bm25": bm25_rank_ids}, limit=candidate_k
            )
            vector_scores = {item.chunk_id: item.native_score for item in vector_items}
            bm25_scores = {item.chunk_id: item.native_score for item in bm25_items}
            hybrid_candidates = [
                self._item(
                    identifier,
                    score,
                    by_id,
                    RetrievalMethod.HYBRID,
                    i,
                    vector_score=vector_scores.get(identifier),
                    bm25_score=bm25_scores.get(identifier),
                    rrf_score=score,
                )
                for i, (identifier, score) in enumerate(fused, 1)
            ]
            if RetrievalMethod.HYBRID in methods:
                rankings[RetrievalMethod.HYBRID.value] = hybrid_candidates[:top_k]
            latencies["hybrid"] = (time.perf_counter() - started) * 1000
        if RetrievalMethod.RERANKED in methods:
            started = time.perf_counter()
            candidates = hybrid_candidates[:candidate_k]
            if self.reranker is None:
                raise ValueError("reranked retrieval requires a reranker")
            if candidates:
                scores = self.reranker.predict([(query, item.text) for item in candidates])
                reranked = sorted(
                    zip(candidates, scores, strict=True),
                    key=lambda pair: (-float(pair[1]), pair[0].chunk_id),
                )
                rankings[RetrievalMethod.RERANKED.value] = [
                    RetrievedChunk(
                        chunk_id=item.chunk_id,
                        document_id=item.document_id,
                        text=item.text,
                        native_score=float(score),
                        rank=i,
                        method=RetrievalMethod.RERANKED,
                        vector_score=item.vector_score,
                        bm25_score=item.bm25_score,
                        rrf_score=item.rrf_score,
                        cross_encoder_score=float(score),
                        candidate_rank=item.rank,
                        reranked_rank=i,
                        document_name=item.document_name,
                        ordinal=item.ordinal,
                        page=item.page,
                    )
                    for i, (item, score) in enumerate(reranked[:top_k], 1)
                ]
            else:
                rankings[RetrievalMethod.RERANKED.value] = []
            latencies["reranked"] = (time.perf_counter() - started) * 1000
        context_method = next(
            method
            for method in (
                RetrievalMethod.RERANKED,
                RetrievalMethod.HYBRID,
                RetrievalMethod.VECTOR,
                RetrievalMethod.BM25,
            )
            if method.value in rankings
        )
        context_started = time.perf_counter()
        context = build_context(rankings.get(context_method.value, []), max_context_tokens)
        latencies["context"] = (time.perf_counter() - context_started) * 1000
        latencies["total"] = (time.perf_counter() - total_started) * 1000
        return RetrievalResult(rankings, latencies, context, context_method)

    @staticmethod
    def _item(
        identifier: str,
        score: float,
        by_id: dict[str, tuple[Document, ChunkRecord]],
        method: RetrievalMethod,
        rank: int,
        *,
        vector_score: float | None = None,
        bm25_score: float | None = None,
        rrf_score: float | None = None,
    ) -> RetrievedChunk:
        doc, chunk = by_id[identifier]
        return RetrievedChunk(
            chunk_id=identifier,
            document_id=doc.id,
            text=chunk.text,
            native_score=float(score),
            rank=rank,
            method=method,
            vector_score=vector_score,
            bm25_score=bm25_score,
            rrf_score=rrf_score,
            document_name=doc.filename,
            ordinal=chunk.ordinal,
            page=chunk.metadata_json.get("page"),
        )
