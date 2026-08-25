"""Small, honest local benchmark for the deterministic offline pipeline."""

from __future__ import annotations

import platform
import statistics
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .embedding import HashEmbeddingProvider
from .models import ChunkRecord, Document
from .retrieval import RetrievalEngine
from .schemas import RetrievalMethod
from .vector_store import LocalVectorStore


def run_benchmark(corpus_size: int = 200, query_count: int = 20) -> dict[str, Any]:
    """Measure wall-clock timings. Results describe only the current machine and run."""
    if corpus_size < 10 or query_count < 1:
        raise ValueError("corpus_size must be at least 10 and query_count must be positive")
    index_id = str(uuid.uuid4())
    document = Document(
        id=str(uuid.uuid4()),
        project_id=str(uuid.uuid4()),
        filename="synthetic-benchmark.txt",
        source_type="TXT",
        content="",
        sha256="0" * 64,
        tags=["benchmark"],
    )
    chunks = [
        ChunkRecord(
            id=str(uuid.uuid4()),
            document_id=document.id,
            index_configuration_id=index_id,
            ordinal=index,
            text=(
                f"Synthetic record {index} describes raft log replication, database recovery, "
                f"and tcp flow control with marker term-{index}."
            ),
            token_count=16,
            start_char=index * 100,
            end_char=index * 100 + 90,
            metadata_json={},
        )
        for index in range(corpus_size)
    ]
    pairs = [(document, chunk) for chunk in chunks]
    embedder = HashEmbeddingProvider()
    embed_started = time.perf_counter()
    embedder.embed([chunk.text for chunk in chunks])
    embedding_seconds = time.perf_counter() - embed_started
    with tempfile.TemporaryDirectory(prefix="contextbench-benchmark-") as directory:
        engine = RetrievalEngine(LocalVectorStore(Path(directory) / "qdrant"), embedder)
        indexing_started = time.perf_counter()
        engine.index(index_id, pairs)
        indexing_ms = (time.perf_counter() - indexing_started) * 1000
        timings: dict[str, list[float]] = {"vector": [], "bm25": [], "hybrid": []}
        for index in range(query_count):
            query = f"raft recovery term-{index % corpus_size}"
            for label, methods in (
                ("vector", [RetrievalMethod.VECTOR]),
                ("bm25", [RetrievalMethod.BM25]),
                ("hybrid", [RetrievalMethod.HYBRID]),
            ):
                started = time.perf_counter()
                engine.retrieve(index_id, query, pairs, methods=methods, top_k=5, candidate_k=20)
                timings[label].append((time.perf_counter() - started) * 1000)
        engine.store.close()
    return {
        "schema": "contextbench.benchmark.v1",
        "measuredAt": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "configuration": {
            "corpusSize": corpus_size,
            "queryCount": query_count,
            "embedding": "contextbench-hash-v1",
            "dimension": embedder.dimension,
        },
        "measurements": {
            "embeddingDocumentsPerSecond": corpus_size / embedding_seconds,
            "indexingMs": indexing_ms,
            "vectorQueryMedianMs": statistics.median(timings["vector"]),
            "bm25QueryMedianMs": statistics.median(timings["bm25"]),
            "hybridQueryMedianMs": statistics.median(timings["hybrid"]),
            "rerank": {
                "status": "NOT_RUN",
                "reason": "No model download in the default benchmark.",
            },
        },
    }
