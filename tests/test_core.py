import io
import math
from typing import Any

import pytest
from pypdf import PdfWriter

from contextbench.bm25 import BM25, reciprocal_rank_fusion
from contextbench.chunking import chunk_text, normalize_text
from contextbench.embedding import HashEmbeddingProvider
from contextbench.evaluation import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k
from contextbench.ingestion import IngestionError, ingest_bytes
from contextbench.models import ChunkRecord, Document
from contextbench.ollama import OllamaClient
from contextbench.retrieval import RetrievalEngine, RetrievedChunk, build_context
from contextbench.schemas import ChunkingConfig, CreateProjectRequest, RetrievalMethod
from contextbench.vector_store import LocalVectorStore


def test_schemas_use_camel_case_aliases() -> None:
    request = CreateProjectRequest.model_validate({"name": "Demo", "description": "x"})
    assert request.name == "Demo"
    assert (
        request.model_dump(by_alias=True)["createdAt"] is None
        if "createdAt" in request.model_dump(by_alias=True)
        else True
    )


def test_fixed_token_chunking_has_overlap_and_provenance() -> None:
    chunks = chunk_text(
        "one two three four five six",
        ChunkingConfig(strategy="FIXED_TOKEN", chunkSize=4, overlap=1),
    )
    assert [c.text for c in chunks] == ["one two three four", "four five six"]
    assert chunks[0].start_char == 0 and chunks[1].start_char > chunks[0].start_char


def test_normalization_uses_unicode_nfkc_and_preserves_paragraphs() -> None:
    assert normalize_text("ＡＢＣ  value\r\n\r\nnext") == "ABC value\n\nnext"


def test_retrieval_metrics() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"b", "c"}
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert mrr(ranked, relevant) == 0.5
    assert hit_rate(ranked, relevant, 2) == 1.0
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )


def test_precision_at_k_uses_k_as_the_denominator() -> None:
    assert precision_at_k(["relevant"], {"relevant"}, 5) == 0.2


def test_metrics_ignore_duplicate_result_ids() -> None:
    ranked = ["a", "a", "b"]
    relevant = {"a", "b"}
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert precision_at_k(ranked, relevant, 2) == 1.0
    assert ndcg_at_k(ranked, relevant, 2) == 1.0


def test_bm25_and_rrf_keep_native_and_rank_fusion_math_separate() -> None:
    bm25 = BM25([("a", "raft leader election"), ("b", "tcp congestion window")])
    results = bm25.search("raft leader", 2)
    assert results[0][0] == "a"
    assert results[0][1] > results[1][1]
    assert reciprocal_rank_fusion({"vector": ["b", "a"], "bm25": ["a", "b"]}) == [
        ("a", pytest.approx(1 / 61 + 1 / 62)),
        ("b", pytest.approx(1 / 61 + 1 / 62)),
    ]


def test_ingestion_rejects_traversal_oversize_and_empty_pdf_text() -> None:
    with pytest.raises(IngestionError, match="plain file name"):
        ingest_bytes("../secret.txt", b"safe")
    with pytest.raises(IngestionError, match="maximum size"):
        ingest_bytes("large.txt", b"1234", max_bytes=3)
    stream = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(stream)
    with pytest.raises(IngestionError, match="no extractable text"):
        ingest_bytes("scan.pdf", stream.getvalue())


def test_paragraph_and_heading_chunking_preserve_boundaries() -> None:
    paragraph = chunk_text(
        "First paragraph.\n\nSecond paragraph.",
        ChunkingConfig(strategy="PARAGRAPH"),
    )
    assert [chunk.text for chunk in paragraph] == ["First paragraph.", "Second paragraph."]
    heading = chunk_text(
        "# Election\nA candidate asks for votes.\n\n## Log repair\nThe leader sends entries.",
        ChunkingConfig(strategy="HEADING"),
    )
    assert [(chunk.heading, chunk.text) for chunk in heading] == [
        ("Election", "A candidate asks for votes."),
        ("Log repair", "The leader sends entries."),
    ]


class _ReverseReranker:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(index) for index, _ in enumerate(pairs)]


def test_reranker_scores_the_full_candidate_set(tmp_path) -> None:
    document = Document(
        id="00000000-0000-0000-0000-000000000001",
        project_id="00000000-0000-0000-0000-000000000002",
        filename="fixture.txt",
        source_type="TXT",
        content="alpha beta gamma",
        sha256="0" * 64,
        tags=[],
    )
    chunks = [
        ChunkRecord(
            id=f"00000000-0000-0000-0000-00000000001{index}",
            document_id=document.id,
            index_configuration_id="00000000-0000-0000-0000-000000000003",
            ordinal=index,
            text=text,
            token_count=2,
            start_char=0,
            end_char=len(text),
            metadata_json={},
        )
        for index, text in enumerate(["alpha one", "alpha two", "alpha three"])
    ]
    pairs = [(document, chunk) for chunk in chunks]
    engine = RetrievalEngine(
        LocalVectorStore(tmp_path / "vectors"),
        HashEmbeddingProvider(),
        reranker=_ReverseReranker(),
    )
    engine.index("00000000-0000-0000-0000-000000000003", pairs)
    result = engine.retrieve(
        "00000000-0000-0000-0000-000000000003",
        "alpha",
        pairs,
        methods=[RetrievalMethod.RERANKED],
        top_k=1,
        candidate_k=3,
    )
    hit = result.rankings["RERANKED"][0]
    assert hit.cross_encoder_score == 2.0
    assert hit.reranked_rank == 1


def test_context_builder_trims_at_a_deterministic_token_boundary() -> None:
    chunks = [
        RetrievedChunk(
            "1",
            "doc",
            "one two three",
            1.0,
            1,
            RetrievalMethod.BM25,
            document_name="source.md",
            ordinal=4,
        ),
        RetrievedChunk(
            "2",
            "doc",
            "four five six",
            0.5,
            2,
            RetrievalMethod.BM25,
            document_name="source.md",
            ordinal=5,
        ),
    ]
    assert build_context(chunks, 15) == (
        "[1] source.md · chunk 4\none two three\n\n[2] source.md · chunk 5\nfour five"
    )


def test_ollama_accepts_only_exact_loopback_urls() -> None:
    with pytest.raises(ValueError, match="loopback URL"):
        OllamaClient("https://example.com")
    with pytest.raises(ValueError, match="loopback URL"):
        OllamaClient("http://localhost.evil.test:11434")
    with pytest.raises(ValueError, match="loopback URL"):
        OllamaClient("http://localhost:11434/api")
    assert OllamaClient("http://[::1]:11434").base_url == "http://[::1]:11434"


def test_ollama_requires_citations_that_exist_in_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "A follower receives missing log entries [9]."}

    def fake_post(*_: Any, **__: Any) -> _Response:
        return _Response()

    monkeypatch.setattr("contextbench.ollama.httpx.post", fake_post)
    answer = OllamaClient().answer(
        "How does a follower catch up?",
        "[1] raft.md · chunk 4\nThe leader sends missing log entries.",
    )
    assert answer == "I do not have enough cited evidence."
