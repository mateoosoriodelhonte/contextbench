import io
import math
import sys
import types
from typing import Any

import pytest
from pypdf import PdfWriter

from contextbench.bm25 import BM25, reciprocal_rank_fusion
from contextbench.chunking import chunk_text, normalize_text
from contextbench.embedding import (
    CrossEncoderReranker,
    HashEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
)
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
    with pytest.raises(IngestionError, match="valid UTF-8"):
        ingest_bytes("invalid.txt", b"\xff")


def test_ingestion_rejects_excessive_pdf_page_count(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Reader:
        pages = [object()] * 1001

    monkeypatch.setattr("contextbench.ingestion.PdfReader", lambda *_args, **_kwargs: _Reader())
    with pytest.raises(IngestionError, match="maximum page count"):
        ingest_bytes("large.pdf", b"%PDF fixture")


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
    assert hit.native_score == 2.0
    assert hit.rank == 1
    assert hit.candidate_rank == 3
    assert hit.vector_score is not None
    assert hit.bm25_score is not None
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
    assert build_context(chunks, 17) == (
        "[1] source.md · chunk 4\n> one two three\n\n[2] source.md · chunk 5\n> four five"
    )


class _FixedVectorStore:
    def search(self, *_: Any, **__: Any) -> list[tuple[str, float, dict[str, Any]]]:
        return [
            ("00000000-0000-0000-0000-000000000002", 0.9, {}),
            ("00000000-0000-0000-0000-000000000001", 0.8, {}),
        ]


def test_final_context_uses_the_highest_order_requested_ranking() -> None:
    document = Document(
        id="00000000-0000-0000-0000-000000000010",
        project_id="00000000-0000-0000-0000-000000000011",
        filename="ranking.md",
        source_type="MD",
        content="",
        sha256="0" * 64,
        tags=[],
    )
    chunks = [
        ChunkRecord(
            id="00000000-0000-0000-0000-000000000001",
            document_id=document.id,
            index_configuration_id="00000000-0000-0000-0000-000000000012",
            ordinal=0,
            text="target target",
            token_count=2,
            start_char=0,
            end_char=13,
            metadata_json={},
        ),
        ChunkRecord(
            id="00000000-0000-0000-0000-000000000002",
            document_id=document.id,
            index_configuration_id="00000000-0000-0000-0000-000000000012",
            ordinal=1,
            text="unrelated",
            token_count=1,
            start_char=14,
            end_char=23,
            metadata_json={},
        ),
    ]
    result = RetrievalEngine(  # type: ignore[arg-type]
        _FixedVectorStore(), HashEmbeddingProvider()
    ).retrieve(
        "00000000-0000-0000-0000-000000000012",
        "target",
        [(document, chunk) for chunk in chunks],
        methods=[RetrievalMethod.VECTOR, RetrievalMethod.BM25, RetrievalMethod.HYBRID],
        top_k=1,
        candidate_k=2,
    )
    assert result.rankings["VECTOR"][0].chunk_id.endswith("2")
    assert result.rankings["HYBRID"][0].chunk_id.endswith("1")
    assert result.context_method is RetrievalMethod.HYBRID
    assert "target target" in result.context
    assert "unrelated" not in result.context


def test_local_ml_providers_resolve_revision_and_use_asymmetric_encoders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("sentence_transformers")

    class _Config:
        _commit_hash = "a" * 40

    class _Transformer:
        auto_model = types.SimpleNamespace(config=_Config())

    class _SentenceTransformer:
        def __init__(self, _: str, **kwargs: object) -> None:
            assert kwargs["revision"] == "requested-revision"

        def __getitem__(self, _: int) -> _Transformer:
            return _Transformer()

        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode_document(self, _: list[str], **__: object) -> list[list[float]]:
            return [[1.0, 0.0]]

        def encode_query(self, _: list[str], **__: object) -> list[list[float]]:
            return [[0.0, 1.0]]

    class _CrossEncoder:
        def __init__(self, _: str, **__: object) -> None:
            self.model = types.SimpleNamespace(config=_Config())

        def predict(self, _: list[tuple[str, str]]) -> list[float]:
            return [0.75]

    module.SentenceTransformer = _SentenceTransformer  # type: ignore[attr-defined]
    module.CrossEncoder = _CrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedder = SentenceTransformersEmbeddingProvider(
        "fixture/model", revision="requested-revision", allow_model_download=True
    )
    assert embedder.embed_documents(["document"]) == [[1.0, 0.0]]
    assert embedder.embed_query("query") == [0.0, 1.0]
    assert embedder.resolved_revision == "a" * 40
    assert embedder.dimension == 2

    reranker = CrossEncoderReranker("fixture/reranker", allow_model_download=True)
    assert reranker.predict([("query", "document")]) == [0.75]
    assert reranker.resolved_revision == "a" * 40


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
        "[1] raft.md · chunk 4\n> The leader sends missing log entries.\n> [9] injected label",
    )
    assert answer == "I do not have enough cited evidence."
