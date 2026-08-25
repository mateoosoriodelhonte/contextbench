"""Validated wire and domain schemas for ContextBench."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


class CBModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class SourceType(StrEnum):
    TXT = "TXT"
    MD = "MD"
    PDF = "PDF"
    SOURCE = "SOURCE"


class ChunkStrategy(StrEnum):
    FIXED_TOKEN = "FIXED_TOKEN"  # noqa: S105
    PARAGRAPH = "PARAGRAPH"
    HEADING = "HEADING"


class EmbeddingProvider(StrEnum):
    HASH = "hash"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class RetrievalMethod(StrEnum):
    VECTOR = "VECTOR"
    BM25 = "BM25"
    HYBRID = "HYBRID"
    RERANKED = "RERANKED"


class ChunkingConfig(CBModel):
    strategy: ChunkStrategy = ChunkStrategy.FIXED_TOKEN
    chunk_size: int = Field(default=256, ge=1, le=8192)
    overlap: int = Field(default=32, ge=0, le=4096)

    @field_validator("overlap")
    @classmethod
    def overlap_less_than_chunk_size(cls, value: int, info: Any) -> int:
        size = info.data.get("chunk_size", 256)
        if value >= size:
            raise ValueError("overlap must be less than chunkSize")
        return value


class EmbeddingConfig(CBModel):
    provider: EmbeddingProvider = EmbeddingProvider.HASH
    model: str = "contextbench-hash-v1"
    revision: str | None = Field(default=None, max_length=200)
    dimension: int = Field(default=64, ge=1, le=8192)
    normalize: bool = True
    allow_model_download: bool = False


class RerankerConfig(CBModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    revision: str | None = Field(default=None, max_length=200)
    allow_model_download: bool = False


class CreateProjectRequest(CBModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)


class ProjectResponse(CBModel):
    id: UUID
    name: str
    description: str | None = None
    created_at: datetime


class IndexConfigurationRequest(CBModel):
    name: str = Field(min_length=1, max_length=200)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class IndexConfigurationResponse(IndexConfigurationRequest):
    id: UUID
    project_id: UUID
    status: str
    vector_dimension: int
    created_at: datetime


class RetrievalFilters(CBModel):
    document_ids: list[UUID] = Field(default_factory=list)
    source_types: list[SourceType] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class RetrieveRequest(CBModel):
    query: str = Field(min_length=1, max_length=10000)
    index_configuration_id: UUID
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int = Field(default=20, ge=1, le=1000)
    methods: list[RetrievalMethod] = Field(default_factory=lambda: [RetrievalMethod.HYBRID])
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    max_context_tokens: int = Field(default=1200, ge=1, le=100000)
    reranker: RerankerConfig | None = None

    @model_validator(mode="after")
    def retrieval_bounds(self) -> RetrieveRequest:
        if self.candidate_k < self.top_k:
            raise ValueError("candidateK must be greater than or equal to topK")
        if not self.methods:
            raise ValueError("methods must contain at least one retrieval method")
        self.methods = list(dict.fromkeys(self.methods))
        return self


class EvaluationQueryRequest(CBModel):
    query: str = Field(min_length=1, max_length=10000)
    relevant_chunk_ids: list[UUID] = Field(default_factory=list)
    dataset_version: int = Field(default=1, ge=1)
    notes: str | None = Field(default=None, max_length=4000)


class EvaluationDatasetImport(CBModel):
    schema_name: Literal["contextbench.evaluation.v1"] = Field(alias="schema")
    dataset_version: int = Field(default=1, ge=1)
    queries: list[EvaluationQueryRequest] = Field(min_length=1, max_length=1000)


class GenerateRequest(CBModel):
    question: str = Field(min_length=1, max_length=10000)
    context: str = Field(max_length=500000)
    model: str = Field(default="llama3.2", min_length=1, max_length=200)
    base_url: str = Field(default="http://127.0.0.1:11434", max_length=2048)
    minimum_evidence_tokens: int = Field(default=20, ge=1, le=10000)
    minimum_query_term_matches: int = Field(default=1, ge=0, le=100)


class ExperimentRequest(CBModel):
    name: str = Field(min_length=1, max_length=200)
    dataset_version: int = Field(default=1, ge=1)
    index_configuration_id: UUID
    method: RetrievalMethod = RetrievalMethod.HYBRID
    reranker: RerankerConfig | None = None
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])

    @field_validator("k_values")
    @classmethod
    def valid_k_values(cls, value: list[int]) -> list[int]:
        if not value or any(k < 1 or k > 1000 for k in value):
            raise ValueError("kValues must contain positive values")
        return sorted(set(value))


class ExperimentComparisonRequest(CBModel):
    experiment_ids: list[UUID] = Field(min_length=2, max_length=20)


class Pagination(CBModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int


class Page(CBModel):
    data: list[Any]
    pagination: Pagination


class ErrorBody(CBModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(CBModel):
    error: ErrorBody


class ChunkResponse(CBModel):
    id: UUID
    document_id: UUID
    ordinal: int
    text: str
    token_count: int
    start_char: int
    end_char: int
    heading: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
