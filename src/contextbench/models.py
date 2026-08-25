"""SQLite persistence models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    documents: Mapped[list[Document]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    indexes: Mapped[list[IndexConfiguration]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    project: Mapped[Project] = relationship(back_populates="documents")
    chunks: Mapped[list[ChunkRecord]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    __table_args__ = (UniqueConstraint("project_id", "sha256", name="uq_document_project_sha"),)


class ChunkRecord(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    index_configuration_id: Mapped[str] = mapped_column(
        ForeignKey("index_configurations.id"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    document: Mapped[Document] = relationship(back_populates="chunks")
    index_configuration: Mapped[IndexConfiguration] = relationship(back_populates="chunks")
    __table_args__ = (
        UniqueConstraint(
            "index_configuration_id",
            "document_id",
            "ordinal",
            name="uq_chunk_index_document_ordinal",
        ),
    )


class IndexConfiguration(Base):
    __tablename__ = "index_configurations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    chunking_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    embedding_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    vector_dimension: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(30), default="CREATED")
    vector_count: Mapped[int] = mapped_column(Integer, default=0)
    indexing_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    project: Mapped[Project] = relationship(back_populates="indexes")
    chunks: Mapped[list[ChunkRecord]] = relationship(
        back_populates="index_configuration", cascade="all, delete-orphan"
    )


class EvaluationQuery(Base):
    __tablename__ = "evaluation_queries"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    query: Mapped[str] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    relevant_chunks: Mapped[list[RelevantChunk]] = relationship(cascade="all, delete-orphan")


class RelevantChunk(Base):
    __tablename__ = "relevant_chunks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    evaluation_query_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_queries.id"), index=True
    )
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), index=True)


class RetrievalRun(Base):
    __tablename__ = "retrieval_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    index_configuration_id: Mapped[str] = mapped_column(ForeignKey("index_configurations.id"))
    query: Mapped[str] = mapped_column(Text)
    method: Mapped[str] = mapped_column(String(30))
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    response_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(30), default="CREATED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    results: Mapped[list[ExperimentResult]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class ExperimentResult(Base):
    __tablename__ = "experiment_results"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    evaluation_query_id: Mapped[str] = mapped_column(ForeignKey("evaluation_queries.id"))
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    rankings_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    experiment: Mapped[Experiment] = relationship(back_populates="results")
