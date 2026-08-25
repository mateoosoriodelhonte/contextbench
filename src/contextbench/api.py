"""FastAPI application for the documented local API."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select

from .chunking import chunk_text
from .db import create_session_factory
from .embedding import (
    HashEmbeddingProvider,
    ModelDownloadRequired,
    SentenceTransformersEmbeddingProvider,
)
from .experiments import compare_experiments, evaluate_rankings, export_experiment
from .ingestion import IngestionError, ingest_bytes
from .models import (
    ChunkRecord,
    Document,
    EvaluationQuery,
    Experiment,
    ExperimentResult,
    IndexConfiguration,
    Project,
    RelevantChunk,
)
from .retrieval import RetrievalEngine
from .schemas import (
    CreateProjectRequest,
    ErrorBody,
    ErrorResponse,
    EvaluationQueryRequest,
    ExperimentRequest,
    IndexConfigurationRequest,
    Page,
    Pagination,
    RetrieveRequest,
)
from .vector_store import LocalVectorStore


class APIError(Exception):
    def __init__(
        self, code: str, message: str, status: int = 400, details: dict[str, Any] | None = None
    ) -> None:
        self.code, self.message, self.status, self.details = code, message, status, details or {}


def _error(exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content=ErrorResponse(
            error=ErrorBody(code=exc.code, message=exc.message, details=exc.details)
        ).model_dump(by_alias=True),
    )


def _page(data: list[Any], page: int, page_size: int, total: int) -> dict[str, Any]:
    return Page(
        data=data,
        pagination=Pagination(
            page=page,
            page_size=page_size,
            total_items=total,
            total_pages=math.ceil(total / page_size) if total else 0,
        ),
    ).model_dump(by_alias=True)


class AppState:
    def __init__(self, db_path: str | Path, vector_path: str | Path) -> None:
        self.factory = create_session_factory(db_path)
        self.store = LocalVectorStore(vector_path)

    def embedder(self, config: IndexConfiguration) -> Any:
        data = config.embedding_json
        if data.get("provider") == "sentence-transformers":
            return SentenceTransformersEmbeddingProvider(
                data.get("model", "BAAI/bge-small-en-v1.5"),
                normalize=bool(data.get("normalize", True)),
                allow_model_download=bool(
                    data.get("allowModelDownload", data.get("allow_model_download", False))
                ),
            )
        return HashEmbeddingProvider(
            int(data.get("dimension", config.vector_dimension)), bool(data.get("normalize", True))
        )


def create_app(
    db_path: str | Path = "contextbench.sqlite3", vector_path: str | Path = "contextbench-qdrant"
) -> FastAPI:
    state = AppState(db_path, vector_path)
    app = FastAPI(title="ContextBench", version="1.0.0")
    app.state.contextbench = state

    @app.exception_handler(APIError)
    async def api_error_handler(_: Any, exc: APIError) -> JSONResponse:
        return _error(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Any, exc: RequestValidationError) -> JSONResponse:
        return _error(
            APIError("VALIDATION_ERROR", "The request is invalid.", 422, {"errors": exc.errors()})
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "storage": "sqlite", "vectorStore": "qdrant-local"}

    @app.get("/api/v1/projects")
    def list_projects(page: int = 1, page_size: int = 50) -> dict[str, Any]:
        with state.factory() as session:
            rows = list(session.scalars(select(Project).order_by(Project.created_at.desc())))
        data = [
            {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "createdAt": row.created_at,
            }
            for row in rows
        ]
        return _page(data[(page - 1) * page_size : page * page_size], page, page_size, len(data))

    @app.post("/api/v1/projects", status_code=201)
    def create_project(request: CreateProjectRequest) -> dict[str, Any]:
        with state.factory.begin() as session:
            project = Project(name=request.name, description=request.description)
            session.add(project)
            session.flush()
            return {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "createdAt": project.created_at,
            }

    def project_or_error(session: Any, project_id: str) -> Project:
        project = session.get(Project, project_id)
        if project is None:
            raise APIError("NOT_FOUND", "Project not found.", 404)
        return cast(Project, project)

    @app.get("/api/v1/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        with state.factory() as session:
            project = project_or_error(session, project_id)
            return {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "createdAt": project.created_at,
                "documentCount": len(project.documents),
                "indexCount": len(project.indexes),
            }

    @app.get("/api/v1/projects/{project_id}/documents")
    def list_documents(project_id: str, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        with state.factory() as session:
            project_or_error(session, project_id)
            rows = list(
                session.scalars(
                    select(Document)
                    .where(Document.project_id == project_id)
                    .order_by(Document.created_at.desc())
                )
            )
            data = [
                {
                    "id": row.id,
                    "filename": row.filename,
                    "sourceType": row.source_type,
                    "sha256": row.sha256,
                    "createdAt": row.created_at,
                }
                for row in rows
            ]
        return _page(data[(page - 1) * page_size : page * page_size], page, page_size, len(data))

    @app.post("/api/v1/projects/{project_id}/documents", status_code=201)
    async def ingest_document(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        data = await file.read()
        try:
            ingested = ingest_bytes(file.filename or "document.txt", data)
        except (IngestionError, UnicodeError) as exc:
            raise APIError("VALIDATION_ERROR", str(exc), 422) from exc
        with state.factory.begin() as session:
            project_or_error(session, project_id)
            existing = session.scalar(
                select(Document).where(
                    Document.project_id == project_id, Document.sha256 == ingested.sha256
                )
            )
            if existing is not None:
                return {
                    "id": existing.id,
                    "filename": existing.filename,
                    "sourceType": existing.source_type,
                    "sha256": existing.sha256,
                    "createdAt": existing.created_at,
                    "reused": True,
                }
            document = Document(
                project_id=project_id,
                filename=ingested.filename,
                source_type=ingested.source_type.value,
                content=ingested.text,
                sha256=ingested.sha256,
                tags=[],
            )
            session.add(document)
            session.flush()
            return {
                "id": document.id,
                "filename": document.filename,
                "sourceType": document.source_type,
                "sha256": document.sha256,
                "createdAt": document.created_at,
                "reused": False,
            }

    @app.get("/api/v1/documents/{document_id}/chunks")
    def list_chunks(document_id: str, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        with state.factory() as session:
            rows = list(
                session.scalars(
                    select(ChunkRecord)
                    .where(ChunkRecord.document_id == document_id)
                    .order_by(ChunkRecord.ordinal)
                )
            )
        data = [
            {
                "id": row.id,
                "documentId": row.document_id,
                "ordinal": row.ordinal,
                "text": row.text,
                "tokenCount": row.token_count,
                "startChar": row.start_char,
                "endChar": row.end_char,
                "heading": row.heading,
                "metadata": row.metadata_json,
            }
            for row in rows
        ]
        return _page(data[(page - 1) * page_size : page * page_size], page, page_size, len(data))

    @app.get("/api/v1/projects/{project_id}/indexes")
    def list_indexes(project_id: str) -> dict[str, Any]:
        with state.factory() as session:
            project_or_error(session, project_id)
            rows = list(
                session.scalars(
                    select(IndexConfiguration).where(IndexConfiguration.project_id == project_id)
                )
            )
        return {"data": [_index_response(row) for row in rows]}

    @app.post("/api/v1/projects/{project_id}/indexes", status_code=201)
    def create_index(project_id: str, request: IndexConfigurationRequest) -> dict[str, Any]:
        with state.factory.begin() as session:
            project = project_or_error(session, project_id)
            embedding = request.embedding.model_dump(mode="json", by_alias=True)
            config = IndexConfiguration(
                project_id=project.id,
                name=request.name,
                chunking_json=request.chunking.model_dump(mode="json", by_alias=True),
                embedding_json=embedding,
                vector_dimension=request.embedding.dimension,
                status="BUILDING",
            )
            session.add(config)
            session.flush()
            for document in project.documents:
                document.chunks.clear()
                document.chunks.extend(
                    ChunkRecord(
                        ordinal=part.ordinal,
                        text=part.text,
                        token_count=part.token_count,
                        start_char=part.start_char,
                        end_char=part.end_char,
                        heading=part.heading,
                        metadata_json=part.metadata or {},
                    )
                    for part in chunk_text(document.content, request.chunking)
                )
            session.flush()
            chunks = [(doc, chunk) for doc in project.documents for chunk in doc.chunks]
            try:
                embedder = state.embedder(config)
                if embedder.dimension == 0:
                    embedder.embed([""])
                    config.vector_dimension = embedder.dimension
                engine = RetrievalEngine(state.store, embedder)
                count = engine.index(config.id, chunks)
                config.status = "READY"
            except ModelDownloadRequired as exc:
                config.status = "FAILED"
                raise APIError(
                    "MODEL_DOWNLOAD_REQUIRED",
                    "The embedding model is not cached.",
                    409,
                    {"model": str(exc)},
                ) from exc
            return _index_response(config) | {"chunkCount": count}

    @app.post("/api/v1/projects/{project_id}/retrieve")
    def retrieve(project_id: str, request: RetrieveRequest) -> dict[str, Any]:
        with state.factory() as session:
            project = project_or_error(session, project_id)
            config = session.get(IndexConfiguration, str(request.index_configuration_id))
            if config is None or config.project_id != project_id:
                raise APIError("NOT_FOUND", "Index configuration not found.", 404)
            chunks = [(doc, chunk) for doc in project.documents for chunk in doc.chunks]
            try:
                result = RetrievalEngine(state.store, state.embedder(config)).retrieve(
                    config.id,
                    request.query,
                    chunks,
                    methods=request.methods,
                    top_k=request.top_k,
                    candidate_k=request.candidate_k,
                    filters=request.filters,
                    max_context_tokens=request.max_context_tokens,
                )
            except ModelDownloadRequired as exc:
                raise APIError(
                    "MODEL_DOWNLOAD_REQUIRED",
                    "The embedding model is not cached.",
                    409,
                    {"model": str(exc)},
                ) from exc
            rankings = {
                method: [
                    {
                        "chunkId": item.chunk_id,
                        "documentId": item.document_id,
                        "text": item.text,
                        "nativeScore": item.native_score,
                        "rank": item.rank,
                        "rrfScore": item.rrf_score,
                        "crossEncoderScore": item.cross_encoder_score,
                        "rerankedRank": item.reranked_rank,
                    }
                    for item in items
                ]
                for method, items in result.rankings.items()
            }
            return {
                "rankings": rankings,
                "latenciesMs": result.latencies_ms,
                "context": result.context,
            }

    @app.get("/api/v1/projects/{project_id}/evaluation-queries")
    def list_eval_queries(project_id: str) -> dict[str, Any]:
        with state.factory() as session:
            project_or_error(session, project_id)
            rows = list(
                session.scalars(
                    select(EvaluationQuery).where(EvaluationQuery.project_id == project_id)
                )
            )
            data = [
                {
                    "id": row.id,
                    "query": row.query,
                    "datasetVersion": row.dataset_version,
                    "relevantChunkIds": [item.chunk_id for item in row.relevant_chunks],
                }
                for row in rows
            ]
        return {"data": data}

    @app.post("/api/v1/projects/{project_id}/evaluation-queries", status_code=201)
    def create_eval_query(project_id: str, request: EvaluationQueryRequest) -> dict[str, Any]:
        with state.factory.begin() as session:
            project_or_error(session, project_id)
            row = EvaluationQuery(
                project_id=project_id, query=request.query, dataset_version=request.dataset_version
            )
            row.relevant_chunks = [
                RelevantChunk(chunk_id=str(chunk_id)) for chunk_id in request.relevant_chunk_ids
            ]
            session.add(row)
            session.flush()
            return {
                "id": row.id,
                "query": row.query,
                "datasetVersion": row.dataset_version,
                "relevantChunkIds": [str(item) for item in request.relevant_chunk_ids],
            }

    @app.get("/api/v1/projects/{project_id}/experiments")
    def list_experiments(project_id: str) -> dict[str, Any]:
        with state.factory() as session:
            project_or_error(session, project_id)
            rows = list(
                session.scalars(select(Experiment).where(Experiment.project_id == project_id))
            )
        return {
            "data": [
                {"id": row.id, "name": row.name, "status": row.status, "createdAt": row.created_at}
                for row in rows
            ]
        }

    @app.post("/api/v1/projects/{project_id}/experiments", status_code=201)
    def execute_experiment(project_id: str, request: ExperimentRequest) -> dict[str, Any]:
        with state.factory.begin() as session:
            project = project_or_error(session, project_id)
            config = session.get(IndexConfiguration, str(request.index_configuration_id))
            if config is None or config.project_id != project_id:
                raise APIError("NOT_FOUND", "Index configuration not found.", 404)
            experiment = Experiment(
                project_id=project_id,
                name=request.name,
                config_json=request.model_dump(mode="json", by_alias=True),
                status="RUNNING",
            )
            session.add(experiment)
            session.flush()
            chunks = [(doc, chunk) for doc in project.documents for chunk in doc.chunks]
            engine = RetrievalEngine(state.store, state.embedder(config))
            queries = list(
                session.scalars(
                    select(EvaluationQuery).where(
                        EvaluationQuery.project_id == project_id,
                        EvaluationQuery.dataset_version == request.dataset_version,
                    )
                )
            )
            for query_row in queries:
                result = engine.retrieve(
                    config.id,
                    query_row.query,
                    chunks,
                    methods=[request.method],
                    top_k=max(request.k_values),
                    candidate_k=max(request.k_values),
                )
                ranked_ids = [
                    item.chunk_id for item in result.rankings.get(request.method.value, [])
                ]
                metrics = evaluate_rankings(
                    ranked_ids,
                    {item.chunk_id for item in query_row.relevant_chunks},
                    request.k_values,
                )
                experiment.results.append(
                    ExperimentResult(
                        evaluation_query_id=query_row.id,
                        metrics_json=metrics,
                        rankings_json={"method": request.method.value, "chunkIds": ranked_ids},
                    )
                )
            experiment.status = "COMPLETED"
            return {
                "id": experiment.id,
                "name": experiment.name,
                "status": experiment.status,
                "resultCount": len(experiment.results),
                "createdAt": experiment.created_at,
            }

    @app.get("/api/v1/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        with state.factory() as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment is None:
                raise APIError("NOT_FOUND", "Experiment not found.", 404)
            aggregate: dict[str, list[float]] = {}
            for result in experiment.results:
                for key, value in result.metrics_json.items():
                    aggregate.setdefault(key, []).append(float(value))
            metrics = {
                key: sum(values) / len(values) for key, values in aggregate.items() if values
            }
            return {
                "id": experiment.id,
                "name": experiment.name,
                "status": experiment.status,
                "configuration": experiment.config_json,
                "metrics": metrics,
                "results": [
                    {
                        "evaluationQueryId": result.evaluation_query_id,
                        "metrics": result.metrics_json,
                        "rankings": result.rankings_json,
                    }
                    for result in experiment.results
                ],
            }

    @app.get("/api/v1/experiments/{experiment_id}/export")
    def export(experiment_id: str) -> JSONResponse:
        with state.factory() as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment is None:
                raise APIError("NOT_FOUND", "Experiment not found.", 404)
            payload = {
                "id": experiment.id,
                "name": experiment.name,
                "status": experiment.status,
                "configuration": experiment.config_json,
                "results": [
                    {
                        "evaluationQueryId": result.evaluation_query_id,
                        "metrics": result.metrics_json,
                        "rankings": result.rankings_json,
                    }
                    for result in experiment.results
                ],
            }
        return JSONResponse(content=json.loads(export_experiment(payload)))

    @app.post("/api/v1/experiments/compare")
    def compare(request: dict[str, Any]) -> dict[str, Any]:
        ids = request.get("experimentIds", [])
        with state.factory() as session:
            experiments: list[dict[str, Any]] = []
            for experiment_id in ids:
                experiment = session.get(Experiment, str(experiment_id))
                if experiment is None:
                    raise APIError("NOT_FOUND", "Experiment not found.", 404)
                aggregate: dict[str, list[float]] = {}
                for result in experiment.results:
                    for key, value in result.metrics_json.items():
                        aggregate.setdefault(key, []).append(float(value))
                experiments.append(
                    {
                        "name": experiment.name,
                        "metrics": {
                            key: sum(values) / len(values)
                            for key, values in aggregate.items()
                            if values
                        },
                    }
                )
        try:
            return compare_experiments(experiments)
        except ValueError as exc:
            raise APIError("VALIDATION_ERROR", str(exc), 422) from exc

    @app.post("/api/v1/demo", status_code=201)
    def create_demo() -> dict[str, Any]:
        return _create_demo(state)

    return app


def _index_response(config: IndexConfiguration) -> dict[str, Any]:
    return {
        "id": config.id,
        "projectId": config.project_id,
        "name": config.name,
        "chunking": config.chunking_json,
        "embedding": config.embedding_json,
        "status": config.status,
        "vectorDimension": config.vector_dimension,
        "createdAt": config.created_at,
    }


def _create_demo(state: AppState) -> dict[str, Any]:
    corpus = [
        (
            "raft.md",
            "# Raft\nA follower that falls behind catches up from the leader log.\n\n"
            "# Safety\nRaft preserves a committed entry across elections.",
        ),
        (
            "retrieval.txt",
            "BM25 ranks lexical matches. Hybrid retrieval combines independent rankings "
            "with reciprocal rank fusion.",
        ),
    ]
    with state.factory.begin() as session:
        project = session.scalar(select(Project).where(Project.name == "ContextBench Demo"))
        project_reused = project is not None
        if project is None:
            project = Project(name="ContextBench Demo", description="Deterministic built-in corpus")
            session.add(project)
            session.flush()
        for filename, text in corpus:
            raw = text.encode()
            digest = __import__("hashlib").sha256(raw).hexdigest()
            document = session.scalar(
                select(Document).where(Document.project_id == project.id, Document.sha256 == digest)
            )
            if document is None:
                document = Document(
                    project_id=project.id,
                    filename=filename,
                    source_type="MD" if filename.endswith("md") else "TXT",
                    content=text,
                    sha256=digest,
                    tags=[],
                )
                session.add(document)
                session.flush()
                for part in chunk_text(
                    text,
                    __import__("contextbench.schemas", fromlist=["ChunkingConfig"]).ChunkingConfig(
                        strategy="PARAGRAPH"
                    ),
                ):
                    document.chunks.append(
                        ChunkRecord(
                            ordinal=part.ordinal,
                            text=part.text,
                            token_count=part.token_count,
                            start_char=part.start_char,
                            end_char=part.end_char,
                            heading=part.heading,
                            metadata_json=part.metadata or {},
                        )
                    )
        config = session.scalar(
            select(IndexConfiguration).where(IndexConfiguration.project_id == project.id)
        )
        if config is None:
            config = IndexConfiguration(
                project_id=project.id,
                name="demo-hash",
                chunking_json={"strategy": "PARAGRAPH", "chunkSize": 256, "overlap": 0},
                embedding_json={
                    "provider": "hash",
                    "model": "contextbench-hash-v1",
                    "dimension": 64,
                    "normalize": True,
                },
                vector_dimension=64,
                status="BUILDING",
            )
            session.add(config)
            session.flush()
            RetrievalEngine(state.store, HashEmbeddingProvider()).index(
                config.id, [(doc, chunk) for doc in project.documents for chunk in doc.chunks]
            )
            config.status = "READY"
            config_reused = False
        else:
            config_reused = True
        return {
            "projectId": project.id,
            "indexConfigurationId": config.id,
            "reused": project_reused and config_reused,
        }
