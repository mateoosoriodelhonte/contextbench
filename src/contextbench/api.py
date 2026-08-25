"""FastAPI application for the documented local API."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Annotated, Any, cast

import httpx
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from .bm25 import tokenize
from .chunking import chunk_text
from .config import DEFAULT_DATABASE_PATH, DEFAULT_VECTOR_PATH
from .db import create_session_factory
from .embedding import (
    CrossEncoderReranker,
    HashEmbeddingProvider,
    ModelDownloadRequired,
    ModelRuntimeUnavailable,
    SentenceTransformersEmbeddingProvider,
)
from .experiments import compare_experiments, evaluate_rankings, export_experiment
from .ingestion import MAX_DOCUMENT_BYTES, IngestionError, ingest_bytes
from .models import (
    ChunkRecord,
    Document,
    EvaluationQuery,
    Experiment,
    ExperimentResult,
    IndexConfiguration,
    Project,
    RelevantChunk,
    RetrievalRun,
    utc_now,
)
from .ollama import OllamaClient
from .retrieval import RetrievalEngine
from .schemas import (
    CreateProjectRequest,
    ErrorBody,
    ErrorResponse,
    EvaluationDatasetImport,
    EvaluationQueryRequest,
    ExperimentComparisonRequest,
    ExperimentRequest,
    GenerateRequest,
    IndexConfigurationRequest,
    Page,
    Pagination,
    RerankerConfig,
    RetrievalMethod,
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


def _chunk_metadata(
    document: Document, start_char: int, base: dict[str, object]
) -> dict[str, object]:
    metadata = dict(base)
    for page in document.metadata_json.get("pages", []):
        if int(page["startChar"]) <= start_char < int(page["endChar"]):
            metadata["page"] = int(page["page"])
            break
    return metadata


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _chunk_snapshot(chunks: list[tuple[Document, ChunkRecord]]) -> dict[str, Any]:
    items = [
        {
            "id": chunk.id,
            "documentId": chunk.document_id,
            "ordinal": chunk.ordinal,
            "startChar": chunk.start_char,
            "endChar": chunk.end_char,
            "textSha256": hashlib.sha256(chunk.text.encode()).hexdigest(),
        }
        for _, chunk in chunks
    ]
    return {"digest": _sha256_json(items), "chunkCount": len(items)}


def _judged_chunks(session: Any, query: EvaluationQuery) -> list[ChunkRecord]:
    return sorted(
        [
            chunk
            for item in query.relevant_chunks
            if (chunk := session.get(ChunkRecord, item.chunk_id)) is not None
        ],
        key=lambda chunk: chunk.id,
    )


def _dataset_snapshot(
    session: Any, queries: list[EvaluationQuery], dataset_version: int
) -> dict[str, Any]:
    items = [
        {
            "id": query.id,
            "query": query.query,
            "notes": query.notes,
            "relevantSources": [
                {
                    "chunkId": chunk.id,
                    "documentId": chunk.document_id,
                    "startChar": chunk.start_char,
                    "endChar": chunk.end_char,
                    "textSha256": hashlib.sha256(chunk.text.encode()).hexdigest(),
                }
                for chunk in _judged_chunks(session, query)
            ],
        }
        for query in queries
    ]
    frozen = {"datasetVersion": dataset_version, "queries": items}
    return {"digest": _sha256_json(frozen), "queryCount": len(items), **frozen}


def _mapped_relevant_ids(
    session: Any,
    query: EvaluationQuery,
    target_chunks: list[tuple[Document, ChunkRecord]],
) -> set[str]:
    sources = _judged_chunks(session, query)
    targets_by_id = {target.id: target for _, target in target_chunks}
    resolved: set[str] = set()
    for source in sources:
        if source.id in targets_by_id:
            resolved.add(source.id)
            continue
        resolved.update(
            target.id
            for _, target in target_chunks
            if target.document_id == source.document_id
            and target.start_char < source.end_char
            and target.end_char > source.start_char
        )
    return resolved


def _aggregate_experiment_metrics(experiment: Experiment) -> dict[str, float]:
    aggregate: dict[str, list[float]] = {}
    for result in experiment.results:
        for key, value in result.metrics_json.items():
            aggregate.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in aggregate.items() if values}


class AppState:
    def __init__(self, db_path: str | Path, vector_path: str | Path) -> None:
        self.factory = create_session_factory(db_path)
        self.store = LocalVectorStore(vector_path)

    def embedder(self, config: IndexConfiguration) -> Any:
        data = config.embedding_json
        if data.get("provider") == "sentence-transformers":
            return SentenceTransformersEmbeddingProvider(
                data.get("model", "BAAI/bge-small-en-v1.5"),
                revision=data.get("revision"),
                normalize=bool(data.get("normalize", True)),
                allow_model_download=bool(
                    data.get("allowModelDownload", data.get("allow_model_download", False))
                ),
            )
        return HashEmbeddingProvider(
            int(data.get("dimension", config.vector_dimension)), bool(data.get("normalize", True))
        )

    @staticmethod
    def reranker(config: RerankerConfig | None) -> CrossEncoderReranker | None:
        if config is None:
            return None
        return CrossEncoderReranker(
            config.model,
            revision=config.revision,
            allow_model_download=config.allow_model_download,
        )


def create_app(
    db_path: str | Path = DEFAULT_DATABASE_PATH,
    vector_path: str | Path = DEFAULT_VECTOR_PATH,
    frontend_dir: str | Path | None = None,
) -> FastAPI:
    state = AppState(db_path, vector_path)
    app = FastAPI(title="ContextBench", version="1.0.0")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]"],
    )
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
    def list_projects(
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> dict[str, Any]:
        with state.factory() as session:
            total = int(session.scalar(select(func.count()).select_from(Project)) or 0)
            rows = list(
                session.scalars(
                    select(Project)
                    .options(selectinload(Project.documents), selectinload(Project.indexes))
                    .order_by(Project.created_at.desc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            )
            data = [
                {
                    "id": row.id,
                    "name": row.name,
                    "description": row.description,
                    "documentCount": len(row.documents),
                    "indexCount": len(row.indexes),
                    "status": "READY",
                    "updatedAt": row.created_at,
                    "createdAt": row.created_at,
                }
                for row in rows
            ]
        return _page(data, page, page_size, total)

    @app.post("/api/v1/projects", status_code=201)
    def create_project(request: CreateProjectRequest) -> dict[str, Any]:
        with state.factory.begin() as session:
            if session.scalar(select(Project.id).where(Project.name == request.name)):
                raise APIError("CONFLICT", "A project with this name already exists.", 409)
            project = Project(name=request.name, description=request.description)
            session.add(project)
            session.flush()
            return {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "documentCount": 0,
                "indexCount": 0,
                "status": "READY",
                "updatedAt": project.created_at,
                "createdAt": project.created_at,
            }

    def project_or_error(session: Any, project_id: str) -> Project:
        project = session.get(Project, project_id)
        if project is None:
            raise APIError("NOT_FOUND", "Project not found.", 404)
        return cast(Project, project)

    def chunks_for_index(
        session: Any, index_configuration_id: str
    ) -> list[tuple[Document, ChunkRecord]]:
        chunks = list(
            session.scalars(
                select(ChunkRecord)
                .where(ChunkRecord.index_configuration_id == index_configuration_id)
                .order_by(ChunkRecord.document_id, ChunkRecord.ordinal)
            )
        )
        return [(chunk.document, chunk) for chunk in chunks]

    @app.get("/api/v1/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        with state.factory() as session:
            project = project_or_error(session, project_id)
            total_tokens = int(
                session.scalar(
                    select(func.coalesce(func.sum(ChunkRecord.token_count), 0))
                    .join(IndexConfiguration)
                    .where(IndexConfiguration.project_id == project_id)
                )
                or 0
            )
            evaluation_count = int(
                session.scalar(
                    select(func.count())
                    .select_from(EvaluationQuery)
                    .where(EvaluationQuery.project_id == project_id)
                )
                or 0
            )
            latest_experiment = session.scalar(
                select(Experiment)
                .where(Experiment.project_id == project_id)
                .order_by(Experiment.created_at.desc())
                .limit(1)
            )
            return {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "createdAt": project.created_at,
                "documentCount": len(project.documents),
                "indexCount": len(project.indexes),
                "totalIndexedTokens": total_tokens,
                "evaluationQueryCount": evaluation_count,
                "latestExperiment": (
                    {
                        "id": latest_experiment.id,
                        "name": latest_experiment.name,
                        "method": latest_experiment.config_json.get("method"),
                        "executedAt": latest_experiment.executed_at,
                        "metrics": _aggregate_experiment_metrics(latest_experiment),
                    }
                    if latest_experiment is not None
                    else None
                ),
                "status": "READY",
                "updatedAt": project.created_at,
            }

    @app.get("/api/v1/projects/{project_id}/documents")
    def list_documents(
        project_id: str,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> dict[str, Any]:
        with state.factory() as session:
            project_or_error(session, project_id)
            total = int(
                session.scalar(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.project_id == project_id)
                )
                or 0
            )
            rows = list(
                session.execute(
                    select(Document, func.count(ChunkRecord.id))
                    .outerjoin(ChunkRecord)
                    .where(Document.project_id == project_id)
                    .group_by(Document.id)
                    .order_by(Document.created_at.desc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                ).all()
            )
            data = [
                {
                    "id": row.id,
                    "filename": row.filename,
                    "sourceType": row.source_type,
                    "tags": row.tags,
                    "chunkCount": chunk_count,
                    "status": "READY",
                    "updatedAt": row.created_at,
                    "sha256": row.sha256,
                    "createdAt": row.created_at,
                }
                for row, chunk_count in rows
            ]
        return _page(data, page, page_size, total)

    @app.post("/api/v1/projects/{project_id}/documents", status_code=201)
    async def ingest_document(
        project_id: str,
        file: UploadFile = File(...),  # noqa: B008
        tags: str = Form(""),  # noqa: B008
    ) -> dict[str, Any]:
        if len(tags) > 2000:
            raise APIError("VALIDATION_ERROR", "Tags exceed the maximum size.", 422)
        data = await file.read(MAX_DOCUMENT_BYTES + 1)
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
                    "tags": existing.tags,
                    "chunkCount": len(existing.chunks),
                    "status": "READY",
                    "updatedAt": existing.created_at,
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
                tags=[tag.strip()[:100] for tag in tags.split(",") if tag.strip()][:20],
                metadata_json=ingested.metadata,
            )
            session.add(document)
            session.flush()
            return {
                "id": document.id,
                "filename": document.filename,
                "sourceType": document.source_type,
                "tags": document.tags,
                "chunkCount": 0,
                "status": "READY",
                "updatedAt": document.created_at,
                "sha256": document.sha256,
                "createdAt": document.created_at,
                "reused": False,
            }

    @app.get("/api/v1/documents/{document_id}/chunks")
    def list_chunks(
        document_id: str,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=200)] = 50,
        index_configuration_id: str | None = None,
    ) -> dict[str, Any]:
        with state.factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise APIError("NOT_FOUND", "Document not found.", 404)
            statement = select(ChunkRecord).where(ChunkRecord.document_id == document_id)
            if index_configuration_id:
                statement = statement.where(
                    ChunkRecord.index_configuration_id == index_configuration_id
                )
            total = int(session.scalar(select(func.count()).select_from(statement.subquery())) or 0)
            rows = list(
                session.scalars(
                    statement.order_by(ChunkRecord.index_configuration_id, ChunkRecord.ordinal)
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            )
        data = [
            {
                "id": row.id,
                "documentId": row.document_id,
                "documentName": document.filename,
                "indexConfigurationId": row.index_configuration_id,
                "ordinal": row.ordinal,
                "text": row.text,
                "tokenCount": row.token_count,
                "startChar": row.start_char,
                "endChar": row.end_char,
                "heading": row.heading,
                "metadata": row.metadata_json,
                "page": row.metadata_json.get("page"),
            }
            for row in rows
        ]
        return _page(data, page, page_size, total)

    @app.get("/api/v1/projects/{project_id}/indexes")
    def list_indexes(project_id: str) -> dict[str, Any]:
        with state.factory() as session:
            project_or_error(session, project_id)
            rows = list(
                session.scalars(
                    select(IndexConfiguration)
                    .where(IndexConfiguration.project_id == project_id)
                    .order_by(IndexConfiguration.created_at)
                )
            )
        return {"data": [_index_response(row) for row in rows]}

    @app.post("/api/v1/projects/{project_id}/indexes", status_code=201)
    def create_index(project_id: str, request: IndexConfigurationRequest) -> dict[str, Any]:
        with state.factory.begin() as session:
            project = project_or_error(session, project_id)
            if not project.documents:
                raise APIError(
                    "VALIDATION_ERROR", "Add at least one document before building an index.", 422
                )
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
                config.chunks.extend(
                    ChunkRecord(
                        document=document,
                        ordinal=part.ordinal,
                        text=part.text,
                        token_count=part.token_count,
                        start_char=part.start_char,
                        end_char=part.end_char,
                        heading=part.heading,
                        metadata_json=_chunk_metadata(
                            document, part.start_char, part.metadata or {}
                        ),
                    )
                    for part in chunk_text(document.content, request.chunking)
                )
            session.flush()
            chunks = chunks_for_index(session, config.id)
            try:
                indexing_started = time.perf_counter()
                embedder = state.embedder(config)
                engine = RetrievalEngine(state.store, embedder)
                count = engine.index(config.id, chunks)
                config.vector_count = count
                config.indexing_ms = (time.perf_counter() - indexing_started) * 1000
                config.embedding_json = config.embedding_json | {
                    "dimension": embedder.dimension,
                    "revision": getattr(embedder, "resolved_revision", None)
                    or config.embedding_json.get("revision"),
                    "allowModelDownload": False,
                }
                config.status = "READY"
            except ModelDownloadRequired as exc:
                config.status = "FAILED"
                raise APIError(
                    "MODEL_DOWNLOAD_REQUIRED",
                    "The requested local model is not cached.",
                    409,
                    {"model": str(exc)},
                ) from exc
            except ModelRuntimeUnavailable as exc:
                config.status = "FAILED"
                raise APIError("MODEL_RUNTIME_UNAVAILABLE", str(exc), 409) from exc
            return _index_response(config) | {"chunkCount": count}

    @app.post("/api/v1/projects/{project_id}/retrieve")
    def retrieve(project_id: str, request: RetrieveRequest) -> dict[str, Any]:
        if RetrievalMethod.RERANKED in request.methods and request.reranker is None:
            raise APIError(
                "RERANKER_REQUIRED",
                "A reranker configuration is required for RERANKED retrieval.",
                422,
            )
        with state.factory.begin() as session:
            project = project_or_error(session, project_id)
            config = session.get(IndexConfiguration, str(request.index_configuration_id))
            if config is None or config.project_id != project_id:
                raise APIError("NOT_FOUND", "Index configuration not found.", 404)
            chunks = chunks_for_index(session, config.id)
            reranker = state.reranker(request.reranker)
            try:
                result = RetrievalEngine(
                    state.store,
                    state.embedder(config),
                    reranker=reranker,
                ).retrieve(
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
                    "The requested local model is not cached.",
                    409,
                    {"model": str(exc)},
                ) from exc
            except ModelRuntimeUnavailable as exc:
                raise APIError("MODEL_RUNTIME_UNAVAILABLE", str(exc), 409) from exc
            document_names = {document.id: document.filename for document in project.documents}
            by_chunk = {chunk.id: chunk for _, chunk in chunks}
            lanes = [
                {
                    "method": method,
                    "label": {
                        "VECTOR": "Vector search",
                        "BM25": "BM25 lexical",
                        "HYBRID": "Hybrid RRF",
                        "RERANKED": "Hybrid + reranker",
                    }[method],
                    "latencyMs": result.latencies_ms.get(method.lower(), 0.0),
                    "hits": [
                        {
                            "chunk": {
                                "id": item.chunk_id,
                                "documentId": item.document_id,
                                "documentName": document_names.get(
                                    item.document_id, "Unknown source"
                                ),
                                "text": item.text,
                                "ordinal": by_chunk[item.chunk_id].ordinal,
                                "tokenCount": by_chunk[item.chunk_id].token_count,
                                "startChar": by_chunk[item.chunk_id].start_char,
                                "endChar": by_chunk[item.chunk_id].end_char,
                                "heading": by_chunk[item.chunk_id].heading,
                            },
                            "nativeScore": item.native_score,
                            "rank": item.rank,
                            "candidateRank": item.candidate_rank,
                            "vectorScore": item.vector_score,
                            "bm25Score": item.bm25_score,
                            "rrfScore": item.rrf_score,
                            "crossEncoderScore": item.cross_encoder_score,
                            "rerankedRank": item.reranked_rank,
                        }
                        for item in items
                    ],
                }
                for method, items in result.rankings.items()
            ]
            context_items = result.rankings.get(result.context_method.value, [])
            response_payload = {
                "query": request.query,
                "lanes": lanes,
                "stageLatency": {
                    "vectorMs": result.latencies_ms.get("vector", 0.0),
                    "bm25Ms": result.latencies_ms.get("bm25", 0.0),
                    "fusionMs": result.latencies_ms.get("hybrid", 0.0),
                    "rerankMs": result.latencies_ms.get("reranked", 0.0),
                    "assembleMs": result.latencies_ms.get("context", 0.0),
                    "totalMs": result.latencies_ms.get("total", 0.0),
                },
                "contextMethod": result.context_method.value,
                "finalContext": result.context,
                "contextTokens": len(result.context.split()),
                "sourceDiversity": len({item.document_id for item in context_items}),
                "reranker": (
                    {
                        "model": request.reranker.model,
                        "revision": reranker.resolved_revision,
                    }
                    if request.reranker is not None and reranker is not None
                    else None
                ),
            }
            session.add(
                RetrievalRun(
                    project_id=project_id,
                    index_configuration_id=config.id,
                    query=request.query,
                    method=",".join(method.value for method in request.methods),
                    request_json=request.model_dump(mode="json", by_alias=True),
                    response_json=response_payload,
                )
            )
            return response_payload

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
                    "notes": row.notes,
                    "createdAt": row.created_at,
                }
                for row in rows
            ]
        return {"data": data}

    def add_eval_query(
        session: Any, project_id: str, request: EvaluationQueryRequest
    ) -> dict[str, Any]:
        project_or_error(session, project_id)
        chunk_ids = {str(chunk_id) for chunk_id in request.relevant_chunk_ids}
        valid_chunk_ids = set(
            session.scalars(
                select(ChunkRecord.id)
                .join(IndexConfiguration)
                .where(
                    IndexConfiguration.project_id == project_id,
                    ChunkRecord.id.in_(chunk_ids),
                )
            )
        )
        if chunk_ids != valid_chunk_ids:
            raise APIError(
                "VALIDATION_ERROR",
                "Every relevant chunk must belong to this project.",
                422,
                {"unknownChunkIds": sorted(chunk_ids - valid_chunk_ids)},
            )
        row = EvaluationQuery(
            project_id=project_id,
            query=request.query,
            dataset_version=request.dataset_version,
            notes=request.notes,
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
            "notes": row.notes,
        }

    @app.post("/api/v1/projects/{project_id}/evaluation-queries", status_code=201)
    def create_eval_query(project_id: str, request: EvaluationQueryRequest) -> dict[str, Any]:
        with state.factory.begin() as session:
            return add_eval_query(session, project_id, request)

    @app.get("/api/v1/projects/{project_id}/evaluation-queries/export")
    def export_eval_queries(project_id: str, dataset_version: int = 1) -> dict[str, Any]:
        with state.factory() as session:
            project_or_error(session, project_id)
            rows = list(
                session.scalars(
                    select(EvaluationQuery).where(
                        EvaluationQuery.project_id == project_id,
                        EvaluationQuery.dataset_version == dataset_version,
                    )
                )
            )
            return {
                "schema": "contextbench.evaluation.v1",
                "datasetVersion": dataset_version,
                "queries": [
                    {
                        "query": row.query,
                        "relevantChunkIds": [item.chunk_id for item in row.relevant_chunks],
                        "notes": row.notes,
                    }
                    for row in rows
                ],
            }

    @app.post("/api/v1/projects/{project_id}/evaluation-queries/import", status_code=201)
    def import_eval_queries(project_id: str, request: EvaluationDatasetImport) -> dict[str, Any]:
        with state.factory.begin() as session:
            created = [
                add_eval_query(
                    session,
                    project_id,
                    query.model_copy(update={"dataset_version": request.dataset_version}),
                )
                for query in request.queries
            ]
        return {"schema": "contextbench.evaluation.v1", "imported": len(created)}

    @app.get("/api/v1/projects/{project_id}/experiments")
    def list_experiments(project_id: str) -> dict[str, Any]:
        with state.factory() as session:
            project_or_error(session, project_id)
            rows = list(
                session.scalars(
                    select(Experiment)
                    .where(Experiment.project_id == project_id)
                    .order_by(Experiment.created_at.desc())
                )
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
            project_or_error(session, project_id)
            config = session.get(IndexConfiguration, str(request.index_configuration_id))
            if config is None or config.project_id != project_id:
                raise APIError("NOT_FOUND", "Index configuration not found.", 404)
            if request.method is RetrievalMethod.RERANKED and request.reranker is None:
                raise APIError(
                    "RERANKER_REQUIRED",
                    "A reranker configuration is required for a reranked experiment.",
                    422,
                )
            chunks = chunks_for_index(session, config.id)
            queries = list(
                session.scalars(
                    select(EvaluationQuery)
                    .where(
                        EvaluationQuery.project_id == project_id,
                        EvaluationQuery.dataset_version == request.dataset_version,
                    )
                    .order_by(EvaluationQuery.id)
                )
            )
            snapshot = _dataset_snapshot(session, queries, request.dataset_version)
            experiment = Experiment(
                project_id=project_id,
                name=request.name,
                config_json=request.model_dump(mode="json", by_alias=True)
                | {
                    "indexSnapshot": _index_response(config) | _chunk_snapshot(chunks),
                    "datasetSnapshot": snapshot,
                    "evaluatedQueryIds": [],
                },
                status="RUNNING",
            )
            session.add(experiment)
            session.flush()
            reranker = state.reranker(request.reranker)
            engine = RetrievalEngine(
                state.store,
                state.embedder(config),
                reranker=reranker,
            )
            evaluated_query_ids: list[str] = []
            for query_row in queries:
                judged_ids = {item.chunk_id for item in query_row.relevant_chunks}
                relevant_ids = _mapped_relevant_ids(session, query_row, chunks)
                if not relevant_ids:
                    continue
                try:
                    result = engine.retrieve(
                        config.id,
                        query_row.query,
                        chunks,
                        methods=[request.method],
                        top_k=max(request.k_values),
                        candidate_k=max(20, max(request.k_values)),
                    )
                except ModelDownloadRequired as exc:
                    raise APIError(
                        "MODEL_DOWNLOAD_REQUIRED",
                        "The reranker model is not cached.",
                        409,
                        {"model": str(exc)},
                    ) from exc
                except ModelRuntimeUnavailable as exc:
                    raise APIError("MODEL_RUNTIME_UNAVAILABLE", str(exc), 409) from exc
                ranked_ids = [
                    item.chunk_id for item in result.rankings.get(request.method.value, [])
                ]
                metrics = evaluate_rankings(
                    ranked_ids,
                    relevant_ids,
                    request.k_values,
                )
                experiment.results.append(
                    ExperimentResult(
                        evaluation_query_id=query_row.id,
                        metrics_json=metrics,
                        rankings_json={
                            "query": query_row.query,
                            "method": request.method.value,
                            "judgedChunkIds": sorted(judged_ids),
                            "resolvedRelevantChunkIds": sorted(relevant_ids),
                            "chunkIds": ranked_ids,
                            "latenciesMs": result.latencies_ms,
                            "contextMethod": result.context_method.value,
                        },
                    )
                )
                evaluated_query_ids.append(query_row.id)
            experiment.config_json = experiment.config_json | {
                "evaluatedQueryIds": evaluated_query_ids,
                "rerankerSnapshot": (
                    {
                        "model": request.reranker.model,
                        "revision": reranker.resolved_revision,
                    }
                    if request.reranker is not None and reranker is not None
                    else None
                ),
            }
            experiment.status = "COMPLETED"
            experiment.executed_at = utc_now()
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
            metrics = _aggregate_experiment_metrics(experiment)
            return {
                "id": experiment.id,
                "name": experiment.name,
                "status": experiment.status,
                "configuration": experiment.config_json,
                "executedAt": experiment.executed_at,
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
                "executedAt": experiment.executed_at.isoformat()
                if experiment.executed_at
                else None,
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
    def compare(request: ExperimentComparisonRequest) -> dict[str, Any]:
        with state.factory() as session:
            experiments: list[dict[str, Any]] = []
            compatibility: tuple[str, str, tuple[str, ...]] | None = None
            for experiment_id in request.experiment_ids:
                experiment = session.get(Experiment, str(experiment_id))
                if experiment is None:
                    raise APIError("NOT_FOUND", "Experiment not found.", 404)
                dataset_digest = str(
                    experiment.config_json.get("datasetSnapshot", {}).get("digest", "")
                )
                evaluated_ids = tuple(experiment.config_json.get("evaluatedQueryIds", []))
                signature = (experiment.project_id, dataset_digest, evaluated_ids)
                if not dataset_digest or (compatibility is not None and signature != compatibility):
                    raise APIError(
                        "INCOMPATIBLE_EXPERIMENTS",
                        "Experiments must share one project and frozen evaluation dataset.",
                        422,
                    )
                compatibility = signature
                experiments.append(
                    {
                        "name": experiment.name,
                        "configuration": {
                            "method": experiment.config_json.get("method"),
                            "indexSnapshot": experiment.config_json.get("indexSnapshot"),
                            "datasetDigest": dataset_digest,
                        },
                        "metrics": _aggregate_experiment_metrics(experiment),
                    }
                )
        try:
            return compare_experiments(experiments)
        except ValueError as exc:
            raise APIError("VALIDATION_ERROR", str(exc), 422) from exc

    @app.post("/api/v1/demo", status_code=201)
    def create_demo() -> dict[str, Any]:
        return _create_demo(state)

    @app.post("/api/v1/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        if len(request.context.split()) < request.minimum_evidence_tokens:
            return {
                "answer": "Retrieved evidence appears insufficient to answer this question.",
                "generated": False,
                "reason": "INSUFFICIENT_EVIDENCE",
            }
        ignored_terms = {"what", "when", "where", "which", "who", "why", "how", "does"}
        query_terms = {
            term
            for term in tokenize(request.question)
            if len(term) >= 3 and term not in ignored_terms
        }
        evidence_terms = set(tokenize(request.context))
        if query_terms and len(query_terms & evidence_terms) < request.minimum_query_term_matches:
            return {
                "answer": "Retrieved evidence appears insufficient to answer this question.",
                "generated": False,
                "reason": "INSUFFICIENT_QUERY_OVERLAP",
            }
        try:
            answer = OllamaClient(request.base_url, request.model).answer(
                request.question, request.context
            )
        except (ValueError, httpx.HTTPError) as exc:
            raise APIError("OLLAMA_UNAVAILABLE", str(exc), 503) from exc
        return {
            "answer": answer,
            "generated": answer != "I do not have enough cited evidence.",
            "reason": None
            if answer != "I do not have enough cited evidence."
            else "CITATIONS_MISSING",
        }

    built_frontend = (
        Path(frontend_dir)
        if frontend_dir is not None
        else Path(__file__).resolve().parents[2] / "frontend" / "dist"
    )
    if (built_frontend / "index.html").is_file():
        assets = built_frontend / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise APIError("NOT_FOUND", "API endpoint not found.", 404)
            candidate = (built_frontend / path).resolve()
            root = built_frontend.resolve()
            if candidate.is_file() and (candidate == root or root in candidate.parents):
                return FileResponse(candidate)
            return FileResponse(root / "index.html")

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
        "vectorCount": config.vector_count,
        "indexingMs": config.indexing_ms,
        "createdAt": config.created_at.isoformat(),
    }


def _create_demo(state: AppState) -> dict[str, Any]:
    corpus = [
        (
            "raft.md",
            "# Follower catch-up\nA Raft leader tracks nextIndex for every follower. After a "
            "rejected AppendEntries call, it moves that follower's nextIndex backward and "
            "retries. If the log prefix is compacted, the leader sends an InstallSnapshot call.\n\n"
            "# Election safety\nA candidate must receive votes from a majority. Voters compare "
            "the candidate's last log term and index before granting a vote.",
        ),
        (
            "storage.md",
            "# Write-ahead logging\nThe log sequence number for a page change must be durable "
            "before the dirty database page is written. This ordering lets recovery replay "
            "committed work after a crash.\n\n# MVCC cleanup\nA long-running transaction delays "
            "cleanup of older row versions and can increase disk use.",
        ),
        (
            "networking.txt",
            "TCP flow control protects the receiver by advertising a receive window. "
            "Congestion control protects the network and uses a separate congestion window. "
            "A sender is limited by the smaller of the two windows.",
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
            for document in project.documents:
                for part in chunk_text(
                    document.content,
                    __import__("contextbench.schemas", fromlist=["ChunkingConfig"]).ChunkingConfig(
                        strategy="PARAGRAPH"
                    ),
                ):
                    config.chunks.append(
                        ChunkRecord(
                            document=document,
                            ordinal=part.ordinal,
                            text=part.text,
                            token_count=part.token_count,
                            start_char=part.start_char,
                            end_char=part.end_char,
                            heading=part.heading,
                            metadata_json=part.metadata or {},
                        )
                    )
            session.flush()
            indexing_started = time.perf_counter()
            config.vector_count = RetrievalEngine(state.store, HashEmbeddingProvider()).index(
                config.id, [(chunk.document, chunk) for chunk in config.chunks]
            )
            config.indexing_ms = (time.perf_counter() - indexing_started) * 1000
            config.status = "READY"
            config_reused = False
        else:
            config_reused = True
        judgments = [
            (
                "What happens when a Raft follower falls behind the leader?",
                "moves that follower's nextIndex backward",
            ),
            (
                "Why must a database write the log before a dirty page?",
                "log sequence number for a page change must be durable",
            ),
            (
                "How does TCP keep a slow receiver from being overwhelmed?",
                "TCP flow control protects the receiver",
            ),
        ]
        for question, marker in judgments:
            existing = session.scalar(
                select(EvaluationQuery).where(
                    EvaluationQuery.project_id == project.id,
                    EvaluationQuery.query == question,
                    EvaluationQuery.dataset_version == 1,
                )
            )
            if existing is not None:
                continue
            relevant = next((chunk for chunk in config.chunks if marker in chunk.text), None)
            if relevant is not None:
                query = EvaluationQuery(
                    project_id=project.id,
                    query=question,
                    notes="Deterministic built-in relevance judgment",
                    dataset_version=1,
                )
                query.relevant_chunks = [RelevantChunk(chunk_id=relevant.id)]
                session.add(query)
        return {
            "projectId": project.id,
            "indexConfigurationId": config.id,
            "reused": project_reused and config_reused,
        }
