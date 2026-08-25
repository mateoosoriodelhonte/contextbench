"""Typer command line interface for local ContextBench workflows."""

# Typer uses declarative defaults in function signatures by design.
# ruff: noqa: B008

from __future__ import annotations

import json
from pathlib import Path

import typer
from click import ClickException

from .api import AppState, create_app
from .chunking import chunk_text
from .db import create_session_factory
from .embedding import HashEmbeddingProvider
from .ingestion import ingest_path
from .models import ChunkRecord, Document, IndexConfiguration, Project
from .retrieval import RetrievalEngine
from .schemas import ChunkingConfig, RetrievalMethod
from .vector_store import LocalVectorStore

app = typer.Typer(help="Local-first RAG retrieval evaluation workbench")
project_app = typer.Typer(help="Manage projects")
app.add_typer(project_app, name="project")


@project_app.command("create")
def project_create(
    name: str,
    description: str | None = None,
    db: Path = typer.Option(Path("contextbench.sqlite3"), "--db"),
) -> None:
    project = AppState(db, Path("contextbench-qdrant")).factory
    with project.begin() as session:
        row = Project(name=name, description=description)
        session.add(row)
        session.flush()
        typer.echo(row.id)


@app.command("ingest")
def ingest(
    project_id: str, path: Path, db: Path = typer.Option(Path("contextbench.sqlite3"), "--db")
) -> None:
    document = ingest_path(path)
    factory = create_session_factory(db)
    with factory.begin() as session:
        row = Document(
            project_id=project_id,
            filename=document.filename,
            source_type=document.source_type.value,
            content=document.text,
            sha256=document.sha256,
            tags=[],
        )
        session.add(row)
        session.flush()
        typer.echo(json.dumps({"documentId": row.id, "sha256": row.sha256}))


@app.command("index")
def index(
    project_id: str,
    db: Path = typer.Option(Path("contextbench.sqlite3"), "--db"),
    vectors: Path = typer.Option(Path("contextbench-qdrant"), "--vectors"),
    name: str = "hash",
    chunk_size: int = 256,
    overlap: int = 32,
) -> None:
    factory = create_session_factory(db)
    with factory.begin() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise typer.BadParameter("project not found")
        config = IndexConfiguration(
            project_id=project.id,
            name=name,
            chunking_json={"strategy": "FIXED_TOKEN", "chunkSize": chunk_size, "overlap": overlap},
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
        chunks: list[tuple[Document, ChunkRecord]] = []
        for doc in project.documents:
            doc.chunks.clear()
            for part in chunk_text(
                doc.content, ChunkingConfig(chunk_size=chunk_size, overlap=overlap)
            ):
                chunk = ChunkRecord(
                    ordinal=part.ordinal,
                    text=part.text,
                    token_count=part.token_count,
                    start_char=part.start_char,
                    end_char=part.end_char,
                    heading=part.heading,
                    metadata_json=part.metadata or {},
                )
                doc.chunks.append(chunk)
                chunks.append((doc, chunk))
        engine = RetrievalEngine(LocalVectorStore(vectors), HashEmbeddingProvider())
        count = engine.index(config.id, chunks)
        config.status = "READY"
        typer.echo(json.dumps({"indexConfigurationId": config.id, "chunkCount": count}))


@app.command("query")
def query(
    project_id: str,
    index_configuration_id: str,
    text: str,
    method: RetrievalMethod = RetrievalMethod.HYBRID,
    db: Path = typer.Option(Path("contextbench.sqlite3"), "--db"),
    vectors: Path = typer.Option(Path("contextbench-qdrant"), "--vectors"),
) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(db, vectors))
    response = client.post(
        f"/api/v1/projects/{project_id}/retrieve",
        json={
            "query": text,
            "indexConfigurationId": index_configuration_id,
            "methods": [method.value],
        },
    )
    if response.is_error:
        raise ClickException(response.text)
    typer.echo(json.dumps(response.json(), default=str, indent=2))


@app.command("eval")
def eval_command(
    project_id: str,
    index_configuration_id: str,
    db: Path = typer.Option(Path("contextbench.sqlite3"), "--db"),
    vectors: Path = typer.Option(Path("contextbench-qdrant"), "--vectors"),
) -> None:
    from fastapi.testclient import TestClient

    response = TestClient(create_app(db, vectors)).post(
        f"/api/v1/projects/{project_id}/experiments",
        json={
            "name": "cli-evaluation",
            "datasetVersion": 1,
            "indexConfigurationId": index_configuration_id,
            "method": "HYBRID",
            "kValues": [1, 3, 5, 10],
        },
    )
    if response.is_error:
        raise ClickException(response.text)
    typer.echo(json.dumps(response.json(), indent=2, default=str))


@app.command("compare")
def compare(
    experiment_ids: list[str] = typer.Argument(...),
    db: Path = typer.Option(Path("contextbench.sqlite3"), "--db"),
    vectors: Path = typer.Option(Path("contextbench-qdrant"), "--vectors"),
) -> None:
    from fastapi.testclient import TestClient

    response = TestClient(create_app(db, vectors)).post(
        "/api/v1/experiments/compare", json={"experimentIds": experiment_ids}
    )
    if response.is_error:
        raise ClickException(response.text)
    typer.echo(json.dumps(response.json(), indent=2))


@app.command("demo")
def demo(
    db: Path = typer.Option(Path("contextbench.sqlite3"), "--db"),
    vectors: Path = typer.Option(Path("contextbench-qdrant"), "--vectors"),
) -> None:
    from .demo import setup_demo

    typer.echo(json.dumps(setup_demo(db, vectors), indent=2))


@app.command("serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    db: Path = typer.Option(Path("contextbench.sqlite3"), "--db"),
    vectors: Path = typer.Option(Path("contextbench-qdrant"), "--vectors"),
) -> None:
    import uvicorn

    uvicorn.run(create_app(db, vectors), host=host, port=port)
