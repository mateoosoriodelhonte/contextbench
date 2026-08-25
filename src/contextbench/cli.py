"""Typer command line interface for local ContextBench workflows."""

# Typer uses declarative defaults in function signatures by design.
# ruff: noqa: B008

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from click import ClickException

from .api import _chunk_metadata, create_app
from .benchmarking import run_benchmark
from .chunking import chunk_text
from .config import DEFAULT_DATABASE_PATH, DEFAULT_VECTOR_PATH
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
DEFAULT_DB = DEFAULT_DATABASE_PATH
DEFAULT_VECTORS = DEFAULT_VECTOR_PATH


@project_app.command("create")
def project_create(
    name: str,
    description: str | None = None,
    db: Path = typer.Option(DEFAULT_DB, "--db"),
) -> None:
    factory = create_session_factory(db)
    with factory.begin() as session:
        row = Project(name=name, description=description)
        session.add(row)
        session.flush()
        typer.echo(row.id)


@app.command("ingest")
def ingest(project_id: str, path: Path, db: Path = typer.Option(DEFAULT_DB, "--db")) -> None:
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
            metadata_json=document.metadata,
        )
        session.add(row)
        session.flush()
        typer.echo(json.dumps({"documentId": row.id, "sha256": row.sha256}))


@app.command("index")
def index(
    project_id: str,
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    vectors: Path = typer.Option(DEFAULT_VECTORS, "--vectors"),
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
            for part in chunk_text(
                doc.content, ChunkingConfig(chunk_size=chunk_size, overlap=overlap)
            ):
                chunk = ChunkRecord(
                    document=doc,
                    ordinal=part.ordinal,
                    text=part.text,
                    token_count=part.token_count,
                    start_char=part.start_char,
                    end_char=part.end_char,
                    heading=part.heading,
                    metadata_json=_chunk_metadata(doc, part.start_char, part.metadata or {}),
                )
                config.chunks.append(chunk)
                chunks.append((doc, chunk))
        session.flush()
        engine = RetrievalEngine(LocalVectorStore(vectors), HashEmbeddingProvider())
        indexing_started = time.perf_counter()
        count = engine.index(config.id, chunks)
        config.vector_count = count
        config.indexing_ms = (time.perf_counter() - indexing_started) * 1000
        config.status = "READY"
        typer.echo(json.dumps({"indexConfigurationId": config.id, "chunkCount": count}))


@app.command("query")
def query(
    project_id: str,
    index_configuration_id: str,
    text: str,
    method: RetrievalMethod = RetrievalMethod.HYBRID,
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    vectors: Path = typer.Option(DEFAULT_VECTORS, "--vectors"),
) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(db, vectors), base_url="http://localhost")
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
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    vectors: Path = typer.Option(DEFAULT_VECTORS, "--vectors"),
    json_output: bool = typer.Option(False, "--json", help="Print the stable JSON schema."),
) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(db, vectors), base_url="http://localhost")
    response = client.post(
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
    experiment = response.json()
    detail = client.get(f"/api/v1/experiments/{experiment['id']}")
    if detail.is_error:
        raise ClickException(detail.text)
    if json_output:
        exported = client.get(f"/api/v1/experiments/{experiment['id']}/export")
        if exported.is_error:
            raise ClickException(exported.text)
        typer.echo(json.dumps(exported.json(), indent=2, default=str))
    else:
        typer.echo(f"{experiment['name']}: {experiment['resultCount']} evaluated queries")
        for name, value in sorted(detail.json()["metrics"].items()):
            typer.echo(f"  {name}: {value:.4f}")


@app.command("compare")
def compare(
    experiment_ids: list[str] = typer.Argument(...),
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    vectors: Path = typer.Option(DEFAULT_VECTORS, "--vectors"),
) -> None:
    from fastapi.testclient import TestClient

    response = TestClient(create_app(db, vectors), base_url="http://localhost").post(
        "/api/v1/experiments/compare", json={"experimentIds": experiment_ids}
    )
    if response.is_error:
        raise ClickException(response.text)
    typer.echo(json.dumps(response.json(), indent=2))


@app.command("demo")
def demo(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    vectors: Path = typer.Option(DEFAULT_VECTORS, "--vectors"),
) -> None:
    from .demo import setup_demo

    typer.echo(json.dumps(setup_demo(db, vectors), indent=2))


@app.command("benchmark")
def benchmark(
    corpus_size: Annotated[int, typer.Option(min=10, max=10000)] = 200,
    query_count: Annotated[int, typer.Option(min=1, max=1000)] = 20,
) -> None:
    """Run the real deterministic local pipeline and print versioned JSON."""
    typer.echo(json.dumps(run_benchmark(corpus_size, query_count), indent=2))


@app.command("generate")
def generate(
    question: str,
    context_file: Annotated[Path, typer.Option("--context", exists=True, dir_okay=False)],
    model: str = "llama3.2",
) -> None:
    """Ask an existing local Ollama model to answer from cited context."""
    from .ollama import OllamaClient

    answer = OllamaClient(model=model).answer(question, context_file.read_text(encoding="utf-8"))
    typer.echo(answer)


@app.command("serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    vectors: Path = typer.Option(DEFAULT_VECTORS, "--vectors"),
    frontend: Path | None = typer.Option(None, "--frontend", exists=True, file_okay=False),
) -> None:
    import uvicorn

    uvicorn.run(create_app(db, vectors, frontend), host=host, port=port)
