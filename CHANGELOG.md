# Changelog

This project follows Semantic Versioning. Dates use ISO 8601.

## 1.0.0 - 2026-08-25

### Added

- Local TXT, Markdown, PDF, and source-text ingestion with provenance.
- Fixed-token, paragraph-aware, and heading-aware chunking configurations.
- Local Sentence Transformers embeddings and persisted Qdrant vector search.
- BM25 retrieval, reciprocal rank fusion, and optional cross-encoder reranking.
- Retrieval datasets, standard IR metrics, frozen experiments, and comparison views.
- FastAPI, SolidJS workbench, CLI, versioned JSON export, deterministic demo, and optional Ollama
  generation.

### Security

- Local-only bind, upload limits, safe text rendering, prompt boundary rules, and no telemetry.
