# Changelog

This project follows Semantic Versioning. Dates use ISO 8601.

## 1.0.0 - 2026-08-25

### Added

- Local TXT, Markdown, PDF, and source-text ingestion with provenance.
- Fixed-token, paragraph-aware, and heading-aware chunking configurations.
- Local Sentence Transformers embeddings and persisted Qdrant vector search.
- BM25 retrieval, reciprocal rank fusion, and optional cross-encoder reranking.
- Retrieval datasets, standard IR metrics, frozen experiments, and comparison views.
- Source-span relevance mapping for valid comparisons across chunking configurations.
- FastAPI, SolidJS workbench, CLI, versioned JSON export, deterministic demo, and optional Ollama
  generation.

### Security

- Loopback host validation, upload and PDF extraction limits, safe text rendering, quoted evidence,
  citation validation, prompt boundary rules, dependency audits, and no telemetry.
