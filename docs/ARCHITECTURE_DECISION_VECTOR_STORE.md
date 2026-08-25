# ADR 0001: persisted Qdrant local mode

Status: accepted for ContextBench V1.

ContextBench needs a real local vector index with metadata filters, deterministic setup, no
account, and no required background service. SQLite remains the durable source of truth for
application entities and experiment records.

## Decision

Use `qdrant-client` in persisted local mode. Create one collection per index configuration.
Store chunk and document identifiers plus filter metadata as payload. Use cosine distance for
normalized BGE embeddings. Keep vector scores separate from BM25 scores. Fuse ranks with
reciprocal rank fusion.

## Why

- It runs in-process and on disk without Docker, a cloud account, or a network service.
- Its collection and payload model matches multiple index configurations and metadata filters.
- The same client contract can later point at a separately managed local Qdrant server.
- Qdrant is recognizable production retrieval technology, while local mode keeps V1 setup
  simple.

## Tradeoffs

- Local mode is intended for local and smaller workloads. ContextBench will state this scope.
- Qdrant local mode does not provide server snapshot operations. ContextBench treats indexes
  as rebuildable artifacts from SQLite documents and frozen configurations.
- SQLite and Qdrant require consistency checks. Index build status changes only after a full
  vector upsert succeeds; failed builds remain inspectable and rebuildable.

## Alternatives

- LanceDB is embedded and supports typed metadata filters, but adds a larger native package and
  a second table-oriented persistence model.
- FAISS is fast and mature, but metadata filtering and persistence orchestration would be custom
  application work.
- A Qdrant Docker service provides stronger operational parity but adds a required service to a
  local-first V1.

