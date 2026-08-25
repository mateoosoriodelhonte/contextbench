# Architecture

ContextBench separates durable records, rebuildable indexes, retrieval logic, and user
interfaces. The split keeps experiments reproducible and lets the web app and CLI call the same
Python services.

```text
Document bytes
  -> validated extraction
  -> normalization with source positions
  -> selected chunker
  -> SQLite chunk records
       |-> local embeddings -> Qdrant collection --|
       |-> BM25 corpus -----------------------------|-> RRF -> optional reranker
                                                        -> context builder
                                                        -> evaluation metrics
                                                        -> experiment record
```

SQLite stores projects, documents, chunks, index configurations, relevance judgments, retrieval
runs, experiments, and metrics. Qdrant stores vectors and filter payloads. A Qdrant collection is
a rebuildable artifact for one frozen index configuration. The configuration records chunking,
embedding identity, vector dimension, normalization, and distance.

## Why local embeddings

Local models keep source material on the machine, cost nothing per query, and make a saved
experiment repeatable. ContextBench uses a provider contract so tests can use deterministic hash
vectors while a normal index uses `BAAI/bge-small-en-v1.5`. Sentence Transformers recommends
separate query and document encoders for asymmetric search when a model supports them. See the
[Sentence Transformers semantic search guide](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html).

The locked package and model review is in [dependency selection](docs/DEPENDENCY_SELECTION.md).

## Why vector and BM25

Dense vectors can match related meaning when the query and evidence use different words. BM25
can find rare names, error codes, identifiers, and exact phrases that a dense model misses. The
two rankings keep their native scores. ContextBench combines ranks with reciprocal rank fusion
because cosine similarity and BM25 scores do not share a meaningful numeric scale.

## Why Qdrant local mode

Qdrant local mode runs in the Python process and persists to disk. It supports payload filters
without an account or another service. The same client can point to a separate local server when
a corpus outgrows embedded use. Qdrant documents local persistence and payload filtering in its
[local-mode guide](https://qdrant.tech/documentation/frameworks/langchain/) and
[quickstart](https://qdrant.tech/documentation/quickstart/). ADR 0001 records the alternatives and
tradeoffs.

## Why optional reranking

The embedding model scores each chunk independently from the query and can search a large corpus
quickly. A cross-encoder reads the query and a candidate chunk together, which costs more but can
order a short candidate list better. ContextBench can rerank a top-20 candidate set with
`cross-encoder/ms-marco-MiniLM-L6-v2`. The model card states that it was trained for passage
ranking and is intended to sort retrieved passages:
[cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

## Why evaluation does not require an LLM

A fluent answer can hide a bad retrieval set. ContextBench evaluates ranked chunk IDs against
explicit relevance judgments. Recall, precision, reciprocal rank, hit rate, and nDCG can all be
computed before any prompt is built. Optional Ollama generation sits after context selection and
does not affect those retrieval metrics.

## Trust boundaries

Document bytes, filenames, PDF structures, query text, tags, imported JSON, and optional Ollama
responses are untrusted. Extraction never executes source files. The API validates sizes and
schemas at its boundary. The UI inserts document text as text nodes. The generation prompt labels
retrieved text as evidence and states that instructions inside evidence have no authority.
