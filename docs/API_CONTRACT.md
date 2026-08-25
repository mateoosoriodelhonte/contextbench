# ContextBench API contract

The local API is versioned under `/api/v1`. JSON fields use `camelCase`. UUID strings are
opaque identifiers. All timestamps use UTC ISO 8601. Paginated project, document, and chunk
lists use:

```json
{
  "data": [],
  "pagination": {
    "page": 1,
    "pageSize": 50,
    "totalItems": 0,
    "totalPages": 0
  }
}
```

Index, evaluation-query, and experiment lists use `{"data": []}` because V1 does not paginate
those smaller project-scoped collections.

Errors use one shape:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request is invalid.",
    "details": {}
  }
}
```

## Resources

| Method        | Path                                                     | Purpose                                                    |
| ------------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| `GET`         | `/health`                                                | Process and storage health.                                |
| `GET`, `POST` | `/api/v1/projects`                                       | List or create projects.                                   |
| `GET`         | `/api/v1/projects/{projectId}`                           | Project dashboard and latest metrics.                      |
| `GET`, `POST` | `/api/v1/projects/{projectId}/documents`                 | List documents or ingest one multipart file.               |
| `GET`         | `/api/v1/documents/{documentId}/chunks`                  | Paginated, inspectable chunks.                             |
| `GET`, `POST` | `/api/v1/projects/{projectId}/indexes`                   | List or build an index configuration.                      |
| `POST`        | `/api/v1/projects/{projectId}/retrieve`                  | Run vector, BM25, hybrid, and optional reranked retrieval. |
| `GET`, `POST` | `/api/v1/projects/{projectId}/evaluation-queries`        | List or create relevance judgments.                        |
| `GET`         | `/api/v1/projects/{projectId}/evaluation-queries/export` | Export one dataset version.                                |
| `POST`        | `/api/v1/projects/{projectId}/evaluation-queries/import` | Import a versioned dataset.                                |
| `GET`, `POST` | `/api/v1/projects/{projectId}/experiments`               | List or execute experiments.                               |
| `GET`         | `/api/v1/experiments/{experimentId}`                     | Frozen configuration and per-query results.                |
| `POST`        | `/api/v1/experiments/compare`                            | Compare two or more real experiment results.               |
| `POST`        | `/api/v1/demo`                                           | Create or reuse the deterministic demo project.            |
| `POST`        | `/api/v1/generate`                                       | Ask an existing loopback Ollama model for a cited answer.  |

## Retrieval request

```json
{
  "query": "What happens when a Raft follower falls behind?",
  "indexConfigurationId": "uuid",
  "topK": 5,
  "candidateK": 20,
  "methods": ["VECTOR", "BM25", "HYBRID", "RERANKED"],
  "filters": {
    "documentIds": [],
    "sourceTypes": [],
    "tags": []
  },
  "maxContextTokens": 1200,
  "reranker": {
    "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "revision": null,
    "allowModelDownload": false
  }
}
```

Each method returns its own native score. Hybrid and reranked hits also retain available vector,
BM25, and RRF scores. Reranked results retain the candidate retrieval rank and add a cross-encoder
score and reranked rank. The response names `contextMethod`, reports each latency stage, and
returns the exact deterministically trimmed final context.

## Index request

```json
{
  "name": "index-256-overlap32",
  "chunking": {
    "strategy": "FIXED_TOKEN",
    "chunkSize": 256,
    "overlap": 32
  },
  "embedding": {
    "provider": "sentence-transformers",
    "model": "BAAI/bge-small-en-v1.5",
    "revision": null,
    "dimension": 384,
    "normalize": true,
    "allowModelDownload": false
  }
}
```

The server returns `MODEL_DOWNLOAD_REQUIRED` before a model download unless the request explicitly
allows it. The stored configuration clears that one-time consent after a successful build. An
index freezes provider, model, resolved immutable revision, actual vector dimension, normalization,
chunking, and vector distance. Changing one creates a new index.

## Experiment request

```json
{
  "name": "hybrid-256-v1",
  "datasetVersion": 1,
  "indexConfigurationId": "uuid",
  "method": "HYBRID",
  "reranker": null,
  "kValues": [1, 3, 5, 10]
}
```

An experiment is immutable after execution. It stores dataset and index digests, the complete
dataset snapshot, mapped relevance IDs, rankings, metric inputs, resolved reranker revision, and
measured latency. Comparison requires one project, one frozen dataset digest, and the same set of
evaluated queries. It also requires the same relevance mapping policy. Its export schema is
`contextbench.experiment.v1`.
