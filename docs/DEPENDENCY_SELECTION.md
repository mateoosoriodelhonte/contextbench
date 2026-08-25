# Dependency selection

Checked on 2026-08-25. `uv.lock` and `frontend/package-lock.json` are the install source of truth.
This note records why the main packages were selected.

| Area                     | Locked choice                             | Reason                                                                                                                                                      |
| ------------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| API                      | FastAPI 0.141.1                           | Typed request and response models, generated OpenAPI, and direct local ASGI use. [PyPI release](https://pypi.org/project/fastapi/)                          |
| Data models              | Pydantic 2.13.4                           | Strict boundary validation and camel-case aliases for the JSON contract. [PyPI release](https://pypi.org/project/pydantic/)                                 |
| Relational storage       | SQLAlchemy 2.0.52 with SQLite             | Versioned entities need transactions and explicit relationships. SQLite keeps setup local. [SQLAlchemy releases](https://pypi.org/project/SQLAlchemy/)      |
| Vector storage           | qdrant-client 1.19.0 in local mode        | On-disk vector search and payload filters without Docker or an account. [Qdrant local guide](https://qdrant.tech/documentation/frameworks/langchain/)       |
| Embeddings and reranking | Sentence Transformers 6.0.0               | One maintained local library supports bi-encoder embeddings and cross-encoder ranking. [PyPI release](https://pypi.org/project/sentence-transformers/)      |
| PDF extraction           | pypdf 6.16.2                              | It reads text layers without adding OCR or a system service. [PyPI release](https://pypi.org/project/pypdf/)                                                |
| Frontend                 | SolidJS 1.9.15                            | The stable 1.x line has fine-grained reactivity and current browser support. [npm package](https://www.npmjs.com/package/solid-js)                          |
| Build                    | Vite 8.2.2 with vite-plugin-solid 2.11.14 | The stable Solid Vite plugin supports Vite 8. [plugin package](https://www.npmjs.com/package/vite-plugin-solid)                                             |
| Frontend tests           | Vitest 4.1.11 and Playwright 1.62.1       | Solid recommends Vitest with its testing library; Playwright runs the complete browser path. [Solid testing guide](https://docs.solidjs.com/guides/testing) |

## Vector store comparison

Qdrant local mode was chosen over LanceDB and FAISS for V1. Qdrant has explicit collections,
payload filters, persisted local use, and a client that can later connect to a separate server.
LanceDB also works in process and has typed metadata filters, but it introduces another
table-oriented data store beside SQLite. Its current wheel is also a larger native dependency.
FAISS has mature nearest-neighbor search but leaves metadata filtering, persistence, and
collection lifecycle to ContextBench.

Local mode fits a single-user laptop workbench. It is not a claim that the embedded engine should
serve a large multi-user deployment. A separately managed Qdrant instance is the documented next
step for that use case.

## Model defaults

`BAAI/bge-small-en-v1.5` has 33.4 million parameters, a 384-value vector, and a 512-token sequence
limit. The model card licenses it under MIT and reports retrieval results for the BGE 1.5 family:
[BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5).

`cross-encoder/ms-marco-MiniLM-L6-v2` has 22.7 million parameters and an Apache 2.0 license. Its
model card describes passage reranking use after first-stage retrieval:
[cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

Neither model is required for unit tests or the deterministic demo. ContextBench checks the cache
and asks before it downloads model files.
