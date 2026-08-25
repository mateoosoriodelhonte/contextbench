# Privacy

ContextBench keeps project data on the machine where it runs. It sends no telemetry, analytics,
crash reports, documents, queries, chunks, vectors, or metrics to a ContextBench service. No such
service exists.

SQLite records and Qdrant collections live under the configured data directory. Sentence
Transformers models live in the local Hugging Face cache. Installing or downloading a model
contacts the package or model host; ContextBench requires explicit approval before the first model
download. After the files are cached, embedding and reranking run locally.

Optional Ollama use sends the selected question and context only to a configured
`127.0.0.1` or `localhost` HTTP endpoint. ContextBench rejects a remote Ollama URL.

Users can remove the `.contextbench/` directory when the application is stopped to delete all
project records and rebuildable vector collections. Backups, filesystem snapshots, and model
caches are separate.
