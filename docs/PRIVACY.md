# Privacy

ContextBench keeps project data on the machine where it runs. It sends no telemetry, analytics,
crash reports, documents, queries, chunks, vectors, or metrics to a ContextBench service. No such
service exists.

SQLite records and Qdrant collections live under the configured data directory. Sentence
Transformers models live in the local Hugging Face cache. Installing or downloading a model
contacts the package or model host; ContextBench requires explicit approval before the first model
download. After the files are cached, embedding and reranking run locally.

Optional Ollama use sends the selected question and context to the configured Ollama HTTP
endpoint. The default endpoint is loopback. A user who changes it to a remote host is responsible
for that host's storage and network behavior.

Deleting a project through the supported application action removes its SQLite records and
rebuildable Qdrant collection. Backups, filesystem snapshots, and model caches are outside that
action.

