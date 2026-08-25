# Benchmarks

`latest.json` is one measured local run, not a portable performance claim. It records the
machine, Python version, corpus size, query count, embedding provider, and exact wall-clock
measurements. Results will change with hardware, load, corpus shape, and package versions.

Reproduce it with:

```bash
uv run contextbench benchmark --corpus-size 200 --query-count 20
```

The default benchmark uses deterministic hash embeddings so it can run without a model download.
It measures embedding throughput, Qdrant local indexing, vector retrieval, BM25, and hybrid RRF.
It reports reranking as `NOT_RUN` instead of inventing a cross-encoder measurement.
