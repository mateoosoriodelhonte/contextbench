# RAG pipeline

ContextBench measures retrieval before optional generation.

1. Extraction reads allowed local file types and records page boundaries when a PDF has a text
   layer.
2. Normalization applies Unicode NFKC, normalizes line endings, removes control characters, and
   collapses horizontal whitespace while retaining paragraph breaks.
3. A selected chunker produces chunks with document, page when known, heading when known,
   normalized character range, token count, and sequence index.
4. A local embedding provider creates dense vectors. Qdrant stores them with filter payloads.
5. BM25 builds a lexical corpus from the same frozen chunk set.
6. A query can run vector search, BM25, or both. Reciprocal rank fusion combines ranks.
7. An optional cross-encoder reranks a bounded candidate set.
8. The context builder selects the highest requested stage in the order reranked, hybrid, vector,
   then BM25. It applies a token limit, adds numbered source headers, and quotes untrusted evidence
   lines so document text cannot create a source label.
9. Evaluation maps saved source-span judgments to the selected index and compares returned chunk
   IDs with the resolved relevant IDs.
10. Ollama can receive the question and selected evidence after retrieval is complete.

The context token count is a whitespace-based estimate. It supports deterministic trimming and
does not claim to match a model-specific tokenizer.

Each retrieval response reports stage latency. These values describe that local run. They are not
quality scores and are not portable benchmark claims.
