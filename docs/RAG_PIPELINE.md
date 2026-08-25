# RAG pipeline

ContextBench measures retrieval before optional generation.

1. Extraction reads allowed local file types and records page boundaries when a PDF has a text
   layer.
2. Normalization repairs safe whitespace problems and maintains a mapping to source character
   positions.
3. A selected chunker produces chunks with document, page, heading, character range, token range,
   and sequence index.
4. A local embedding provider creates dense vectors. Qdrant stores them with filter payloads.
5. BM25 builds a lexical corpus from the same frozen chunk set.
6. A query can run vector search, BM25, or both. Reciprocal rank fusion combines ranks.
7. An optional cross-encoder reranks a bounded candidate set.
8. The context builder applies a token limit in rank order and prints the exact selected text.
9. Evaluation compares returned chunk IDs with saved relevance judgments.
10. Ollama can receive the question and selected evidence after retrieval is complete.

Each retrieval response reports stage latency. These values describe that local run. They are not
quality scores and are not portable benchmark claims.

