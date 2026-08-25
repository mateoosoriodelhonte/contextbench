# Reranking

Reranking is optional. The default path retrieves 20 candidates with vector, BM25, or hybrid
search and asks a cross-encoder to score each `(query, chunk)` pair. The final view shows both the
candidate rank and reranked rank.

The default model is `cross-encoder/ms-marco-MiniLM-L6-v2`. Its model card reports 22.7 million
parameters and Apache 2.0 licensing. It was trained for passage ranking:
[model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

Reranker scores only order candidates from that run. They are not calibrated answer confidence.
If the model is absent, retrieval and evaluation continue without reranking. ContextBench does not
download it without explicit approval.
