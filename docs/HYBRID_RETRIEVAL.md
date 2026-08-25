# Hybrid retrieval

ContextBench uses reciprocal rank fusion (RRF). It combines positions, not raw scores:

```text
RRF(d) = sum_m 1 / (rank_m(d) + c)
```

`d` is a chunk, `m` is a retrieval method, rank starts at 1, and `c` defaults to 60. A chunk that
appears in both rankings receives both terms. A missing chunk receives no term for that method.

RRF avoids a false conversion between cosine similarity and BM25. ContextBench still returns each
native score next to its method rank. V1 uses equal contribution from vector and BM25 ranks.
