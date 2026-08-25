# Hybrid retrieval

ContextBench uses weighted reciprocal rank fusion (RRF). It combines positions, not raw scores:

```text
RRF(d) = sum_m weight_m / (rank_m(d) + c)
```

`d` is a chunk, `m` is a retrieval method, rank starts at 1, and `c` defaults to 60. A chunk that
appears in both rankings receives both terms. A missing chunk receives no term for that method.
The default vector and BM25 weights are both 1.0.

RRF avoids a false conversion between cosine similarity and BM25. ContextBench still returns each
native score next to its method rank. A weighted configuration changes the contribution of a
ranking in a defined way; it does not normalize the original scores.

