# Evaluation metrics

Let `R` be the set of relevant chunk IDs for one query. Let `L_k` be the first `k` returned chunk
IDs. Duplicate returned IDs are removed before scoring.

## Recall at K

```text
Recall@K = |R intersect L_k| / |R|
```

A case with no relevance judgments is not scored. It is reported as unjudged instead of receiving
a perfect or zero value.

## Precision at K

```text
Precision@K = |R intersect L_k| / k
```

The denominator is `k`, even when fewer than `k` results are returned. This makes an incomplete
result list visible.

## Hit rate

```text
HitRate@K = 1 if |R intersect L_k| > 0, else 0
```

The aggregate hit rate is the mean over judged queries.

## Mean reciprocal rank

For one query, reciprocal rank is `1 / rank` for the first relevant returned chunk and 0 when no
relevant chunk is returned. MRR is the mean reciprocal rank over judged queries.

## nDCG at K

ContextBench supports binary relevance in V1:

```text
DCG@K = sum_i=1..k relevance_i / log2(i + 1)
nDCG@K = DCG@K / IDCG@K
```

`IDCG@K` is the DCG of an ideal ranking with `min(|R|, k)` relevant chunks first. nDCG is 0 when
the returned ranking has no gain. An unjudged query is excluded from the aggregate.

The saved chunk IDs anchor judgments to normalized document spans. The same index uses the exact
judged ID. Another index maps each anchor to one representative chunk in the same document. It
chooses the chunk with the greatest covered fraction, then the earliest span and ID for stable
ties. One judgment therefore stays one relevance unit when smaller chunks create more overlaps.
The experiment stores the mapping and its `single-best-overlap-v1` policy. This is a deterministic
chunk-level approximation; V1 does not support labels that target text below the chunk level.

Metrics are macro means across judged queries. Per-query values, resolved relevance IDs, frozen
dataset digest, index digest, mapping policy, and latency stay in the experiment export so one
failure cannot be hidden by the average.
