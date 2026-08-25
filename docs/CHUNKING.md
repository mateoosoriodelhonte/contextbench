# Chunking

Chunking is part of an index configuration. Changing a strategy, size, or overlap creates another
configuration; it does not erase the existing one.

## Fixed token

The fixed-token strategy walks a deterministic token sequence in windows of `chunkSize`. The next
window starts `chunkSize - overlap` tokens after the current start. The overlap must be smaller
than the chunk size. This strategy makes size comparisons easy but can split a paragraph or
heading from its body.

## Paragraph aware

The paragraph-aware strategy makes one normalized paragraph per chunk. It does not split a long
paragraph in V1. This choice makes the semantic boundary and its tradeoff easy to inspect.

## Heading aware

The heading-aware strategy treats Markdown-style headings as section metadata. It makes one chunk
from the body under each heading and copies the nearest heading into that chunk. A heading is
metadata, not an instruction to the retrieval or generation system.

## Provenance

Every chunk records the source document, zero-based chunk index, page when known, heading when
known, normalized character start/end, token count, and normalized content. The UI shows these
fields with every result.
