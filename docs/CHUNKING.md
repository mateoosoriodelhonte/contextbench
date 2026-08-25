# Chunking

Chunking is part of an index configuration. Changing a strategy, size, or overlap creates another
configuration; it does not erase the existing one.

## Fixed token

The fixed-token strategy walks a deterministic token sequence in windows of `chunkSize`. The next
window starts `chunkSize - overlap` tokens after the current start. The overlap must be smaller
than the chunk size. This strategy makes size comparisons easy but can split a paragraph or
heading from its body.

## Paragraph aware

The paragraph-aware strategy adds complete paragraphs until the next paragraph would exceed the
target. One paragraph can exceed the target when splitting it would lose the chosen semantic
boundary. Source ranges include the whitespace between retained paragraphs.

## Heading aware

The heading-aware strategy treats Markdown-style headings as section metadata. It groups section
text to the target size and copies the nearest heading path into each chunk. A heading is metadata,
not an instruction to the retrieval or generation system.

## Provenance

Every chunk records the source document, zero-based chunk index, page when known, heading when
known, original character start/end, estimated token start/end, and normalized content. The UI
shows these fields with every result.

