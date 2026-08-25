import type { EvaluationQuery, Experiment, IndexConfiguration, Project, RetrievalLane, RetrievalResponse } from "./types";

const chunk = (id: string, documentName: string, text: string, ordinal: number, tokenCount: number) => ({
  id,
  documentId: `doc-${documentName.toLowerCase().replaceAll(" ", "-")}`,
  documentName,
  text,
  ordinal,
  tokenCount,
});

export const demoProject: Project = {
  id: "demo-project",
  name: "Raft paper corpus",
  description: "A deterministic local corpus for inspecting retrieval behavior.",
  documentCount: 18,
  indexCount: 3,
  updatedAt: "2026-08-25T14:12:00Z",
  status: "READY",
};

export const demoIndexes: IndexConfiguration[] = [
  { id: "index-256", name: "bge-small · 256 / 32", chunkSize: 256, overlap: 32, embeddingModel: "BAAI/bge-small-en-v1.5", dimension: 384, status: "READY", createdAt: "2026-08-20T10:00:00Z" },
  { id: "index-512", name: "bge-small · 512 / 64", chunkSize: 512, overlap: 64, embeddingModel: "BAAI/bge-small-en-v1.5", dimension: 384, status: "READY", createdAt: "2026-08-18T16:30:00Z" },
  { id: "index-tfidf", name: "BM25 lexical baseline", chunkSize: 256, overlap: 0, embeddingModel: "none", dimension: 0, status: "READY", createdAt: "2026-08-15T11:20:00Z" },
];

const chunks = [
  chunk("chunk-01", "raft.pdf", "A follower that crashes or is partitioned from the leader may fall behind the leader's log. When it rejoins, the leader repairs the follower by sending missing entries.", 12, 34),
  chunk("chunk-02", "raft.pdf", "If a follower's log is inconsistent with the leader's log, the leader removes the conflicting entry and all entries that follow it, then appends the leader's entries.", 14, 32),
  chunk("chunk-03", "raft.pdf", "The nextIndex for each follower is initialized to the leader's last log index plus one. Decrementing nextIndex retries the append until the logs agree.", 15, 29),
  chunk("chunk-04", "raft-notes.md", "Raft keeps the replicated log consistent through a term and index pair. A successful AppendEntries response advances matchIndex for the follower.", 7, 26),
];

const baseHits = chunks.map((item, index) => ({ chunk: item, nativeScore: [0.91, 0.86, 0.77, 0.64][index] ?? 0.5, rank: index + 1 }));

export const demoLanes: RetrievalLane[] = [
  { method: "VECTOR", label: "Vector search", latencyMs: 18, hits: baseHits },
  { method: "BM25", label: "BM25 lexical", latencyMs: 11, hits: baseHits.map((hit, index) => ({ ...hit, nativeScore: [12.84, 10.12, 8.09, 5.2][index] ?? 4, rank: [1, 2, 4, 3][index] ?? index + 1 })) },
  { method: "HYBRID", label: "Hybrid RRF", latencyMs: 25, hits: baseHits.map((hit, index) => ({ ...hit, nativeScore: [0.0164, 0.0161, 0.0156, 0.0149][index] ?? 0.01, rrfScore: [0.0164, 0.0161, 0.0156, 0.0149][index] ?? 0.01, rank: index + 1 })) },
  { method: "RERANKED", label: "Reranked", latencyMs: 49, hits: baseHits.map((hit, index) => ({ ...hit, nativeScore: [0.98, 0.9, 0.81, 0.62][index] ?? 0.5, crossEncoderScore: [0.98, 0.9, 0.81, 0.62][index] ?? 0.5, candidateRank: index + 1, rerankedRank: index + 1, rank: index + 1 })) },
];

export const demoRetrieval: RetrievalResponse = {
  query: "What happens when a Raft follower falls behind?",
  lanes: demoLanes,
  stageLatency: { tokenizeMs: 2, retrieveMs: 25, rerankMs: 49, assembleMs: 3, totalMs: 79 },
  finalContext: "[raft.pdf · chunk 12]\nA follower that crashes or is partitioned from the leader may fall behind the leader's log. When it rejoins, the leader repairs the follower by sending missing entries.\n\n[raft.pdf · chunk 14]\nIf a follower's log is inconsistent with the leader's log, the leader removes the conflicting entry and all entries that follow it, then appends the leader's entries.",
  contextTokens: 118,
  sourceDiversity: 2,
};

export const demoEvaluationQueries: EvaluationQuery[] = [
  { id: "eval-01", query: "What happens when a Raft follower falls behind?", relevantDocumentIds: ["doc-raft-pdf"], datasetVersion: 1, createdAt: "2026-08-22T12:00:00Z" },
  { id: "eval-02", query: "How does a leader repair a conflicting log?", relevantDocumentIds: ["doc-raft-pdf"], datasetVersion: 1, createdAt: "2026-08-22T12:01:00Z" },
  { id: "eval-03", query: "When can a candidate become leader?", relevantDocumentIds: ["doc-raft-pdf", "doc-raft-notes-md"], datasetVersion: 1, createdAt: "2026-08-22T12:02:00Z" },
];

export const demoExperiments: Experiment[] = [
  { id: "exp-hybrid-256", name: "hybrid-256-v1", method: "HYBRID", datasetVersion: 1, indexConfigurationId: "index-256", status: "COMPLETE", createdAt: "2026-08-24T09:15:00Z" },
  { id: "exp-vector-512", name: "vector-512-v1", method: "VECTOR", datasetVersion: 1, indexConfigurationId: "index-512", status: "COMPLETE", createdAt: "2026-08-24T09:10:00Z" },
];
