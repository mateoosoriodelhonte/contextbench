export type RetrievalMethod = "VECTOR" | "BM25" | "HYBRID" | "RERANKED";

export interface Pagination {
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
}

export interface ListResponse<T> {
  data: T[];
  pagination: Pagination;
}

export interface ApiErrorPayload {
  error?: {
    code?: string;
    message?: string;
    details?: Record<string, unknown>;
  };
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  documentCount: number;
  indexCount: number;
  updatedAt: string;
  status: "READY" | "INGESTING" | "ERROR";
}

export interface ProjectOverview extends Project {
  latestMetrics?: {
    recallAt5: number;
    mrrAt10: number;
    queryCount: number;
  };
}

export interface Document {
  id: string;
  filename: string;
  sourceType: string;
  tags: string[];
  chunkCount: number;
  status: "READY" | "PROCESSING" | "ERROR";
  updatedAt: string;
}

export interface IndexConfiguration {
  id: string;
  name: string;
  chunkSize: number;
  overlap: number;
  embeddingModel: string;
  dimension: number;
  status: "READY" | "BUILDING" | "ERROR";
  createdAt: string;
}

export interface Chunk {
  id: string;
  documentId: string;
  documentName: string;
  text: string;
  ordinal: number;
  tokenCount: number;
}

export interface RetrievalRequest {
  query: string;
  indexConfigurationId: string;
  topK: number;
  candidateK: number;
  methods: RetrievalMethod[];
  filters: {
    documentIds: string[];
    sourceTypes: string[];
    tags: string[];
  };
  maxContextTokens: number;
}

export interface RetrievalHit {
  chunk: Chunk;
  nativeScore: number;
  rank: number;
  candidateRank?: number;
  rrfScore?: number;
  crossEncoderScore?: number;
  rerankedRank?: number;
}

export interface RetrievalLane {
  method: RetrievalMethod;
  label: string;
  latencyMs: number;
  hits: RetrievalHit[];
}

export interface RetrievalResponse {
  query: string;
  lanes: RetrievalLane[];
  stageLatency: {
    tokenizeMs: number;
    retrieveMs: number;
    rerankMs: number;
    assembleMs: number;
    totalMs: number;
  };
  finalContext: string;
  contextTokens: number;
  sourceDiversity: number;
}

export interface EvaluationQuery {
  id: string;
  query: string;
  relevantDocumentIds: string[];
  datasetVersion: number;
  createdAt: string;
}

export interface Experiment {
  id: string;
  name: string;
  method: RetrievalMethod;
  datasetVersion: number;
  indexConfigurationId: string;
  status: "QUEUED" | "RUNNING" | "COMPLETE" | "ERROR";
  createdAt: string;
}

export interface ExperimentComparison {
  experiments: Array<{ id: string; name: string; method: RetrievalMethod }>;
  metrics: Array<{ name: string; values: number[]; higherIsBetter: boolean }>;
}
