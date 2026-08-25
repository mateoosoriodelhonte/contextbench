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
  createdAt: string;
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
  sha256: string;
}

export interface IndexConfiguration {
  id: string;
  name: string;
  projectId: string;
  chunking: {
    strategy: "FIXED_TOKEN" | "PARAGRAPH" | "HEADING";
    chunkSize: number;
    overlap: number;
  };
  embedding: {
    provider: "hash" | "sentence-transformers";
    model: string;
    revision?: string;
    dimension: number;
    normalize: boolean;
    allowModelDownload: boolean;
  };
  vectorDimension: number;
  vectorCount: number;
  indexingMs: number;
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
  startChar: number;
  endChar: number;
  heading?: string;
  page?: number;
  indexConfigurationId?: string;
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
  reranker?: {
    model: string;
    allowModelDownload: boolean;
  };
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
    vectorMs: number;
    bm25Ms: number;
    fusionMs: number;
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
  relevantChunkIds: string[];
  notes?: string;
  datasetVersion: number;
  createdAt: string;
}

export interface Experiment {
  id: string;
  name: string;
  method?: RetrievalMethod;
  datasetVersion?: number;
  indexConfigurationId?: string;
  status: "QUEUED" | "RUNNING" | "COMPLETED" | "ERROR";
  createdAt: string;
}

export interface ExperimentComparison {
  schema: "contextbench.experiment.v1";
  experiments: Array<{ name: string; metrics: Record<string, number> }>;
  metricKeys: string[];
}

export interface DemoSetup {
  projectId: string;
  indexConfigurationId: string;
  reused: boolean;
}
