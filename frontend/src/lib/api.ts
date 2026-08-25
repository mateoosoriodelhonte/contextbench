import type {
  Chunk,
  DemoSetup,
  Document,
  EvaluationQuery,
  Experiment,
  ExperimentComparison,
  IndexConfiguration,
  ListResponse,
  Project,
  ProjectOverview,
  RetrievalMethod,
  RetrievalRequest,
  RetrievalResponse,
} from "./types";

export class ContextBenchApiError extends Error {
  readonly code: string;
  readonly details: Record<string, unknown>;

  constructor(
    message: string,
    code = "REQUEST_FAILED",
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ContextBenchApiError";
    this.code = code;
    this.details = details;
  }
}

export class ContextBenchApi {
  private readonly baseUrl: string;

  constructor(baseUrl = "") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const isFormData =
      typeof FormData !== "undefined" && init?.body instanceof FormData;
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body && !isFormData
          ? { "Content-Type": "application/json" }
          : {}),
        ...init?.headers,
      },
    });
    const payload = (await response.json()) as T & {
      error?: {
        code?: string;
        message?: string;
        details?: Record<string, unknown>;
      };
    };
    if (!response.ok) {
      throw new ContextBenchApiError(
        payload.error?.message ??
          `Request failed with status ${response.status}.`,
        payload.error?.code,
        payload.error?.details,
      );
    }
    return payload;
  }

  listProjects(): Promise<ListResponse<Project>> {
    return this.request("/api/v1/projects");
  }

  createProject(name: string, description?: string): Promise<Project> {
    return this.request("/api/v1/projects", {
      method: "POST",
      body: JSON.stringify({ name, description: description || null }),
    });
  }

  getProject(projectId: string): Promise<ProjectOverview> {
    return this.request(`/api/v1/projects/${encodeURIComponent(projectId)}`);
  }

  listDocuments(projectId: string): Promise<ListResponse<Document>> {
    return this.request(
      `/api/v1/projects/${encodeURIComponent(projectId)}/documents`,
    );
  }

  uploadDocument(projectId: string, file: File, tags = ""): Promise<Document> {
    const body = new FormData();
    body.append("file", file);
    body.append("tags", tags);
    return this.request(
      `/api/v1/projects/${encodeURIComponent(projectId)}/documents`,
      { method: "POST", body },
    );
  }

  listIndexes(projectId: string): Promise<{ data: IndexConfiguration[] }> {
    return this.request(
      `/api/v1/projects/${encodeURIComponent(projectId)}/indexes`,
    );
  }

  createIndex(
    projectId: string,
    request: {
      name: string;
      chunking: { strategy: string; chunkSize: number; overlap: number };
      embedding: {
        provider: string;
        model: string;
        dimension: number;
        normalize: boolean;
        allowModelDownload: boolean;
      };
    },
  ): Promise<IndexConfiguration & { chunkCount: number }> {
    return this.request(
      `/api/v1/projects/${encodeURIComponent(projectId)}/indexes`,
      { method: "POST", body: JSON.stringify(request) },
    );
  }

  listChunks(
    documentId: string,
    indexConfigurationId?: string,
  ): Promise<ListResponse<Chunk>> {
    const query = indexConfigurationId
      ? `?index_configuration_id=${encodeURIComponent(indexConfigurationId)}`
      : "";
    return this.request(
      `/api/v1/documents/${encodeURIComponent(documentId)}/chunks${query}`,
    );
  }

  retrieve(
    projectId: string,
    request: RetrievalRequest,
  ): Promise<RetrievalResponse> {
    return this.request(
      `/api/v1/projects/${encodeURIComponent(projectId)}/retrieve`,
      { method: "POST", body: JSON.stringify(request) },
    );
  }

  listEvaluationQueries(
    projectId: string,
  ): Promise<{ data: EvaluationQuery[] }> {
    return this.request(
      `/api/v1/projects/${encodeURIComponent(projectId)}/evaluation-queries`,
    );
  }

  createEvaluationQuery(
    projectId: string,
    request: {
      query: string;
      relevantChunkIds: string[];
      datasetVersion: number;
      notes?: string;
    },
  ): Promise<EvaluationQuery> {
    return this.request(
      `/api/v1/projects/${encodeURIComponent(projectId)}/evaluation-queries`,
      { method: "POST", body: JSON.stringify(request) },
    );
  }

  listExperiments(projectId: string): Promise<{ data: Experiment[] }> {
    return this.request(
      `/api/v1/projects/${encodeURIComponent(projectId)}/experiments`,
    );
  }

  createExperiment(
    projectId: string,
    request: {
      name: string;
      datasetVersion: number;
      indexConfigurationId: string;
      method: RetrievalMethod;
      kValues: number[];
    },
  ): Promise<Experiment & { resultCount: number }> {
    return this.request(
      `/api/v1/projects/${encodeURIComponent(projectId)}/experiments`,
      { method: "POST", body: JSON.stringify(request) },
    );
  }

  getExperiment(experimentId: string): Promise<{
    id: string;
    name: string;
    status: string;
    metrics: Record<string, number>;
  }> {
    return this.request(
      `/api/v1/experiments/${encodeURIComponent(experimentId)}`,
    );
  }

  compareExperiments(experimentIds: string[]): Promise<ExperimentComparison> {
    return this.request("/api/v1/experiments/compare", {
      method: "POST",
      body: JSON.stringify({ experimentIds }),
    });
  }

  createDemo(): Promise<DemoSetup> {
    return this.request("/api/v1/demo", { method: "POST" });
  }
}
