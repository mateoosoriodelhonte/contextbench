import type {
  Chunk,
  Document,
  EvaluationQuery,
  Experiment,
  ExperimentComparison,
  IndexConfiguration,
  ListResponse,
  Project,
  ProjectOverview,
  RetrievalRequest,
  RetrievalResponse,
} from "./types";

export class ContextBenchApiError extends Error {
  readonly code: string;
  readonly details: Record<string, unknown>;

  constructor(message: string, code = "REQUEST_FAILED", details: Record<string, unknown> = {}) {
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
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
    const payload = (await response.json()) as T & { error?: { code?: string; message?: string; details?: Record<string, unknown> } };
    if (!response.ok) {
      throw new ContextBenchApiError(
        payload.error?.message ?? `Request failed with status ${response.status}.`,
        payload.error?.code,
        payload.error?.details,
      );
    }
    return payload;
  }

  listProjects(): Promise<ListResponse<Project>> {
    return this.request("/api/v1/projects");
  }

  getProject(projectId: string): Promise<ProjectOverview> {
    return this.request(`/api/v1/projects/${encodeURIComponent(projectId)}`);
  }

  listDocuments(projectId: string): Promise<ListResponse<Document>> {
    return this.request(`/api/v1/projects/${encodeURIComponent(projectId)}/documents`);
  }

  listIndexes(projectId: string): Promise<ListResponse<IndexConfiguration>> {
    return this.request(`/api/v1/projects/${encodeURIComponent(projectId)}/indexes`);
  }

  listChunks(documentId: string): Promise<ListResponse<Chunk>> {
    return this.request(`/api/v1/documents/${encodeURIComponent(documentId)}/chunks`);
  }

  retrieve(projectId: string, request: RetrievalRequest): Promise<RetrievalResponse> {
    return this.request(`/api/v1/projects/${encodeURIComponent(projectId)}/retrieve`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  listEvaluationQueries(projectId: string): Promise<ListResponse<EvaluationQuery>> {
    return this.request(`/api/v1/projects/${encodeURIComponent(projectId)}/evaluation-queries`);
  }

  listExperiments(projectId: string): Promise<ListResponse<Experiment>> {
    return this.request(`/api/v1/projects/${encodeURIComponent(projectId)}/experiments`);
  }

  compareExperiments(experimentIds: string[]): Promise<ExperimentComparison> {
    return this.request("/api/v1/experiments/compare", {
      method: "POST",
      body: JSON.stringify({ experimentIds }),
    });
  }
}
