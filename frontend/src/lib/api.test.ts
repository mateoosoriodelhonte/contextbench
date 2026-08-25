import { beforeEach, describe, expect, it, vi } from "vitest";
import { ContextBenchApi } from "./api";

describe("ContextBenchApi", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("sends a retrieval request with the API contract fields", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ data: { id: "run-1" } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const api = new ContextBenchApi("http://localhost:8000");
    await api.retrieve("project-1", {
      query: "raft follower",
      indexConfigurationId: "index-1",
      topK: 5,
      candidateK: 20,
      methods: ["VECTOR", "BM25", "HYBRID", "RERANKED"],
      filters: { documentIds: [], sourceTypes: [], tags: [] },
      maxContextTokens: 1200,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/projects/project-1/retrieve",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          query: "raft follower",
          indexConfigurationId: "index-1",
          topK: 5,
          candidateK: 20,
          methods: ["VECTOR", "BM25", "HYBRID", "RERANKED"],
          filters: { documentIds: [], sourceTypes: [], tags: [] },
          maxContextTokens: 1200,
        }),
      }),
    );
  });

  it("surfaces contract errors without exposing raw response internals", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          error: {
            code: "VALIDATION_ERROR",
            message: "The query is required.",
          },
        }),
        {
          status: 422,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    const api = new ContextBenchApi("http://localhost:8000");
    await expect(api.listProjects()).rejects.toThrow("The query is required.");
  });
});
