import { expect, type Page } from "@playwright/test";

export class WorkbenchPage {
  constructor(readonly page: Page) {}

  async createProject(name: string): Promise<void> {
    await this.page.goto("/projects");
    await this.page.getByLabel("Project name").fill(name);
    await this.page
      .getByLabel("Description")
      .fill("Browser-tested local retrieval corpus");
    await this.page.getByRole("button", { name: "Create project" }).click();
    await expect(this.page.getByRole("heading", { name })).toBeVisible();
  }

  async addDocument(): Promise<void> {
    await this.page.getByRole("link", { name: "Documents" }).click();
    await this.page.getByLabel("Document").setInputFiles({
      name: "raft-browser.md",
      mimeType: "text/markdown",
      buffer: Buffer.from(
        "# Catch-up\nA follower catches up when the leader retries AppendEntries from an earlier nextIndex.\n\n" +
          "# Snapshot\nA leader sends InstallSnapshot when the required log prefix was compacted.",
      ),
    });
    await this.page.getByLabel(/Tags/).fill("consensus, browser");
    await this.page.getByRole("button", { name: "Add document" }).click();
    await expect(this.page.getByText("raft-browser.md")).toBeVisible();
  }

  async buildIndex(): Promise<void> {
    await this.page.getByRole("link", { name: "Indexes" }).click();
    await this.page.getByLabel("Index name").fill("local-hash-heading");
    await this.page.getByLabel("Chunking").selectOption("HEADING");
    await this.page.getByRole("button", { name: "Build index" }).click();
    await expect(
      this.page.getByRole("heading", { name: "local-hash-heading" }),
    ).toBeVisible();
    await expect(this.page.getByText("ACTIVE")).toBeVisible();
  }

  async retrieveAndJudge(): Promise<void> {
    await this.page.getByRole("link", { name: /Query debugger/ }).click();
    await this.page
      .getByLabel("Query")
      .fill("How does a follower catch up after compaction?");
    await this.page.getByRole("button", { name: "Run retrieval" }).click();
    await expect(
      this.page.getByRole("heading", { name: "Vector search" }),
    ).toBeVisible();
    await expect(
      this.page.getByRole("heading", { name: "BM25 lexical" }),
    ).toBeVisible();
    await expect(
      this.page.getByRole("heading", { name: "Hybrid RRF" }),
    ).toBeVisible();
    await this.page
      .getByRole("button", { name: "Inspect chunk" })
      .first()
      .click();
    await expect(this.page.getByText(/Characters/).first()).toBeVisible();
    await this.page
      .getByRole("checkbox", { name: /Mark .* relevant/ })
      .first()
      .check();
    await this.page
      .getByRole("button", { name: /Save 1 marked chunk/ })
      .click();
    await expect(
      this.page.getByText(/Saved evaluation query with 1 relevant chunk/),
    ).toBeVisible();
  }

  async runExperiment(name: string, method: "HYBRID" | "BM25"): Promise<void> {
    await this.page.getByRole("link", { name: "Experiments" }).click();
    await this.page.getByLabel("Experiment name").fill(name);
    await this.page.getByLabel("Method").selectOption(method);
    await this.page.getByRole("button", { name: "Run experiment" }).click();
    await expect(this.page.getByText(name, { exact: true })).toBeVisible();
    await expect(
      this.page.getByText("recall@1", { exact: true }),
    ).toBeVisible();
  }

  async compareExperiments(): Promise<void> {
    for (const checkbox of await this.page
      .getByRole("checkbox", { name: /Select browser-/ })
      .all()) {
      await checkbox.check();
    }
    await this.page
      .getByRole("link", { name: /Compare selected \(2\)/ })
      .click();
    await this.page.getByRole("button", { name: "Compare runs" }).click();
    await expect(
      this.page.getByRole("columnheader", { name: "browser-hybrid" }),
    ).toBeVisible();
    await expect(
      this.page.getByRole("columnheader", { name: "browser-bm25" }),
    ).toBeVisible();
    await expect(this.page.getByText("ndcg@5", { exact: true })).toBeVisible();
  }
}
