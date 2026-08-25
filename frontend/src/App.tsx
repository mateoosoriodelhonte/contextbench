import { A, Route, Router, useLocation, useNavigate } from "@solidjs/router";
import {
  For,
  Match,
  Show,
  Switch,
  createEffect,
  createMemo,
  createResource,
  createSignal,
  type JSX,
  type Accessor,
  type Setter,
} from "solid-js";
import {
  ArrowUpRight,
  ChevronDown,
  ChevronRight,
  Database,
  FileText,
  Gauge,
  Layers,
  Menu,
  Play,
  Plus,
  Settings,
  Sliders,
  Spark,
  Terminal,
} from "./components/Icons";
import { ContextBenchApi } from "./lib/api";
import type {
  ExperimentComparison,
  RetrievalHit,
  RetrievalLane,
  RetrievalMethod,
  RetrievalResponse,
} from "./lib/types";
import "./styles.css";

const api = new ContextBenchApi();

const navItems = [
  { label: "Overview", suffix: "", icon: Gauge },
  { label: "Documents", suffix: "/documents", icon: FileText },
  { label: "Indexes", suffix: "/indexes", icon: Layers },
  { label: "Query debugger", suffix: "/query", icon: Terminal, primary: true },
  { label: "Evaluation set", suffix: "/evaluation", icon: Database },
  { label: "Experiments", suffix: "/experiments", icon: Sliders },
  { label: "Comparison", suffix: "/comparison", icon: ArrowUpRight },
];

const methodTone: Record<RetrievalMethod, string> = {
  VECTOR: "vector",
  BM25: "bm25",
  HYBRID: "hybrid",
  RERANKED: "reranked",
};

interface Workspace {
  projectId: Accessor<string>;
  indexId: Accessor<string>;
  revision: Accessor<number>;
  selectedExperiments: Accessor<string[]>;
  setProjectId: (id: string) => void;
  setIndexId: (id: string) => void;
  setSelectedExperiments: Setter<string[]>;
  refresh: () => void;
}

function saved(key: string): string {
  try {
    return typeof window === "undefined"
      ? ""
      : (window.localStorage.getItem(key) ?? "");
  } catch {
    return "";
  }
}

export function App() {
  return (
    <Router>
      <Route path="*" component={Workbench} />
    </Router>
  );
}

function Workbench() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = createSignal(false);
  const [projectId, setProjectIdSignal] = createSignal(
    saved("contextbench.projectId"),
  );
  const [indexId, setIndexIdSignal] = createSignal(
    saved("contextbench.indexId"),
  );
  const [revision, setRevision] = createSignal(0);
  const [selectedExperiments, setSelectedExperiments] = createSignal<string[]>(
    [],
  );
  const setProjectId = (id: string) => {
    setProjectIdSignal(id);
    try {
      window.localStorage.setItem("contextbench.projectId", id);
    } catch {
      /* storage is optional */
    }
  };
  const setIndexId = (id: string) => {
    setIndexIdSignal(id);
    try {
      window.localStorage.setItem("contextbench.indexId", id);
    } catch {
      /* storage is optional */
    }
  };
  const workspace: Workspace = {
    projectId,
    indexId,
    revision,
    selectedExperiments,
    setProjectId,
    setIndexId,
    setSelectedExperiments,
    refresh: () => setRevision((value) => value + 1),
  };
  const [project] = createResource(
    () => [projectId(), revision()] as const,
    ([id]) => (id ? api.getProject(id) : undefined),
  );
  const page = createMemo(() => {
    const path = location.pathname;
    if (path === "/projects") return "projects";
    if (path === "/") return "query";
    for (const item of navItems.slice(1)) {
      if (path.endsWith(item.suffix))
        return item.label.toLowerCase().replace(" ", "-");
    }
    if (path.endsWith("/settings")) return "settings";
    return "overview";
  });
  const projectBase = () => `/projects/${projectId() || "local"}`;

  return (
    <div class="app-shell">
      <a class="skip-link" href="#main-content">
        Skip to content
      </a>
      <aside
        classList={{ sidebar: true, "sidebar--open": mobileOpen() }}
        aria-label="Workbench navigation"
      >
        <div class="brand-lockup">
          <span class="brand-mark">CB</span>
          <span>ContextBench</span>
          <span class="brand-version">v1</span>
        </div>
        <div class="sidebar-project">
          <span class="eyebrow">LOCAL PROJECT</span>
          <A href="/projects" class="project-switcher">
            <span class="status-dot" aria-hidden="true" />
            {project()?.name ?? "Choose a project"}
            <ChevronRight size={14} />
          </A>
        </div>
        <nav class="primary-nav" aria-label="Project sections">
          <For each={navItems}>
            {(item) => (
              <A
                href={`${projectBase()}${item.suffix}`}
                activeClass="nav-item--active"
                classList={{
                  "nav-item": true,
                  "nav-item--primary": item.primary ?? false,
                }}
                onClick={() => setMobileOpen(false)}
              >
                <item.icon size={16} />
                <span>{item.label}</span>
                <Show when={item.primary}>
                  <span class="nav-key">⌘ K</span>
                </Show>
              </A>
            )}
          </For>
        </nav>
        <div class="sidebar-bottom">
          <A
            href={`${projectBase()}/settings`}
            activeClass="nav-item--active"
            class="nav-item"
          >
            <Settings size={16} />
            <span>Settings</span>
          </A>
          <div class="connection">
            <span class="status-dot" aria-hidden="true" />
            Local API <span class="mono">:8000</span>
          </div>
        </div>
      </aside>
      <div class="main-column">
        <header class="topbar">
          <button
            class="icon-button menu-button"
            type="button"
            aria-label="Toggle navigation"
            onClick={() => setMobileOpen((open) => !open)}
          >
            <Menu />
          </button>
          <div class="breadcrumbs">
            <A href="/projects">Projects</A>
            <ChevronRight size={13} />
            <strong>{project()?.name ?? "Local workspace"}</strong>
          </div>
          <div class="topbar-actions">
            <span class="env-badge">LOCAL ONLY</span>
            <span class="avatar" aria-label="No account required">
              $0
            </span>
          </div>
        </header>
        <main id="main-content">
          <Switch fallback={<OverviewView workspace={workspace} />}>
            <Match when={page() === "projects"}>
              <ProjectsView workspace={workspace} />
            </Match>
            <Match when={page() === "overview"}>
              <OverviewView workspace={workspace} />
            </Match>
            <Match when={page() === "documents"}>
              <DocumentsView workspace={workspace} />
            </Match>
            <Match when={page() === "indexes"}>
              <IndexesView workspace={workspace} />
            </Match>
            <Match when={page() === "query-debugger" || page() === "query"}>
              <QueryDebugger workspace={workspace} />
            </Match>
            <Match when={page() === "evaluation-set"}>
              <EvaluationView workspace={workspace} />
            </Match>
            <Match when={page() === "experiments"}>
              <ExperimentsView workspace={workspace} />
            </Match>
            <Match when={page() === "comparison"}>
              <ComparisonView workspace={workspace} />
            </Match>
            <Match when={page() === "settings"}>
              <SettingsView />
            </Match>
          </Switch>
        </main>
      </div>
    </div>
  );
}

function PageHeader(props: {
  eyebrow?: string;
  title: string;
  description: string;
}) {
  return (
    <div class="page-header">
      <div>
        <span class="eyebrow">{props.eyebrow ?? "PROJECT WORKBENCH"}</span>
        <h1>{props.title}</h1>
        <p>{props.description}</p>
      </div>
    </div>
  );
}

function Metric(props: { label: string; value: string; note: string }) {
  return (
    <div class="metric">
      <span>{props.label}</span>
      <strong class="mono">{props.value}</strong>
      <small>{props.note}</small>
    </div>
  );
}

function StatusPill(props: {
  children: string;
  tone?: "ready" | "muted" | "amber";
}) {
  return (
    <span class={`status-pill status-pill--${props.tone ?? "ready"}`}>
      <span class="pill-mark" aria-hidden="true" />
      {props.children}
    </span>
  );
}

function InlineError(props: { message: string }) {
  return (
    <Show when={props.message}>
      <p class="inline-error" role="alert">
        {props.message}
      </p>
    </Show>
  );
}

function RequireProject(props: {
  workspace: Workspace;
  children: JSX.Element;
}) {
  return (
    <Show
      when={props.workspace.projectId()}
      fallback={
        <section class="panel empty-state">
          <h2>Choose a local project first</h2>
          <p>
            Create a project or load the deterministic demo. No account is
            needed.
          </p>
          <A href="/projects" class="button button--primary">
            Open projects
          </A>
        </section>
      }
    >
      {props.children}
    </Show>
  );
}

function ProjectsView(props: { workspace: Workspace }) {
  const navigate = useNavigate();
  const [name, setName] = createSignal("");
  const [description, setDescription] = createSignal("");
  const [error, setError] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [projects, { refetch }] = createResource(
    () => props.workspace.revision(),
    () => api.listProjects(),
  );
  const choose = (id: string) => {
    props.workspace.setProjectId(id);
    props.workspace.setIndexId("");
    navigate(`/projects/${id}`);
  };
  const createProject = async (event: SubmitEvent) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const project = await api.createProject(
        name().trim(),
        description().trim(),
      );
      props.workspace.refresh();
      await refetch();
      choose(project.id);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not create the project.",
      );
    } finally {
      setBusy(false);
    }
  };
  const loadDemo = async () => {
    setBusy(true);
    setError("");
    try {
      const demo = await api.createDemo();
      props.workspace.setProjectId(demo.projectId);
      props.workspace.setIndexId(demo.indexConfigurationId);
      props.workspace.refresh();
      navigate(`/projects/${demo.projectId}/query`);
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Could not create the demo.",
      );
    } finally {
      setBusy(false);
    }
  };
  return (
    <div class="page">
      <PageHeader
        eyebrow="LOCAL WORKSPACE"
        title="Projects"
        description="Create a private local corpus or load a deterministic zero-download demo."
      />
      <section class="panel form-panel compact-form">
        <form onSubmit={createProject}>
          <label for="project-name">Project name</label>
          <input
            id="project-name"
            required
            maxlength="200"
            value={name()}
            onInput={(event) => setName(event.currentTarget.value)}
            placeholder="Distributed Systems Notes"
          />
          <label for="project-description">Description</label>
          <input
            id="project-description"
            value={description()}
            onInput={(event) => setDescription(event.currentTarget.value)}
            placeholder="What this corpus measures"
          />
          <div class="form-actions">
            <button
              class="button button--primary"
              type="submit"
              disabled={busy()}
            >
              <Plus size={15} />
              Create project
            </button>
            <button
              class="button button--secondary"
              type="button"
              onClick={loadDemo}
              disabled={busy()}
            >
              <Spark size={15} />
              Load real demo
            </button>
          </div>
          <InlineError message={error()} />
        </form>
      </section>
      <section class="project-grid" aria-label="Local projects">
        <For
          each={projects()?.data ?? []}
          fallback={<p class="helper-text">No projects yet.</p>}
        >
          {(project) => (
            <button
              class="project-card"
              type="button"
              onClick={() => choose(project.id)}
            >
              <div class="project-card-top">
                <span class="project-icon">
                  <Database size={20} />
                </span>
                <StatusPill>{project.status}</StatusPill>
              </div>
              <h2>{project.name}</h2>
              <p>{project.description || "Local retrieval project"}</p>
              <div class="project-card-meta">
                <span>
                  <strong>{project.documentCount}</strong> documents
                </span>
                <span>
                  <strong>{project.indexCount}</strong> indexes
                </span>
              </div>
              <div class="project-card-footer">
                <span class="mono">{project.id.slice(0, 8)}</span>
                <ArrowUpRight size={15} />
              </div>
            </button>
          )}
        </For>
      </section>
    </div>
  );
}

function OverviewView(props: { workspace: Workspace }) {
  const [project] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.getProject(id) : undefined),
  );
  const [documents] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.listDocuments(id) : undefined),
  );
  const [indexes] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.listIndexes(id) : undefined),
  );
  const totalChunks = () =>
    (documents()?.data ?? []).reduce(
      (sum, document) => sum + document.chunkCount,
      0,
    );
  const latest = () => indexes()?.data.at(-1);
  return (
    <div class="page">
      <PageHeader
        eyebrow="PROJECT OVERVIEW"
        title={project()?.name ?? "Local project"}
        description={
          project()?.description ?? "Live facts from the local SQLite database."
        }
      />
      <RequireProject workspace={props.workspace}>
        <section class="metrics-grid">
          <Metric
            label="Documents"
            value={String(project()?.documentCount ?? 0)}
            note="Source files stored locally"
          />
          <Metric
            label="Index configurations"
            value={String(project()?.indexCount ?? 0)}
            note="Frozen, independent configurations"
          />
          <Metric
            label="Stored chunks"
            value={String(totalChunks())}
            note="Across all index configurations"
          />
          <Metric
            label="Indexed tokens"
            value={String(project()?.totalIndexedTokens ?? 0)}
            note="Whitespace token estimate"
          />
          <Metric
            label="Latest vector count"
            value={String(latest()?.vectorCount ?? 0)}
            note={latest()?.embedding.model ?? "Build an index"}
          />
          <Metric
            label="Evaluation queries"
            value={String(project()?.evaluationQueryCount ?? 0)}
            note={
              project()?.latestExperiment
                ? `Latest: ${project()?.latestExperiment?.name}`
                : "Create relevance judgments"
            }
          />
        </section>
        <Show when={project()?.latestExperiment}>
          {(experiment) => (
            <section class="panel">
              <div class="section-heading">
                <h2>{experiment().name}</h2>
                <span>{experiment().method} · latest experiment</span>
              </div>
              <div class="metric-strip">
                <For each={Object.entries(experiment().metrics)}>
                  {([metric, value]) => (
                    <span>
                      <small>{metric}</small>
                      <strong class="mono">{value.toFixed(4)}</strong>
                    </span>
                  )}
                </For>
              </div>
            </section>
          )}
        </Show>
        <section class="panel panel--accent">
          <h2>Retrieval first</h2>
          <p class="panel-copy">
            Use the debugger to compare native vector scores, BM25 scores,
            deterministic RRF ranks, and optional local reranking.
          </p>
          <A
            href={`/projects/${props.workspace.projectId()}/query`}
            class="button button--primary"
          >
            Open query debugger <ArrowUpRight size={15} />
          </A>
        </section>
      </RequireProject>
    </div>
  );
}

function DocumentsView(props: { workspace: Workspace }) {
  const [fileInput, setFileInput] = createSignal<HTMLInputElement>();
  const [tags, setTags] = createSignal("");
  const [error, setError] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [documents, { refetch }] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.listDocuments(id) : undefined),
  );
  const upload = async (event: SubmitEvent) => {
    event.preventDefault();
    const file = fileInput()?.files?.[0];
    if (!file) {
      setError("Choose a TXT, Markdown, PDF, or source file.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api.uploadDocument(props.workspace.projectId(), file, tags());
      const input = fileInput();
      if (input) input.value = "";
      setTags("");
      props.workspace.refresh();
      await refetch();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div class="page">
      <PageHeader
        eyebrow="UNTRUSTED LOCAL CORPUS"
        title="Documents"
        description="Files are parsed as data. Ingested code is never executed."
      />
      <RequireProject workspace={props.workspace}>
        <section class="panel form-panel compact-form">
          <form onSubmit={upload}>
            <label for="document-file">Document</label>
            <input
              ref={setFileInput}
              id="document-file"
              type="file"
              required
              accept=".txt,.md,.markdown,.pdf,.py,.ts,.js,.go,.rs,.json,.yaml,.yml"
            />
            <label for="document-tags">
              Tags{" "}
              <span class="field-note">
                Comma-separated; used by retrieval filters.
              </span>
            </label>
            <input
              id="document-tags"
              value={tags()}
              onInput={(event) => setTags(event.currentTarget.value)}
              placeholder="consensus, notes"
            />
            <div class="form-actions">
              <button
                class="button button--primary"
                type="submit"
                disabled={busy()}
              >
                <Plus size={15} />
                Add document
              </button>
            </div>
            <InlineError message={error()} />
          </form>
        </section>
        <section class="panel table-panel">
          <table>
            <thead>
              <tr>
                <th>Document</th>
                <th>Type</th>
                <th>Chunks</th>
                <th>Tags</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <For
                each={documents()?.data ?? []}
                fallback={
                  <tr>
                    <td colspan="5">No documents yet.</td>
                  </tr>
                }
              >
                {(document) => (
                  <tr>
                    <td>
                      <div class="table-primary">
                        <FileText size={16} />
                        <strong>{document.filename}</strong>
                      </div>
                      <span class="table-sub mono">
                        {document.sha256.slice(0, 12)}
                      </span>
                    </td>
                    <td>
                      <span class="type-label">{document.sourceType}</span>
                    </td>
                    <td class="mono">{document.chunkCount}</td>
                    <td>{document.tags.join(", ") || "—"}</td>
                    <td>
                      <StatusPill>{document.status}</StatusPill>
                    </td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </section>
      </RequireProject>
    </div>
  );
}

function IndexesView(props: { workspace: Workspace }) {
  const [name, setName] = createSignal("hash-256-overlap32");
  const [strategy, setStrategy] = createSignal("FIXED_TOKEN");
  const [size, setSize] = createSignal(256);
  const [overlap, setOverlap] = createSignal(32);
  const [provider, setProvider] = createSignal("hash");
  const [allowDownload, setAllowDownload] = createSignal(false);
  const [error, setError] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [indexes, { refetch }] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.listIndexes(id) : undefined),
  );
  const build = async (event: SubmitEvent) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const sentence = provider() === "sentence-transformers";
      const index = await api.createIndex(props.workspace.projectId(), {
        name: name(),
        chunking: {
          strategy: strategy(),
          chunkSize: size(),
          overlap: strategy() === "FIXED_TOKEN" ? overlap() : 0,
        },
        embedding: {
          provider: provider(),
          model: sentence ? "BAAI/bge-small-en-v1.5" : "contextbench-hash-v1",
          dimension: sentence ? 384 : 64,
          normalize: true,
          allowModelDownload: allowDownload(),
        },
      });
      props.workspace.setIndexId(index.id);
      props.workspace.refresh();
      await refetch();
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Index build failed.",
      );
    } finally {
      setBusy(false);
    }
  };
  return (
    <div class="page">
      <PageHeader
        eyebrow="FROZEN RETRIEVAL CONFIGURATION"
        title="Indexes"
        description="Each index keeps its own chunks and vectors, so configurations remain comparable."
      />
      <RequireProject workspace={props.workspace}>
        <section class="panel form-panel compact-form">
          <form onSubmit={build}>
            <label for="index-name">Index name</label>
            <input
              id="index-name"
              required
              value={name()}
              onInput={(event) => setName(event.currentTarget.value)}
            />
            <div class="config-grid">
              <label for="chunk-strategy">
                Chunking
                <select
                  id="chunk-strategy"
                  value={strategy()}
                  onChange={(event) => setStrategy(event.currentTarget.value)}
                >
                  <option value="FIXED_TOKEN">Fixed token</option>
                  <option value="PARAGRAPH">Paragraph-aware</option>
                  <option value="HEADING">Heading-aware</option>
                </select>
              </label>
              <label for="chunk-size">
                Chunk size
                <input
                  id="chunk-size"
                  type="number"
                  min="1"
                  max="8192"
                  value={size()}
                  onInput={(event) =>
                    setSize(Number(event.currentTarget.value))
                  }
                />
              </label>
              <label for="chunk-overlap">
                Overlap
                <input
                  id="chunk-overlap"
                  type="number"
                  min="0"
                  max="4096"
                  disabled={strategy() !== "FIXED_TOKEN"}
                  value={overlap()}
                  onInput={(event) =>
                    setOverlap(Number(event.currentTarget.value))
                  }
                />
              </label>
              <label for="embedding-provider">
                Embedding
                <select
                  id="embedding-provider"
                  value={provider()}
                  onChange={(event) => setProvider(event.currentTarget.value)}
                >
                  <option value="hash">Deterministic hash · no download</option>
                  <option value="sentence-transformers">
                    BGE small · local model
                  </option>
                </select>
              </label>
            </div>
            <Show when={provider() === "sentence-transformers"}>
              <div class="download-notice">
                <strong>Model download notice</strong>
                <p>
                  BAAI/bge-small-en-v1.5 is about 130 MB and uses 384
                  dimensions. It is stored in the Hugging Face cache, normally{" "}
                  <span class="mono">~/.cache/huggingface</span>.
                </p>
                <p class="helper-text">
                  Install local ML support first with uv sync --extra ml.
                </p>
                <label class="checkbox-row">
                  <input
                    type="checkbox"
                    checked={allowDownload()}
                    onChange={(event) =>
                      setAllowDownload(event.currentTarget.checked)
                    }
                  />
                  <span>
                    <strong>Allow this model download</strong>
                    <small>
                      Leave clear to use only an existing cached copy.
                    </small>
                  </span>
                </label>
              </div>
            </Show>
            <div class="form-actions">
              <button
                class="button button--primary"
                type="submit"
                disabled={busy()}
              >
                <Play size={15} />
                Build index
              </button>
            </div>
            <InlineError message={error()} />
          </form>
        </section>
        <section class="index-grid">
          <For
            each={indexes()?.data ?? []}
            fallback={<p class="helper-text">No indexes yet.</p>}
          >
            {(index) => (
              <button
                type="button"
                class="index-card"
                onClick={() => props.workspace.setIndexId(index.id)}
              >
                <div class="index-card-head">
                  <div>
                    <span class="eyebrow">CONFIGURATION</span>
                    <h2>{index.name}</h2>
                  </div>
                  <StatusPill
                    tone={
                      props.workspace.indexId() === index.id ? "amber" : "ready"
                    }
                  >
                    {props.workspace.indexId() === index.id
                      ? "ACTIVE"
                      : index.status}
                  </StatusPill>
                </div>
                <div class="index-specs">
                  <div>
                    <span>Chunking</span>
                    <strong class="mono">
                      {index.chunking.strategy} · {index.chunking.chunkSize} /{" "}
                      {index.chunking.overlap}
                    </strong>
                  </div>
                  <div>
                    <span>Vectors</span>
                    <strong class="mono">
                      {index.vectorCount} × {index.vectorDimension}
                    </strong>
                  </div>
                  <div>
                    <span>Embedding</span>
                    <strong>{index.embedding.model}</strong>
                  </div>
                </div>
                <div class="index-card-foot">
                  <span class="mono">
                    {index.indexingMs.toFixed(1)} ms indexing
                  </span>
                  <span>
                    Use index <ArrowUpRight size={13} />
                  </span>
                </div>
              </button>
            )}
          </For>
        </section>
      </RequireProject>
    </div>
  );
}

function QueryDebugger(props: { workspace: Workspace }) {
  const [query, setQuery] = createSignal(
    "What happens when a Raft follower falls behind?",
  );
  const [topK, setTopK] = createSignal(5);
  const [candidateK, setCandidateK] = createSignal(20);
  const [contextLimit, setContextLimit] = createSignal(1200);
  const [documentId, setDocumentId] = createSignal("");
  const [sourceType, setSourceType] = createSignal("");
  const [tag, setTag] = createSignal("");
  const [rerank, setRerank] = createSignal(false);
  const [allowRerankerDownload, setAllowRerankerDownload] = createSignal(false);
  const [result, setResult] = createSignal<RetrievalResponse>();
  const [selectedChunks, setSelectedChunks] = createSignal<string[]>([]);
  const [isRunning, setIsRunning] = createSignal(false);
  const [error, setError] = createSignal("");
  const [notice, setNotice] = createSignal("");
  const [indexes] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.listIndexes(id) : undefined),
  );
  const [documents] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.listDocuments(id) : undefined),
  );
  createEffect(() => {
    const items = indexes()?.data ?? [];
    if (
      items.length &&
      !items.some((item) => item.id === props.workspace.indexId())
    ) {
      props.workspace.setIndexId(items[0]?.id ?? "");
    }
  });
  const run = async () => {
    if (!props.workspace.indexId()) {
      setError("Build or select an index first.");
      return;
    }
    setIsRunning(true);
    setError("");
    setNotice("");
    setSelectedChunks([]);
    try {
      const methods: RetrievalMethod[] = ["VECTOR", "BM25", "HYBRID"];
      if (rerank()) methods.push("RERANKED");
      setResult(
        await api.retrieve(props.workspace.projectId(), {
          query: query().trim(),
          indexConfigurationId: props.workspace.indexId(),
          topK: topK(),
          candidateK: candidateK(),
          methods,
          filters: {
            documentIds: documentId() ? [documentId()] : [],
            sourceTypes: sourceType() ? [sourceType()] : [],
            tags: tag() ? [tag()] : [],
          },
          maxContextTokens: contextLimit(),
          ...(rerank()
            ? {
                reranker: {
                  model: "cross-encoder/ms-marco-MiniLM-L6-v2",
                  allowModelDownload: allowRerankerDownload(),
                },
              }
            : {}),
        }),
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Retrieval failed.");
    } finally {
      setIsRunning(false);
    }
  };
  const toggleChunk = (id: string) =>
    setSelectedChunks((items) =>
      items.includes(id) ? items.filter((item) => item !== id) : [...items, id],
    );
  const saveEvaluation = async () => {
    if (!selectedChunks().length) {
      setError("Mark at least one retrieved chunk as relevant.");
      return;
    }
    try {
      await api.createEvaluationQuery(props.workspace.projectId(), {
        query: query(),
        relevantChunkIds: selectedChunks(),
        datasetVersion: 1,
        notes: "Created in Query Debugger",
      });
      props.workspace.refresh();
      setNotice(
        `Saved evaluation query with ${selectedChunks().length} relevant chunk(s).`,
      );
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not save the evaluation query.",
      );
    }
  };
  return (
    <div class="page page--debugger">
      <PageHeader
        eyebrow="RETRIEVAL OBSERVABILITY"
        title="Query debugger"
        description="Run the real local pipeline. Native score scales remain separate."
      />
      <RequireProject workspace={props.workspace}>
        <section class="query-console panel">
          <div class="query-console-main">
            <label for="query-input">Query</label>
            <textarea
              id="query-input"
              aria-label="Query"
              value={query()}
              onInput={(event) => setQuery(event.currentTarget.value)}
              rows={2}
            />
            <div class="query-console-foot">
              <span class="helper-text">
                Mark result chunks to build a versioned evaluation set.
              </span>
              <button
                class="button button--primary"
                type="button"
                onClick={run}
                disabled={isRunning() || !query().trim()}
              >
                <Play size={15} />
                {isRunning() ? "Running…" : "Run retrieval"}
              </button>
            </div>
          </div>
          <div class="query-config">
            <div class="config-row">
              <label for="index-select">Index</label>
              <select
                id="index-select"
                value={props.workspace.indexId()}
                onChange={(event) =>
                  props.workspace.setIndexId(event.currentTarget.value)
                }
              >
                <option value="" selected={!props.workspace.indexId()}>
                  Select an index
                </option>
                <For each={indexes()?.data ?? []}>
                  {(index) => (
                    <option
                      value={index.id}
                      selected={index.id === props.workspace.indexId()}
                    >
                      {index.name}
                    </option>
                  )}
                </For>
              </select>
            </div>
            <div class="config-grid">
              <label for="top-k">
                Top K
                <input
                  id="top-k"
                  type="number"
                  min="1"
                  max="100"
                  value={topK()}
                  onInput={(event) =>
                    setTopK(Number(event.currentTarget.value))
                  }
                />
              </label>
              <label for="candidate-k">
                Candidate K
                <input
                  id="candidate-k"
                  type="number"
                  min="1"
                  max="1000"
                  value={candidateK()}
                  onInput={(event) =>
                    setCandidateK(Number(event.currentTarget.value))
                  }
                />
              </label>
              <label for="context-limit">
                Estimated context tokens
                <input
                  id="context-limit"
                  type="number"
                  min="1"
                  max="100000"
                  value={contextLimit()}
                  onInput={(event) =>
                    setContextLimit(Number(event.currentTarget.value))
                  }
                />
              </label>
            </div>
            <div class="config-grid">
              <label for="document-filter">
                Document filter
                <select
                  id="document-filter"
                  value={documentId()}
                  onChange={(event) => setDocumentId(event.currentTarget.value)}
                >
                  <option value="">All documents</option>
                  <For each={documents()?.data ?? []}>
                    {(document) => (
                      <option value={document.id}>{document.filename}</option>
                    )}
                  </For>
                </select>
              </label>
              <label for="source-filter">
                Source filter
                <select
                  id="source-filter"
                  value={sourceType()}
                  onChange={(event) => setSourceType(event.currentTarget.value)}
                >
                  <option value="">All types</option>
                  <option value="MD">Markdown</option>
                  <option value="TXT">Text</option>
                  <option value="PDF">PDF</option>
                  <option value="SOURCE">Source</option>
                </select>
              </label>
              <label for="tag-filter">
                Required tag
                <input
                  id="tag-filter"
                  value={tag()}
                  onInput={(event) => setTag(event.currentTarget.value)}
                  placeholder="optional"
                />
              </label>
            </div>
            <label class="checkbox-row">
              <input
                type="checkbox"
                checked={rerank()}
                onChange={(event) => setRerank(event.currentTarget.checked)}
              />
              <span>
                <strong>Use local cross-encoder</strong>
                <small>Optional; disabled by default.</small>
              </span>
            </label>
            <Show when={rerank()}>
              <div class="download-notice">
                <strong>Model download notice</strong>
                <p>
                  cross-encoder/ms-marco-MiniLM-L6-v2 is about 90 MB and is
                  stored in the Hugging Face cache.
                </p>
                <p class="helper-text">
                  Install local ML support first with uv sync --extra ml.
                </p>
                <label class="checkbox-row">
                  <input
                    type="checkbox"
                    checked={allowRerankerDownload()}
                    onChange={(event) =>
                      setAllowRerankerDownload(event.currentTarget.checked)
                    }
                  />
                  <span>
                    <strong>Allow this reranker download</strong>
                    <small>Leave clear to require a cached copy.</small>
                  </span>
                </label>
              </div>
            </Show>
          </div>
        </section>
        <InlineError message={error()} />
        <Show when={notice()}>
          <p class="success-notice" role="status">
            {notice()}
          </p>
        </Show>
        <Show
          when={result()}
          fallback={
            <section class="panel empty-state">
              <h2>No retrieval run yet</h2>
              <p>
                Choose an index and run a query. The results below will come
                from Qdrant local and the in-process BM25 index.
              </p>
            </section>
          }
        >
          {(data) => (
            <>
              <Pipeline result={data()} />
              <section class="debug-summary">
                <Metric
                  label="Total latency"
                  value={`${data().stageLatency.totalMs.toFixed(2)} ms`}
                  note="Measured end to end"
                />
                <Metric
                  label="Final context"
                  value={`${data().contextTokens} estimated tokens`}
                  note="Whitespace estimate; deterministic trim"
                />
                <Metric
                  label="Source diversity"
                  value={`${data().sourceDiversity} docs`}
                  note="Unique sources in context"
                />
                <div class="method-filter">
                  <span class="eyebrow">RELEVANCE</span>
                  <button
                    class="button button--secondary"
                    type="button"
                    onClick={saveEvaluation}
                    disabled={!selectedChunks().length}
                  >
                    Save {selectedChunks().length} marked chunk(s)
                  </button>
                </div>
              </section>
              <section class="lanes">
                <For each={data().lanes}>
                  {(lane) => (
                    <RetrievalLaneCard
                      lane={lane}
                      query={query()}
                      selected={selectedChunks()}
                      onToggle={toggleChunk}
                    />
                  )}
                </For>
              </section>
              <section class="panel context-panel">
                <div class="context-toggle">
                  <span>
                    <span class="eyebrow">FINAL CONTEXT</span>
                    <strong>Exact context sent downstream</strong>
                  </span>
                </div>
                <div class="context-meta">
                  <span class="mono">
                    {data().contextTokens} / {contextLimit()} estimated tokens
                  </span>
                  <span>{data().sourceDiversity} unique sources</span>
                  <span>{data().contextMethod} ranking</span>
                  <span>
                    Deterministic whole-chunk order with boundary trim
                  </span>
                </div>
                <pre>{data().finalContext}</pre>
              </section>
            </>
          )}
        </Show>
      </RequireProject>
    </div>
  );
}

function Pipeline(props: { result: RetrievalResponse }) {
  const stages = () => [
    { label: "Vector", value: props.result.stageLatency.vectorMs },
    { label: "BM25", value: props.result.stageLatency.bm25Ms },
    { label: "Fuse", value: props.result.stageLatency.fusionMs },
    { label: "Rerank", value: props.result.stageLatency.rerankMs },
    { label: "Assemble", value: props.result.stageLatency.assembleMs },
  ];
  return (
    <section class="pipeline panel">
      <div class="section-heading">
        <h2>Request pipeline</h2>
        <span>Measured stages</span>
      </div>
      <div class="pipeline-track">
        <For each={stages()}>
          {(stage, index) => (
            <>
              <div class="pipeline-stage">
                <span class="pipeline-index">0{index() + 1}</span>
                <strong>{stage.label}</strong>
                <span class="mono">{stage.value.toFixed(2)} ms</span>
              </div>
              <Show when={index() < stages().length - 1}>
                <span class="pipeline-arrow" aria-hidden="true">
                  →
                </span>
              </Show>
            </>
          )}
        </For>
      </div>
    </section>
  );
}

function RetrievalLaneCard(props: {
  lane: RetrievalLane;
  query: string;
  selected: string[];
  onToggle: (id: string) => void;
}) {
  const [opened, setOpened] = createSignal(true);
  return (
    <article class={`lane-card lane-card--${methodTone[props.lane.method]}`}>
      <div class="lane-head">
        <div class="lane-title">
          <span class="lane-code">{props.lane.method}</span>
          <div>
            <h2>{props.lane.label}</h2>
            <p>
              {props.lane.hits.length} inspectable chunks ·{" "}
              {props.lane.latencyMs.toFixed(2)} ms
            </p>
          </div>
        </div>
        <div class="lane-metrics">
          <button
            class="icon-button"
            type="button"
            aria-label={`Toggle ${props.lane.label} chunks`}
            aria-expanded={opened()}
            onClick={() => setOpened((value) => !value)}
          >
            <ChevronDown size={16} />
          </button>
        </div>
      </div>
      <Show when={opened()}>
        <div class="hit-list">
          <For each={props.lane.hits}>
            {(hit) => (
              <HitRow
                hit={hit}
                method={props.lane.method}
                query={props.query}
                selected={props.selected.includes(hit.chunk.id)}
                onToggle={() => props.onToggle(hit.chunk.id)}
              />
            )}
          </For>
        </div>
      </Show>
    </article>
  );
}

function HighlightedText(props: { text: string; query: string }) {
  const parts = createMemo(() => {
    const terms = props.query
      .toLowerCase()
      .split(/\s+/)
      .filter((term) => term.length > 3)
      .slice(0, 5);
    const expression = terms
      .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|");
    if (!expression) return [{ text: props.text, matched: false }];
    return props.text
      .split(new RegExp(`(${expression})`, "ig"))
      .map((text) => ({ text, matched: terms.includes(text.toLowerCase()) }));
  });
  return (
    <span>
      <For each={parts()}>
        {(part) => (
          <Show when={part.matched} fallback={part.text}>
            <mark>{part.text}</mark>
          </Show>
        )}
      </For>
    </span>
  );
}

function HitRow(props: {
  hit: RetrievalHit;
  method: RetrievalMethod;
  query: string;
  selected: boolean;
  onToggle: () => void;
}) {
  const [details, setDetails] = createSignal(false);
  return (
    <div class="hit-row">
      <div class="rank-cell">
        <span class="rank-number">
          {String(props.hit.rank).padStart(2, "0")}
        </span>
        <span class="rank-label">
          {props.method === "RERANKED" ? "rerank" : "rank"}
        </span>
        <Show when={props.hit.candidateRank != null}>
          <span class="rank-label">
            from {String(props.hit.candidateRank).padStart(2, "0")}
          </span>
        </Show>
      </div>
      <div class="hit-content">
        <div class="hit-meta">
          <strong>{props.hit.chunk.documentName}</strong>
          <span class="mono">
            chunk {String(props.hit.chunk.ordinal).padStart(2, "0")}
          </span>
          <Show when={props.hit.chunk.page}>
            <span>page {props.hit.chunk.page}</span>
          </Show>
        </div>
        <p>
          <HighlightedText text={props.hit.chunk.text} query={props.query} />
        </p>
        <div class="hit-details">
          <button
            class="text-button"
            type="button"
            onClick={() => setDetails((value) => !value)}
          >
            {details() ? "Hide details" : "Inspect chunk"}
          </button>
          <label class="checkbox-row relevance-check">
            <input
              type="checkbox"
              aria-label={`Mark ${props.hit.chunk.documentName} chunk ${props.hit.chunk.ordinal} relevant`}
              checked={props.selected}
              onChange={() => props.onToggle()}
            />
            <span>
              <strong>Relevant</strong>
            </span>
          </label>
        </div>
        <Show when={details()}>
          <div class="hit-details">
            <span>
              Tokens <strong class="mono">{props.hit.chunk.tokenCount}</strong>
            </span>
            <span>
              Characters{" "}
              <strong class="mono">
                {props.hit.chunk.startChar}–{props.hit.chunk.endChar}
              </strong>
            </span>
            <Show when={props.hit.rrfScore != null}>
              <span>
                RRF{" "}
                <strong class="mono">{props.hit.rrfScore?.toFixed(6)}</strong>
              </span>
            </Show>
            <Show when={props.hit.vectorScore != null}>
              <span>
                Vector{" "}
                <strong class="mono">
                  {props.hit.vectorScore?.toFixed(6)}
                </strong>
              </span>
            </Show>
            <Show when={props.hit.bm25Score != null}>
              <span>
                BM25{" "}
                <strong class="mono">{props.hit.bm25Score?.toFixed(3)}</strong>
              </span>
            </Show>
            <Show when={props.hit.crossEncoderScore != null}>
              <span>
                Cross-encoder{" "}
                <strong class="mono">
                  {props.hit.crossEncoderScore?.toFixed(4)}
                </strong>
              </span>
            </Show>
          </div>
        </Show>
      </div>
      <div class="score-cell">
        <strong class="mono">
          {props.hit.nativeScore.toFixed(props.method === "BM25" ? 3 : 6)}
        </strong>
        <span>
          {
            {
              VECTOR: "cosine score",
              BM25: "BM25 score",
              HYBRID: "RRF score",
              RERANKED: "cross-encoder score",
            }[props.method]
          }
        </span>
      </div>
    </div>
  );
}

function EvaluationView(props: { workspace: Workspace }) {
  const [queries] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.listEvaluationQueries(id) : undefined),
  );
  return (
    <div class="page">
      <PageHeader
        eyebrow="RELEVANCE JUDGMENTS"
        title="Evaluation set"
        description="Versioned, human-marked relevant chunks. Metrics use these judgments only."
      />
      <RequireProject workspace={props.workspace}>
        <div class="dataset-banner">
          <div>
            <span class="eyebrow">ACTIVE DATASET</span>
            <h2>Evaluation queries · v1</h2>
          </div>
          <span class="mono">{queries()?.data.length ?? 0} queries</span>
          <StatusPill tone="amber">LOCAL</StatusPill>
        </div>
        <section class="panel table-panel">
          <table>
            <thead>
              <tr>
                <th>Query</th>
                <th>Relevant chunks</th>
                <th>Version</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              <For
                each={queries()?.data ?? []}
                fallback={
                  <tr>
                    <td colspan="4">
                      Mark relevant results in the Query Debugger.
                    </td>
                  </tr>
                }
              >
                {(item) => (
                  <tr>
                    <td>
                      <strong>{item.query}</strong>
                      <span class="table-sub mono">{item.id}</span>
                    </td>
                    <td class="mono">{item.relevantChunkIds.length}</td>
                    <td class="mono">v{item.datasetVersion}</td>
                    <td>{item.notes || "—"}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </section>
      </RequireProject>
    </div>
  );
}

function ExperimentsView(props: { workspace: Workspace }) {
  const [name, setName] = createSignal("hybrid-v1");
  const [method, setMethod] = createSignal<RetrievalMethod>("HYBRID");
  const [error, setError] = createSignal("");
  const [latestMetrics, setLatestMetrics] =
    createSignal<Record<string, number>>();
  const [indexes] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.listIndexes(id) : undefined),
  );
  const [experiments, { refetch }] = createResource(
    () => [props.workspace.projectId(), props.workspace.revision()] as const,
    ([id]) => (id ? api.listExperiments(id) : undefined),
  );
  const run = async (event: SubmitEvent) => {
    event.preventDefault();
    setError("");
    const indexId = props.workspace.indexId() || indexes()?.data[0]?.id;
    if (!indexId) {
      setError("Build an index first.");
      return;
    }
    try {
      const experiment = await api.createExperiment(
        props.workspace.projectId(),
        {
          name: name(),
          datasetVersion: 1,
          indexConfigurationId: indexId,
          method: method(),
          kValues: [1, 3, 5, 10],
        },
      );
      const detail = await api.getExperiment(experiment.id);
      setLatestMetrics(detail.metrics);
      props.workspace.refresh();
      await refetch();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Experiment failed.");
    }
  };
  const toggle = (id: string) =>
    props.workspace.setSelectedExperiments((items) =>
      items.includes(id)
        ? items.filter((item) => item !== id)
        : [...items, id].slice(-4),
    );
  return (
    <div class="page">
      <PageHeader
        eyebrow="IMMUTABLE EVALUATION RUNS"
        title="Experiments"
        description="Run standard IR metrics over one dataset version and one frozen index."
      />
      <RequireProject workspace={props.workspace}>
        <section class="panel form-panel compact-form">
          <form onSubmit={run}>
            <label for="experiment-name">Experiment name</label>
            <input
              id="experiment-name"
              required
              value={name()}
              onInput={(event) => setName(event.currentTarget.value)}
            />
            <label for="experiment-method">Method</label>
            <select
              id="experiment-method"
              value={method()}
              onChange={(event) =>
                setMethod(event.currentTarget.value as RetrievalMethod)
              }
            >
              <option value="VECTOR">Vector</option>
              <option value="BM25">BM25</option>
              <option value="HYBRID">Hybrid RRF</option>
            </select>
            <div class="form-actions">
              <button class="button button--primary" type="submit">
                <Play size={15} />
                Run experiment
              </button>
              <A
                href={`/projects/${props.workspace.projectId()}/comparison`}
                class="button button--secondary"
              >
                Compare selected ({props.workspace.selectedExperiments().length}
                )
              </A>
            </div>
            <InlineError message={error()} />
          </form>
        </section>
        <Show when={latestMetrics()}>
          {(metrics) => (
            <section class="metrics-grid">
              <For each={Object.entries(metrics())}>
                {([key, value]) => (
                  <Metric
                    label={key}
                    value={value.toFixed(4)}
                    note="Computed from stored rankings"
                  />
                )}
              </For>
            </section>
          )}
        </Show>
        <section class="experiment-list">
          <For
            each={experiments()?.data ?? []}
            fallback={<p class="helper-text">No completed experiments yet.</p>}
          >
            {(experiment) => (
              <label class="experiment-row">
                <input
                  type="checkbox"
                  aria-label={`Select ${experiment.name}`}
                  checked={props.workspace
                    .selectedExperiments()
                    .includes(experiment.id)}
                  onChange={() => toggle(experiment.id)}
                />
                <div class="experiment-id">
                  <span class="experiment-icon">
                    <Spark size={17} />
                  </span>
                  <div>
                    <strong>{experiment.name}</strong>
                    <span class="table-sub mono">{experiment.id}</span>
                  </div>
                </div>
                <StatusPill>{experiment.status}</StatusPill>
                <ChevronRight size={16} />
              </label>
            )}
          </For>
        </section>
      </RequireProject>
    </div>
  );
}

function ComparisonView(props: { workspace: Workspace }) {
  const [comparison, setComparison] = createSignal<ExperimentComparison>();
  const [error, setError] = createSignal("");
  const compare = async () => {
    if (props.workspace.selectedExperiments().length < 2) {
      setError("Select at least two experiments on the Experiments page.");
      return;
    }
    try {
      setError("");
      setComparison(
        await api.compareExperiments(props.workspace.selectedExperiments()),
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Comparison failed.");
    }
  };
  return (
    <div class="page">
      <PageHeader
        eyebrow="EXPERIMENT ANALYSIS"
        title="Comparison"
        description="Compare only stored metrics computed from the same local relevance data."
      />
      <RequireProject workspace={props.workspace}>
        <section class="compare-controls panel">
          <div>
            <span class="eyebrow">SELECTED RUNS</span>
            <h2>{props.workspace.selectedExperiments().length} experiments</h2>
            <p>Selection stays in this browser session.</p>
          </div>
          <button
            class="button button--primary"
            type="button"
            onClick={compare}
          >
            Compare runs <ArrowUpRight size={15} />
          </button>
        </section>
        <InlineError message={error()} />
        <Show when={comparison()}>
          {(data) => (
            <section class="panel compare-table">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <For each={data().experiments}>
                      {(experiment) => <th>{experiment.name}</th>}
                    </For>
                  </tr>
                </thead>
                <tbody>
                  <For each={data().metricKeys}>
                    {(metric) => (
                      <tr>
                        <td>
                          <strong>{metric}</strong>
                        </td>
                        <For each={data().experiments}>
                          {(experiment) => (
                            <td class="mono">
                              {experiment.metrics[metric]?.toFixed(4) ?? "—"}
                            </td>
                          )}
                        </For>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </section>
          )}
        </Show>
      </RequireProject>
    </div>
  );
}

function SettingsView() {
  return (
    <div class="page">
      <PageHeader
        eyebrow="LOCAL-FIRST CONFIGURATION"
        title="Settings"
        description="ContextBench binds to loopback and sends no telemetry."
      />
      <section class="settings-grid">
        <section class="panel form-panel">
          <h2>API connection</h2>
          <label for="api-url">Base URL</label>
          <input id="api-url" value="http://127.0.0.1:8000" readOnly />
          <p class="field-note">
            The Vite development server proxies API calls to this loopback
            address.
          </p>
        </section>
        <section class="panel form-panel">
          <h2>Privacy boundary</h2>
          <p>
            Documents, chunks, vectors, queries, and experiment results remain
            in local SQLite and Qdrant files. There is no sign-in, analytics, or
            cloud upload.
          </p>
          <p>
            Optional Ollama requests are limited to{" "}
            <span class="mono">127.0.0.1</span> or{" "}
            <span class="mono">localhost</span>.
          </p>
        </section>
      </section>
    </div>
  );
}
