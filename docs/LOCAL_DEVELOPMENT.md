# Local development

ContextBench needs Python 3.12, Node 22 or newer, `uv`, and npm. It does not need Docker.

```bash
git clone https://github.com/mateoosoriodelhonte/contextbench.git
cd contextbench
uv sync --extra dev

cd frontend
npm ci
cd ..
```

Start the API and frontend in separate terminals:

```bash
uv run contextbench serve
```

```bash
cd frontend
npm run dev
```

Both servers bind to `127.0.0.1`. The Vite server proxies `/api` to port 8000. ContextBench stores
local state under `.contextbench/` unless `CONTEXTBENCH_DATA_DIR` names another directory.

Create the deterministic demo without downloading a model:

```bash
uv run contextbench demo
```

To build a semantic index for the first time, choose Sentence Transformers in the web app. The UI
shows the model, size, and cache location before it enables download consent. ContextBench never
pulls an Ollama model.

Run all local gates:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest
uv build
cd frontend
npm run format:check
npm run lint
npm run typecheck
npm test
npm run build
npm run e2e
```
