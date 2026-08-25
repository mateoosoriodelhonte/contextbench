# Local development

ContextBench needs Python 3.12, Node 20 or newer, `uv`, and npm. It does not need Docker.

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
uv run contextbench demo --embedding-provider deterministic-hash
```

To build a semantic index for the first time, pass the explicit model-download flag shown by the
CLI. ContextBench never pulls an Ollama model.

