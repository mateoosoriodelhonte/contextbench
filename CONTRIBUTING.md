# Contributing

ContextBench accepts focused changes that preserve local operation and retrieval correctness.
Open an issue before a change that alters an API, database model, metric, chunking rule, retrieval
score, or model default.

## Local checks

Install Python 3.12 and Node 20 or newer. The project uses `uv` and npm lockfiles.

```bash
uv sync --extra dev
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest

cd frontend
npm ci
npm run format:check
npm run lint
npm run typecheck
npm test
npm run build
```

Run Playwright when a user flow changes. Do not add a required cloud service or paid API. Tests
must use synthetic or public fixtures. A test may not download a large model unless it carries the
`real_model` marker and is disabled in the default CI path.

## Pull requests

Keep a pull request tied to one issue or one narrow child issue. State the behavior change, tests,
model download effect, storage migration, and privacy effect. Use `Refs #...` for partial work.
Use a closing keyword only when the pull request meets every acceptance criterion in the issue.

Commit messages use `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, or `chore:` followed by a short
reason for the change.

