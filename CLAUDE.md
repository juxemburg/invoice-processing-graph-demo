# Project: Invoice processing agent

## Project Overview

This is a Pydantic AI Graph-based invoice processing agent. It extracts data
from invoice images, normalizes fields, categorizes expenses via LLM, and
produces a structured JSON report.

**Architecture:** Standalone service functions + thin graph node wrappers.
Business logic lives in `src/invoice_agent/services/`. Graph nodes in
`src/invoice_agent/nodes/` are thin orchestration wrappers only.

---

## Package Manager: uv (MANDATORY)

This project uses **uv** exclusively. Do NOT use `pip`, `pip install`,
`python -m pip`, or `poetry`. Every dependency and script interaction must go
through `uv`.

### Common Commands

```bash
# Install/sync all dependencies (including dev group)
uv sync

# Add a production dependency
uv add <package>

# Add a dev dependency
uv add --group dev <package>

# Run the CLI entrypoint
uv run invoice-agent process --input ./invoice_samples/

# Run a one-off Python command
uv run python -c "from invoice_agent.models.schemas import FinalReport; print('OK')"

# Run pytest (all tests)
uv run pytest

# Run specific test tier
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/e2e/ -v

# Run a single test file
uv run pytest tests/unit/test_normalizer.py -v

# Run a single test by name
uv run pytest tests/unit/test_normalizer.py -k "test_eu_format_space_thousands" -v
```

### Dependency Rules

- **Always run `uv sync` after modifying `pyproject.toml`** to ensure the
  lockfile and virtualenv are up to date.
- **Always use `uv add`** to add dependencies — never edit the `[dependencies]`
  list in `pyproject.toml` manually.
- Dev tools (pytest, ruff, pyright, pylint) go in `[dependency-groups] dev`.
- Production dependencies go in `[project] dependencies`.

---

## Code Quality: Linting, Formatting, Type Checking

Run ALL of these before considering any task complete:

```bash
# Format code (auto-fix)
uv run ruff format src/ tests/

# Lint code (auto-fix what's possible)
uv run ruff check src/ tests/ --fix

# Lint code (report only — no fixes)
uv run ruff check src/ tests/

# Type check
uv run pyright src/

# Pylint (stricter analysis)
uv run pylint src/invoice_agent/
```

### Quality Gate Checklist

Before marking any phase or task as done, verify ALL of the following pass
with zero errors:

1. `uv run ruff format src/ tests/` — no reformatting needed
2. `uv run ruff check src/ tests/` — no lint errors
3. `uv run pyright src/` — no type errors
4. `uv run pytest tests/unit/ -v` — all unit tests pass
5. `uv run pytest tests/integration/ -v` — all integration tests pass

If any check fails, fix the issue before proceeding.

---

## Testing Conventions

### Test Runner

Always use `uv run pytest`. Never use bare `pytest` or `python -m pytest`.

### Test Layout

```
tests/
  unit/           # Pure function tests — no API calls, no async graph
  integration/    # Node transition tests — mocked LLM via patch/TestModel
  e2e/            # Full graph run — requires ANTHROPIC_API_KEY
```

### Configuration (already in pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

This means:
- Imports like `from invoice_agent.services.normalizer import parse_amount`
  work in tests without installing the package in editable mode.
- `uv run pytest` automatically discovers tests under `tests/`.

### Async Tests

Add this to `pyproject.toml` if using async tests:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

And ensure `pytest-asyncio` is in the dev dependencies:

```bash
uv add --group dev pytest-asyncio
```

### E2E Tests

E2E tests require a real API key and are skipped automatically when
`ANTHROPIC_API_KEY` is not set. To run them:

```bash
ANTHROPIC_API_KEY=sk-... uv run pytest tests/e2e/ -v
```

Never run E2E tests in a loop or as part of routine verification — they cost
real money and are slow (~4s per invoice).

---

## Self-Verification Workflow

After writing or modifying any code, run this exact sequence:

```bash
# Step 1: Format
uv run ruff format src/ tests/

# Step 2: Lint
uv run ruff check src/ tests/ --fix

# Step 3: Type check
uv run pyright src/

# Step 4: Unit tests
uv run pytest tests/unit/ -v

# Step 5: Integration tests
uv run pytest tests/integration/ -v
```

If Steps 1-5 all pass, the code is ready. If any step fails, fix the issue
and re-run FROM THAT STEP onward (not just the failing step — downstream
checks may also be affected).

---

## Project Structure

```
caseware-take-home-exercise/
├── CLAUDE.md                  ← you are here
├── pyproject.toml
├── uv.lock
├── .env                       ← ANTHROPIC_API_KEY (never commit)
├── .env.example
├── invoice_samples/           ← test invoice images
├── research/                  ← design docs, diagrams
└── src/
    └── invoice_agent/
        ├── __init__.py
        ├── models/
        │   ├── __init__.py
        │   └── schemas.py     ← Pydantic models, dataclasses, type aliases
        ├── services/
        │   ├── __init__.py
        │   ├── extractor.py   ← VLM agent + image helpers
        │   ├── validator.py   ← Schema validation guardrail
        │   ├── normalizer.py  ← Pure functions: parse_amount, parse_date
        │   ├── categorizer.py ← LLM categorization agent
        │   ├── aggregator.py  ← Deterministic Decimal math
        │   └── reporter.py    ← LLM narrative + deterministic numbers
        ├── nodes/
        │   ├── __init__.py
        │   └── pipeline.py    ← GraphState + all graph nodes
        ├── graph.py            ← Graph wiring
        └── cli.py              ← Typer CLI entrypoint
tests/
    unit/
        test_normalizer.py
        test_aggregator.py
        test_validator.py
        test_extractor.py
        test_categorizer.py
    integration/
        test_nodes.py
    e2e/
        test_pipeline.py
```

---

## Code Style Rules

- **Line length:** 88 characters (configured in ruff and pylint)
- **Quotes:** double quotes (configured in ruff format)
- **Imports:** sorted by ruff (isort rules via `select = ["I"]`)
- **Type hints:** required on all function signatures. Use `from __future__
  import annotations` at the top of every module.
- **Money:** always use `decimal.Decimal`, never `float`.
- **Docstrings:** required on all public functions and classes.

---

## Key Architectural Rules

1. **Services have no graph imports.** Files in `services/` must never import
   from `nodes/` or `graph.py`. They are standalone and independently testable.

2. **Nodes have no business logic.** Files in `nodes/` call service functions
   and manage state transitions only. If you're writing an `if` statement that
   isn't about "which node comes next," it belongs in a service.

3. **Data flows forward via node fields.** Nodes pass data to the next node
   through constructor arguments (dataclass fields), NOT through mutable
   shared state on `GraphState`. The only fields on `GraphState` are
   `image_paths`, `current_index`, and `processed`.

4. **No LLM math.** The aggregation step uses `Decimal` arithmetic only. The
   LLM is never asked to compute totals, sums, or any numeric operations.

5. **Structured output enforcement.** LLM calls use pydantic-ai's `result_type`
   parameter to enforce output schemas. The `CategoryResult` model has a
   validator that requires notes when category is "Other".

---

## Environment Variables

| Variable             | Required | Description                    |
|----------------------|----------|--------------------------------|
| `ANTHROPIC_API_KEY`  | Yes      | API key for Claude calls       |

Load via `python-dotenv` from `.env` file at project root. The CLI does this
automatically via `load_dotenv()` in `cli.py`.

---

## Common Pitfalls

- **Don't use `asyncio.run()` inside pytest async tests.** With
  `asyncio_mode = "auto"`, just `await` directly in `async def test_*`
  functions.
- **Don't import from `nodes/` in service files.** This creates circular
  dependencies.
- **Don't use `float` for money.** Always `Decimal`.
- **Don't run E2E tests routinely.** They hit a real API. Use unit and
  integration tests for iterative development.
- **Don't forget `uv sync` after adding dependencies.** The lockfile must
  stay in sync.