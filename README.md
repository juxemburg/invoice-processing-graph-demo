# Invoice Processing Agent

A CLI agent that processes invoice images through a 5-step pipeline (extraction, validation, normalization, categorization, aggregation + report) using a pydantic-ai Graph. It extracts data from invoice images via a vision LLM, normalizes fields, categorizes expenses, and produces a structured JSON expense report.

## Quickstart

**Prerequisites:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
# Install dependencies
uv sync

# Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Process invoices
uv run invoice-agent --input ./invoice_samples/

# Write output to a file
uv run invoice-agent --input ./invoice_samples/ --output output/report.json
```

## Running Tests

```bash
# Unit tests (no API key needed)
uv run pytest tests/unit/ -v

# Integration tests (mocked LLM, no API key needed)
uv run pytest tests/integration/ -v

# E2E tests (requires ANTHROPIC_API_KEY)
uv run pytest tests/e2e/ -v
```
