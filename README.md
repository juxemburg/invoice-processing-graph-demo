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
```

## Running the Scoring Script

Process invoices using the `invoice-agent process` subcommand:

```bash
# Process invoices in the default samples directory
uv run invoice-agent process --input ./invoice_samples/

# Process invoices in a custom directory
uv run invoice-agent process --input /path/to/your/invoices/

# Write output to a specific file
uv run invoice-agent process --input ./invoice_samples/ --output output/report.json
```

### CLI Options

| Option     | Description                                     | Default       |
| ---------- | ----------------------------------------------- | ------------- |
| `--input`  | Path to the directory containing invoice images | Required      |
| `--output` | Path to write the JSON report                   | `report.json` |

The agent will:

1. **Extract** structured data from each invoice image using a vision LLM
2. **Validate** the extracted fields against the expected schema
3. **Normalize** amounts and dates into canonical formats
4. **Categorize** each expense via LLM
5. **Aggregate** totals and generate a narrative JSON report

## Running Tests

```bash
# Unit tests (no API key needed)
uv run pytest tests/unit/ -v

# Integration tests (mocked LLM, no API key needed)
uv run pytest tests/integration/ -v

# E2E tests (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-ant-... uv run pytest tests/e2e/ -v
```
