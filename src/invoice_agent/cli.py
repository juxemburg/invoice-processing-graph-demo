"""Typer CLI entrypoint for the invoice agent."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from pydantic_graph import GraphRunResult

from invoice_agent.graph import invoice_graph
from invoice_agent.models.schemas import FinalReport
from invoice_agent.nodes.pipeline import GraphState, ProcessNextInvoice
from invoice_agent.settings import get_settings
from invoice_agent.tracing import capture_root_context, init_tracing

app = typer.Typer(help="Invoice expense report agent.")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}


@app.command()
def process(
    input_folder: Path = typer.Option(
        ..., "--input", "-i", help="Folder containing invoice images"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Write JSON to file (default: stdout)"
    ),
) -> None:
    """Process invoice images and produce a structured expense report."""
    # Validate settings early (fail fast if ANTHROPIC_API_KEY is missing)
    settings = get_settings()

    # Initialize tracing BEFORE any agent is created
    init_tracing()

    if not input_folder.is_dir():
        typer.echo(f"Error: '{input_folder}' is not a directory.", err=True)
        raise typer.Exit(1)

    image_paths = sorted(
        p for p in input_folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        typer.echo(
            f"Error: No supported images found in '{input_folder}'.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Processing {len(image_paths)} invoice(s)...")

    async def _run_graph(
        paths: list[Path], graph_state: GraphState
    ) -> GraphRunResult[GraphState, FinalReport]:
        """Run the graph, optionally inside a Langfuse trace context."""
        if settings.langfuse_enabled:
            from langfuse import get_client

            langfuse = get_client()
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            with langfuse.start_as_current_observation(
                as_type="span",
                name="invoice-agent-run",
                input={
                    "invoice_count": len(paths),
                    "files": [p.name for p in paths],
                },
                metadata={
                    "session_id": f"run_{run_id}",
                    "user_id": "invoice-agent-cli",
                    "tags": ["invoice-processing", "v0.1.0"],
                    "release": "0.1.0",
                },
            ) as root:
                capture_root_context()
                res = await invoice_graph.run(ProcessNextInvoice(), state=graph_state)
                report = res.output
                root.update(
                    output={
                        "total_spend": str(report.total_spend),
                        "spend_by_category": report.spend_by_category,
                        "invoice_count": len(report.invoices),
                        "issues_count": len(report.issues_and_assumptions),
                    },
                )
            langfuse.flush()
            return res
        return await invoice_graph.run(ProcessNextInvoice(), state=graph_state)

    state = GraphState(image_paths=image_paths)
    result = asyncio.run(_run_graph(image_paths, state))

    json_output = json.dumps(dataclasses.asdict(result.output), indent=2)

    if output:
        output.write_text(json_output)
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(json_output)
