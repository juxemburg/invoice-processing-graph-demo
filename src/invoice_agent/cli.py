"""Typer CLI entrypoint for the invoice agent."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import typer

from invoice_agent.graph import invoice_graph
from invoice_agent.nodes.pipeline import GraphState, ProcessNextInvoice
from invoice_agent.settings import get_settings

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
    get_settings()

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

    state = GraphState(image_paths=image_paths)
    result = asyncio.run(invoice_graph.run(ProcessNextInvoice(), state=state))

    json_output = json.dumps(dataclasses.asdict(result.output), indent=2)

    if output:
        output.write_text(json_output)
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(json_output)
