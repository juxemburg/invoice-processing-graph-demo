"""Typer CLI entrypoint for the invoice agent."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from pydantic_graph import GraphRunResult

from invoice_agent.graph import invoice_graph
from invoice_agent.models.schemas import FinalReport
from invoice_agent.nodes.pipeline import GraphState, ProcessNextInvoice
from invoice_agent.services.eval_client import (
    fetch_recent_traces,
    fetch_trace_data,
    push_scores,
)
from invoice_agent.services.eval_scoring import (
    aggregate_trace_scores,
    parse_span_outputs,
    score_aggregate_span,
    score_extract_span,
)
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
            from langfuse import get_client, propagate_attributes

            langfuse = get_client()
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            with (
                langfuse.start_as_current_observation(
                    as_type="span",
                    name="invoice-agent-run",
                    input={
                        "invoice_count": len(paths),
                        "files": [p.name for p in paths],
                    },
                ) as root,
                propagate_attributes(
                    session_id=f"run_{run_id}",
                    user_id="invoice-agent-cli",
                    tags=["invoice-processing", "v0.1.0"],
                    version="0.1.0",
                ),
            ):
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


# ── Evaluate command ─────────────────────────────────────────────────────────


def _evaluate_single(trace_id: str) -> dict[str, tuple[float, str]]:
    """Run full evaluation pipeline for a single trace.

    Returns a dict mapping dimension names to (score, detail) tuples.
    """
    data = fetch_trace_data(trace_id)
    parsed = parse_span_outputs(data["observations"])

    # 1. Score each ExtractNode span
    extract_scores: list[float] = []
    for entry in parsed["extract"]:
        score, detail = score_extract_span(entry["output"])
        extract_scores.append(score)
        push_scores(
            trace_id,
            {"extraction_completeness": (score, detail)},
            observation_id=entry["id"],
        )

    # 2. Score AggregateNode span
    agg_score: float | None = None
    if parsed["aggregate"] is not None:
        agg_entry = parsed["aggregate"]
        norm_outputs = [e["output"] for e in parsed["normalize"]]
        cat_outputs = [e["output"] for e in parsed["categorize"]]
        score, detail = score_aggregate_span(
            agg_entry["output"], norm_outputs, cat_outputs
        )
        agg_score = score
        push_scores(
            trace_id,
            {"reconciliation": (score, detail)},
            observation_id=agg_entry["id"],
        )

    # 3. Collect existing judge scores from CategorizeNode spans
    judge_values = [
        s["value"]
        for s in data["scores"]
        if s.get("name") == "categorizer_evaluator_v2"
    ]

    # 4. Aggregate into trace-level scores
    trace_scores = aggregate_trace_scores(extract_scores, agg_score, judge_values)
    push_scores(trace_id, trace_scores)

    return trace_scores


def _print_single_report(trace_id: str, scores: dict[str, tuple[float, str]]) -> None:
    """Print formatted single-run evaluation report."""
    schema_val, schema_detail = scores["schema"]
    recon_val, recon_detail = scores["reconciliation"]
    cat_val, cat_detail = scores["category_quality"]
    composite_val, composite_label = scores["composite"]

    label_map = {
        "PASS": "production ready",
        "WARN": "review recommended",
        "INVESTIGATE": "needs investigation",
        "BLOCK": "not production ready",
    }

    typer.echo(f"\nEvaluation for trace {trace_id}")
    typer.echo("\u2550" * 56)
    typer.echo("")
    typer.echo(f"  Schema / Completeness:    {schema_val:.2f}  ({schema_detail})")
    typer.echo(f"  Reconciliation:           {recon_val:.2f}  ({recon_detail})")
    typer.echo(f"  Category Quality:         {cat_val:.2f}  ({cat_detail})")
    typer.echo("  " + "\u2500" * 54)
    typer.echo(
        f"  Overall:                  {composite_val:.2f}  "
        f"{composite_label} \u2014 {label_map.get(composite_label, '')}"
    )
    typer.echo("")
    typer.echo("  Scores pushed to Langfuse \u2713")


def _print_multi_report(results: list[dict[str, str | float]]) -> None:
    """Print formatted multi-run comparison table."""
    typer.echo(f"\nEvaluation Summary (last {len(results)} runs)")
    typer.echo("\u2550" * 72)
    typer.echo("")
    typer.echo(
        "  Trace ID          Session               Schema  Recon  CatQ   Overall"
    )
    sep = "  " + "\u2500" * 70
    typer.echo(sep)
    for row in results:
        tid = str(row["trace_id"])[:10] + "..."
        sid = str(row.get("session_id", ""))[:20]
        typer.echo(
            f"  {tid:<16}  {sid:<20}  "
            f"{row['schema']:>5.2f}   "
            f"{row['recon']:>4.2f}   "
            f"{row['cat_q']:>4.2f}   "
            f"{row['overall']:>6.2f}"
        )
    typer.echo("")
    typer.echo("  Scores pushed to Langfuse \u2713")


@app.command()
def evaluate(
    trace_id: Optional[str] = typer.Option(
        None, "--trace-id", "-t", help="Evaluate a specific trace"
    ),
    last: Optional[int] = typer.Option(
        None, "--last", "-n", help="Evaluate the N most recent runs"
    ),
) -> None:
    """Evaluate invoice-agent traces and push scores to Langfuse."""
    if trace_id is None and last is None:
        typer.echo(
            "Error: provide --trace-id or --last. "
            "Run 'invoice-agent evaluate --help' for usage.",
            err=True,
        )
        raise typer.Exit(1)

    if trace_id is not None:
        typer.echo(f"Evaluating trace {trace_id}...")
        scores = _evaluate_single(trace_id)
        _print_single_report(trace_id, scores)
    else:
        assert last is not None
        typer.echo(f"Fetching last {last} traces...")
        traces = fetch_recent_traces(last)
        if not traces:
            typer.echo("No traces found.", err=True)
            raise typer.Exit(1)

        rows: list[dict[str, str | float]] = []
        for t in traces:
            tid = t["trace_id"]
            scores = _evaluate_single(tid)
            rows.append(
                {
                    "trace_id": tid,
                    "session_id": t.get("session_id", ""),
                    "schema": scores["schema"][0],
                    "recon": scores["reconciliation"][0],
                    "cat_q": scores["category_quality"][0],
                    "overall": scores["composite"][0],
                }
            )
        _print_multi_report(rows)
