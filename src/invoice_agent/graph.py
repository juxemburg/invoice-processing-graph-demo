"""Graph wiring for the invoice processing pipeline."""

from __future__ import annotations

from pydantic_graph import Graph

from invoice_agent.nodes.pipeline import (
    AggregateNode,
    CategorizeNode,
    ExtractNode,
    NormalizeNode,
    ProcessNextInvoice,
    ReportNode,
    ValidateNode,
)

invoice_graph = Graph(
    nodes=[
        ProcessNextInvoice,
        ExtractNode,
        ValidateNode,
        NormalizeNode,
        CategorizeNode,
        AggregateNode,
        ReportNode,
    ]
)
