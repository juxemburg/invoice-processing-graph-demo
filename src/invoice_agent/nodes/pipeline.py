"""GraphState and all graph nodes (thin orchestration wrappers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_graph import BaseNode, End, GraphRunContext

from invoice_agent.models.schemas import (
    AggregationResult,
    ExtractedInvoice,
    FinalReport,
    NormalizedFields,
    ProcessedInvoice,
    ValidationResult,
)
from invoice_agent.services.aggregator import aggregate
from invoice_agent.services.categorizer import categorize_invoice
from invoice_agent.services.extractor import extract_invoice
from invoice_agent.services.normalizer import (
    infer_currency,
    parse_amount,
    parse_date,
)
from invoice_agent.services.reporter import build_report
from invoice_agent.services.validator import validate_extraction
from invoice_agent.tracing import log_span_output, traced_node

# ── Shared graph state ─────────────────────────────────────────────────────────
# Holds only what must persist across the whole run:
# the list of paths to iterate over and the accumulating results.


@dataclass
class GraphState:
    """Mutable state shared across all nodes in the graph."""

    image_paths: list[Path]
    current_index: int = 0
    processed: list[ProcessedInvoice] = field(default_factory=list)


# ── Node: route to next invoice or aggregate ────────────────────────────────


@dataclass
class ProcessNextInvoice(BaseNode[GraphState]):
    """Router node: dispatch to ExtractNode or AggregateNode."""

    @traced_node
    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> ExtractNode | AggregateNode:
        """Route to next invoice or aggregate if all processed."""
        if ctx.state.current_index < len(ctx.state.image_paths):
            path = ctx.state.image_paths[ctx.state.current_index]
            log_span_output(
                {
                    "decision": "process_next",
                    "invoice_file": path.name,
                    "index": ctx.state.current_index,
                    "total_invoices": len(ctx.state.image_paths),
                }
            )
            return ExtractNode(path=path)
        log_span_output(
            {
                "decision": "aggregate",
                "total_processed": len(ctx.state.processed),
            }
        )
        return AggregateNode()


# ── Node: extraction ─────────────────────────────────────────────────────────


@dataclass
class ExtractNode(BaseNode[GraphState]):
    """Call the VLM extraction service on a single image."""

    path: Path = field(default_factory=Path)

    @traced_node
    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> ValidateNode | ProcessNextInvoice:
        """Extract invoice data; on failure, record and skip."""
        try:
            extracted = await extract_invoice(self.path)
            log_span_output(
                {
                    "vendor": extracted.vendor,
                    "total_raw": extracted.total_raw,
                    "extraction_confidence": extracted.extraction_confidence,
                    "line_items_count": len(extracted.line_items),
                }
            )
            return ValidateNode(path=self.path, extracted=extracted)
        except Exception as exc:
            log_span_output(
                {
                    "error": str(exc),
                    "status": "extraction_failed",
                }
            )
            # VLM call failed after retries — log and skip this invoice
            print(f"[WARN] Extraction failed for {self.path.name}: {exc}")
            ctx.state.processed.append(
                ProcessedInvoice(
                    source_file=self.path.name,
                    vendor=None,
                    invoice_date=None,
                    invoice_number=None,
                    total=None,
                    currency=None,
                    category="Other",
                    confidence=0.0,
                    notes=f"Extraction failed: {exc}",
                    extraction_failed=True,
                )
            )
            ctx.state.current_index += 1
            return ProcessNextInvoice()


# ── Node: schema validation (guardrail) ──────────────────────────────────────


@dataclass
class ValidateNode(BaseNode[GraphState]):
    """Run schema validation guardrail on extracted data."""

    path: Path = field(default_factory=Path)
    extracted: ExtractedInvoice = field(default_factory=ExtractedInvoice)

    @traced_node
    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> NormalizeNode | ProcessNextInvoice:
        """Validate extraction; skip invoice if invalid."""
        validation = validate_extraction(self.extracted)
        log_span_output(
            {
                "is_valid": validation.is_valid,
                "missing_fields": validation.missing_fields,
                "warnings": validation.warnings,
            }
        )

        if not validation.is_valid:
            ctx.state.processed.append(
                ProcessedInvoice(
                    source_file=self.path.name,
                    vendor=self.extracted.vendor,
                    invoice_date=None,
                    invoice_number=self.extracted.invoice_number,
                    total=None,
                    currency=None,
                    category="Other",
                    confidence=self.extracted.extraction_confidence,
                    notes=(
                        "Validation failed — missing: "
                        + ", ".join(validation.missing_fields)
                    ),
                    extraction_failed=True,
                    validation_warnings=validation.warnings,
                )
            )
            ctx.state.current_index += 1
            return ProcessNextInvoice()

        return NormalizeNode(
            path=self.path,
            extracted=self.extracted,
            validation=validation,
        )


# ── Node: normalization ──────────────────────────────────────────────────────


@dataclass
class NormalizeNode(BaseNode[GraphState]):
    """Parse and normalize extracted fields."""

    path: Path = field(default_factory=Path)
    extracted: ExtractedInvoice = field(default_factory=ExtractedInvoice)
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult(True, [], [])
    )

    @traced_node
    async def run(self, ctx: GraphRunContext[GraphState]) -> CategorizeNode:
        """Normalize amount, date, and currency fields."""
        normalized = NormalizedFields(
            total=parse_amount(self.extracted.total_raw),
            invoice_date=parse_date(self.extracted.invoice_date_raw),
            currency=infer_currency(
                self.extracted.currency_raw,
                self.extracted.total_raw,
            ),
        )
        log_span_output(
            {
                "total_raw": self.extracted.total_raw,
                "total_normalized": str(normalized.total),
                "date_raw": self.extracted.invoice_date_raw,
                "date_normalized": normalized.invoice_date,
                "currency_raw": self.extracted.currency_raw,
                "currency_normalized": normalized.currency,
            }
        )
        return CategorizeNode(
            path=self.path,
            extracted=self.extracted,
            normalized=normalized,
            validation=self.validation,
        )


# ── Node: categorization ─────────────────────────────────────────────────────


@dataclass
class CategorizeNode(BaseNode[GraphState]):
    """Call the LLM categorization service."""

    path: Path = field(default_factory=Path)
    extracted: ExtractedInvoice = field(default_factory=ExtractedInvoice)
    normalized: NormalizedFields = field(
        default_factory=lambda: NormalizedFields(None, None, None)
    )
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult(True, [], [])
    )

    @traced_node
    async def run(self, ctx: GraphRunContext[GraphState]) -> ProcessNextInvoice:
        """Categorize the invoice and append to processed list."""
        cat = await categorize_invoice(self.extracted)
        log_span_output(
            {
                "category": cat.category,
                "confidence": cat.confidence,
                "notes": cat.notes,
            }
        )

        ctx.state.processed.append(
            ProcessedInvoice(
                source_file=self.path.name,
                vendor=self.extracted.vendor,
                invoice_date=self.normalized.invoice_date,
                invoice_number=self.extracted.invoice_number,
                total=self.normalized.total,
                currency=self.normalized.currency,
                category=cat.category,
                confidence=cat.confidence,
                notes=cat.notes,
                extraction_failed=False,
                validation_warnings=self.validation.warnings,
            )
        )
        ctx.state.current_index += 1
        return ProcessNextInvoice()


# ── Node: aggregation ────────────────────────────────────────────────────────


@dataclass
class AggregateNode(BaseNode[GraphState]):
    """Run deterministic Decimal aggregation."""

    @traced_node
    async def run(self, ctx: GraphRunContext[GraphState]) -> ReportNode:
        """Aggregate all processed invoices."""
        aggregation = aggregate(ctx.state.processed)
        log_span_output(
            {
                "total_spend": str(aggregation.total_spend),
                "spend_by_category": {
                    k: str(v) for k, v in aggregation.spend_by_category.items()
                },
                "invoice_count": aggregation.invoice_count,
                "failed_count": aggregation.failed_count,
            }
        )
        return ReportNode(aggregation=aggregation)


# ── Node: report ──────────────────────────────────────────────────────────────


@dataclass
class ReportNode(BaseNode[GraphState, None, FinalReport]):
    """Build the final report with LLM narrative and deterministic numbers."""

    aggregation: AggregationResult = field(
        default_factory=lambda: AggregationResult(
            total_spend=0,  # type: ignore[arg-type]
            spend_by_category={},
            invoice_count=0,
            failed_count=0,
        )
    )

    @traced_node
    async def run(self, ctx: GraphRunContext[GraphState]) -> End[FinalReport]:
        """Generate the final report and end the graph."""
        report = await build_report(ctx.state.processed, self.aggregation)
        log_span_output(
            {
                "issues_count": len(report.issues_and_assumptions),
                "invoice_count": len(report.invoices),
            }
        )
        return End(report)
