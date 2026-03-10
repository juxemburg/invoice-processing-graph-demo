"""Pure aggregation function. No LLM calls, no I/O."""

from __future__ import annotations

from decimal import Decimal

from invoice_agent.models.schemas import AggregationResult, ProcessedInvoice


def aggregate(invoices: list[ProcessedInvoice]) -> AggregationResult:
    """Sum invoice totals by category using Decimal arithmetic.

    Invoices with extraction_failed=True or total=None are excluded from
    sums and counted in failed_count.
    """
    total = Decimal("0")
    by_category: dict[str, Decimal] = {}
    failed = 0

    for inv in invoices:
        if inv.extraction_failed or inv.total is None:
            failed += 1
            continue
        total += inv.total
        by_category[inv.category] = (
            by_category.get(inv.category, Decimal("0")) + inv.total
        )

    return AggregationResult(
        total_spend=total,
        spend_by_category=by_category,
        invoice_count=len(invoices),
        failed_count=failed,
    )
