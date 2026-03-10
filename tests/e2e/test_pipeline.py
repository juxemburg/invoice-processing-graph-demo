"""E2E test: run the full graph against invoice_samples/.

Requires ANTHROPIC_API_KEY. Skipped automatically when key is absent.
"""

import asyncio
import os
from decimal import Decimal
from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).parents[2] / "invoice_samples"

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping E2E test",
)


@pytest.fixture(scope="module")
def report():
    """Run the full graph and return the FinalReport."""
    from invoice_agent.graph import invoice_graph
    from invoice_agent.nodes.pipeline import GraphState, ProcessNextInvoice

    image_paths = sorted(
        p
        for p in SAMPLES_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    state = GraphState(image_paths=image_paths)
    result = asyncio.run(invoice_graph.run(ProcessNextInvoice(), state=state))
    return result.output


class TestE2EPipeline:
    def test_report_contains_both_invoices(self, report):
        # Arrange — 2 images in invoice_samples/
        # Act
        count = len(report.invoices)
        # Assert
        assert count == 2

    def test_all_invoices_have_required_fields(self, report):
        # Arrange
        required = {
            "source_file",
            "vendor",
            "invoice_date",
            "total",
            "category",
        }
        # Act / Assert
        for inv in report.invoices:
            for f in required:
                assert inv.get(f) is not None, f"Missing '{f}' in {inv['source_file']}"

    def test_total_spend_equals_sum_of_invoice_totals(self, report):
        # Arrange
        invoice_sum = sum(Decimal(inv["total"]) for inv in report.invoices)
        # Act
        reported_total = Decimal(report.total_spend)
        # Assert
        assert reported_total == invoice_sum

    def test_categories_are_valid(self, report):
        # Arrange
        from invoice_agent.models.schemas import VALID_CATEGORIES

        # Act / Assert
        for inv in report.invoices:
            assert inv["category"] in VALID_CATEGORIES, (
                f"Invalid category '{inv['category']}' in {inv['source_file']}"
            )

    def test_issues_and_assumptions_is_non_empty_list(self, report):
        # Arrange / Act
        issues = report.issues_and_assumptions
        # Assert
        assert isinstance(issues, list)
        assert len(issues) > 0

    def test_mixed_currency_mentioned_in_issues(self, report):
        # Arrange — samples have GBP (052.png) + USD (batch1-0001.jpg)
        # Act / Assert
        assert any(
            "currencies" in issue.lower() or "GBP" in issue or "USD" in issue
            for issue in report.issues_and_assumptions
        )
