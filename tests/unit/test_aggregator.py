"""Unit tests for invoice_agent.services.aggregator."""

from decimal import Decimal

from invoice_agent.models.schemas import ProcessedInvoice
from invoice_agent.services.aggregator import aggregate


def _make_invoice(
    source_file: str = "inv.jpg",
    total: Decimal | None = Decimal("100.00"),
    currency: str | None = "USD",
    category: str = "Office Supplies",
    extraction_failed: bool = False,
) -> ProcessedInvoice:
    """Create a ProcessedInvoice helper for tests."""
    return ProcessedInvoice(
        source_file=source_file,
        vendor="ACME Corp",
        invoice_date="2024-01-01",
        invoice_number="001",
        total=total,
        currency=currency,
        category=category,
        confidence=0.9,
        notes=None,
        extraction_failed=extraction_failed,
    )


class TestAggregate:
    def test_single_invoice_total(self):
        # Arrange
        invoices = [_make_invoice(total=Decimal("150.00"))]
        # Act
        result = aggregate(invoices)
        # Assert
        assert result.total_spend == Decimal("150.00")
        assert result.invoice_count == 1
        assert result.failed_count == 0

    def test_multiple_invoices_same_category(self):
        # Arrange
        invoices = [
            _make_invoice(total=Decimal("100.00"), category="Office Supplies"),
            _make_invoice(total=Decimal("200.00"), category="Office Supplies"),
        ]
        # Act
        result = aggregate(invoices)
        # Assert
        assert result.total_spend == Decimal("300.00")
        assert result.spend_by_category["Office Supplies"] == Decimal("300.00")

    def test_multiple_invoices_different_categories(self):
        # Arrange
        invoices = [
            _make_invoice(total=Decimal("100.00"), category="Office Supplies"),
            _make_invoice(total=Decimal("500.00"), category="Travel (air/hotel)"),
        ]
        # Act
        result = aggregate(invoices)
        # Assert
        assert result.total_spend == Decimal("600.00")
        assert result.spend_by_category["Office Supplies"] == Decimal("100.00")
        assert result.spend_by_category["Travel (air/hotel)"] == Decimal("500.00")

    def test_failed_invoice_excluded_from_total(self):
        # Arrange
        invoices = [
            _make_invoice(total=Decimal("100.00"), extraction_failed=False),
            _make_invoice(total=Decimal("999.00"), extraction_failed=True),
        ]
        # Act
        result = aggregate(invoices)
        # Assert
        assert result.total_spend == Decimal("100.00")
        assert result.failed_count == 1

    def test_invoice_with_none_total_counted_as_failed(self):
        # Arrange
        invoices = [
            _make_invoice(total=Decimal("100.00")),
            _make_invoice(total=None),
        ]
        # Act
        result = aggregate(invoices)
        # Assert
        assert result.total_spend == Decimal("100.00")
        assert result.failed_count == 1

    def test_empty_invoice_list_returns_zero_totals(self):
        # Arrange
        invoices: list[ProcessedInvoice] = []
        # Act
        result = aggregate(invoices)
        # Assert
        assert result.total_spend == Decimal("0")
        assert result.spend_by_category == {}
        assert result.invoice_count == 0
        assert result.failed_count == 0
