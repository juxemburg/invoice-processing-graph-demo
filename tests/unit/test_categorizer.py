"""Unit tests for pure helper in invoice_agent.services.categorizer."""

from invoice_agent.models.schemas import ExtractedInvoice
from invoice_agent.services.categorizer import build_categorization_prompt


class TestBuildCategorizationPrompt:
    def test_includes_vendor_name(self):
        # Arrange
        extracted = ExtractedInvoice(
            vendor="Delta Airlines",
            line_items=[{"description": "Flight NYC-LAX"}],
            total_raw="$450.00",
        )
        # Act
        result = build_categorization_prompt(extracted)
        # Assert
        assert "Delta Airlines" in result

    def test_includes_all_line_item_descriptions(self):
        # Arrange
        extracted = ExtractedInvoice(
            vendor="Acme",
            line_items=[
                {"description": "Office chair"},
                {"description": "Desk lamp"},
            ],
            total_raw="$300.00",
        )
        # Act
        result = build_categorization_prompt(extracted)
        # Assert
        assert "Office chair" in result
        assert "Desk lamp" in result

    def test_handles_none_vendor_with_unknown_placeholder(self):
        # Arrange
        extracted = ExtractedInvoice(vendor=None, total_raw="$100.00")
        # Act
        result = build_categorization_prompt(extracted)
        # Assert
        assert "Unknown" in result

    def test_handles_empty_line_items(self):
        # Arrange
        extracted = ExtractedInvoice(vendor="ACME", line_items=[], total_raw="$50.00")
        # Act
        result = build_categorization_prompt(extracted)
        # Assert
        assert "None" in result
