"""Unit tests for invoice_agent.services.validator."""

from invoice_agent.models.schemas import ExtractedInvoice
from invoice_agent.services.validator import validate_extraction


class TestValidateExtraction:
    def test_valid_invoice_passes(self):
        # Arrange
        extracted = ExtractedInvoice(
            vendor="ACME Corp",
            total_raw="$100.00",
            extraction_confidence=0.9,
        )
        # Act
        result = validate_extraction(extracted)
        # Assert
        assert result.is_valid is True
        assert result.missing_fields == []

    def test_missing_vendor_fails_validation(self):
        # Arrange
        extracted = ExtractedInvoice(
            vendor=None, total_raw="$100.00", extraction_confidence=0.9
        )
        # Act
        result = validate_extraction(extracted)
        # Assert
        assert result.is_valid is False
        assert "vendor" in result.missing_fields

    def test_missing_total_fails_validation(self):
        # Arrange
        extracted = ExtractedInvoice(
            vendor="ACME", total_raw=None, extraction_confidence=0.9
        )
        # Act
        result = validate_extraction(extracted)
        # Assert
        assert result.is_valid is False
        assert "total_raw" in result.missing_fields

    def test_low_confidence_fails_validation(self):
        # Arrange — confidence below 0.5 threshold
        extracted = ExtractedInvoice(
            vendor="ACME", total_raw="$100.00", extraction_confidence=0.3
        )
        # Act
        result = validate_extraction(extracted)
        # Assert
        assert result.is_valid is False
        assert "extraction_confidence_too_low" in result.missing_fields
        assert any("threshold" in w for w in result.warnings)

    def test_confidence_exactly_at_threshold_fails(self):
        # Arrange — boundary: 0.49 is below threshold (< 0.5 is the check)
        extracted = ExtractedInvoice(
            vendor="ACME", total_raw="$100.00", extraction_confidence=0.49
        )
        # Act
        result = validate_extraction(extracted)
        # Assert
        assert result.is_valid is False

    def test_confidence_at_threshold_passes(self):
        # Arrange — exactly 0.5 should pass (not < 0.5)
        extracted = ExtractedInvoice(
            vendor="ACME", total_raw="$100.00", extraction_confidence=0.5
        )
        # Act
        result = validate_extraction(extracted)
        # Assert
        assert result.is_valid is True

    def test_missing_optional_fields_produce_warnings_not_failure(self):
        # Arrange — invoice_number and invoice_date are optional
        extracted = ExtractedInvoice(
            vendor="ACME",
            total_raw="$100.00",
            invoice_number=None,
            invoice_date_raw=None,
            extraction_confidence=0.9,
        )
        # Act
        result = validate_extraction(extracted)
        # Assert
        assert result.is_valid is True
        assert len(result.warnings) > 0
