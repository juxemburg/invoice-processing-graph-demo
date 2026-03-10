"""Schema validation guardrail.

Checks extracted invoice fields for completeness and logs confidence
thresholding decisions explicitly.
"""

from __future__ import annotations

from invoice_agent.models.schemas import ExtractedInvoice, ValidationResult

# Fields required for a fully scoreable invoice
REQUIRED_FIELDS = ("vendor", "total_raw")
LOW_CONFIDENCE_THRESHOLD = 0.5


def validate_extraction(extracted: ExtractedInvoice) -> ValidationResult:
    """Check that required fields are present and confidence is acceptable.

    Returns a ValidationResult with:
    - is_valid: False if any required field is missing OR confidence is below
      threshold (invoice will be skipped for categorization)
    - missing_fields: list of required fields that are absent
    - warnings: non-blocking issues (e.g. missing invoice_number)
    """
    missing: list[str] = []
    warnings_list: list[str] = []

    for field_name in REQUIRED_FIELDS:
        value = getattr(extracted, field_name, None)
        if not value:
            missing.append(field_name)

    # Optional fields — warn but don't fail
    if not extracted.invoice_date_raw:
        warnings_list.append("invoice_date_raw is missing")
    if not extracted.invoice_number:
        warnings_list.append("invoice_number is missing")

    # Confidence thresholding — explicit and visible
    if extracted.extraction_confidence < LOW_CONFIDENCE_THRESHOLD:
        missing.append("extraction_confidence_too_low")
        warnings_list.append(
            f"Extraction confidence {extracted.extraction_confidence:.2f} "
            f"is below threshold {LOW_CONFIDENCE_THRESHOLD} "
            "— invoice flagged for manual review"
        )

    return ValidationResult(
        is_valid=len(missing) == 0,
        missing_fields=missing,
        warnings=warnings_list,
    )
