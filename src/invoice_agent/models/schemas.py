"""Pydantic models, dataclasses, and type aliases for the invoice agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

# ── Category taxonomy ─────────────────────────────────────────────────────────

ExpenseCategory = Literal[
    "Travel (air/hotel)",
    "Meals & Entertainment",
    "Software / Subscriptions",
    "Professional Services",
    "Office Supplies",
    "Shipping / Postage",
    "Utilities",
    "Other",
]

VALID_CATEGORIES: tuple[str, ...] = (
    "Travel (air/hotel)",
    "Meals & Entertainment",
    "Software / Subscriptions",
    "Professional Services",
    "Office Supplies",
    "Shipping / Postage",
    "Utilities",
    "Other",
)


# ── LLM structured output schemas ─────────────────────────────────────────────


class ExtractedInvoice(BaseModel):
    """Raw fields as returned by the VLM extraction call."""

    vendor: Optional[str] = None
    invoice_date_raw: Optional[str] = None
    invoice_number: Optional[str] = None
    line_items: list[dict] = []  # type: ignore[type-arg]
    total_raw: Optional[str] = None
    currency_raw: Optional[str] = None
    extraction_confidence: float = 0.5
    extraction_notes: Optional[str] = None


class CategoryResult(BaseModel):
    """Output of the categorization LLM call."""

    category: ExpenseCategory
    confidence: float
    notes: Optional[str] = None

    @model_validator(mode="after")
    def other_requires_notes(self) -> CategoryResult:
        """Require an explanatory note when category is 'Other'."""
        if self.category == "Other" and not self.notes:
            raise ValueError("'Other' category must include an explanatory note")
        return self


class ReportNarrative(BaseModel):
    """LLM-generated narrative for the Issues & Assumptions section."""

    issues_and_assumptions: list[str]


# ── Processing intermediates ──────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Output of the schema validation guardrail."""

    is_valid: bool
    missing_fields: list[str]
    warnings: list[str]


@dataclass
class NormalizedFields:
    """Output of the normalizer service."""

    total: Optional[Decimal]
    invoice_date: Optional[str]  # ISO 8601, or None if unparseable
    currency: Optional[str]  # ISO 4217 best-effort


@dataclass
class ProcessedInvoice:
    """Fully processed invoice, ready for aggregation."""

    source_file: str
    vendor: Optional[str]
    invoice_date: Optional[str]
    invoice_number: Optional[str]
    total: Optional[Decimal]
    currency: Optional[str]
    category: ExpenseCategory
    confidence: float
    notes: Optional[str]
    extraction_failed: bool = False
    validation_warnings: list[str] = field(default_factory=list)


# ── Aggregation & report ───────────────────────────────────────────────────────


@dataclass
class AggregationResult:
    """Result of deterministic Decimal aggregation."""

    total_spend: Decimal
    spend_by_category: dict[str, Decimal]
    invoice_count: int
    failed_count: int


@dataclass
class FinalReport:
    """The structured JSON report output."""

    total_spend: str
    spend_by_category: dict[str, str]
    invoices: list[dict]  # type: ignore[type-arg]
    issues_and_assumptions: list[str]
