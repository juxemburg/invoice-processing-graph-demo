"""Constrained LLM categorization agent and prompt builder."""

from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from invoice_agent.models.schemas import CategoryResult, ExtractedInvoice
from invoice_agent.settings import get_settings

CATEGORIZATION_SYSTEM_PROMPT = (
    "You are an expense categorization assistant. "
    "Assign exactly one expense category from the allowed list "
    "based on the vendor name and line item descriptions. "
    "Allowed categories: Travel (air/hotel), Meals & Entertainment, "
    "Software / Subscriptions, Professional Services, Office Supplies, "
    "Shipping / Postage, Utilities, Other. "
    "If category is 'Other', you MUST provide an explanatory note."
)


@lru_cache(maxsize=1)
def _get_categorization_agent() -> Agent[None, CategoryResult]:
    """Create and cache the categorization agent."""
    return Agent(
        get_settings().categorization_model,
        output_type=CategoryResult,
        retries=2,
        system_prompt=CATEGORIZATION_SYSTEM_PROMPT,
        defer_model_check=True,
    )


def build_categorization_prompt(extracted: ExtractedInvoice) -> str:
    """Build the user prompt string for the categorization agent."""
    line_desc = "; ".join(
        item.get("description", "") for item in (extracted.line_items or [])
    )
    return (
        f"Vendor: {extracted.vendor or 'Unknown'}\n"
        f"Line items: {line_desc or 'None'}\n"
        f"Total: {extracted.total_raw or 'Unknown'}"
    )


async def categorize_invoice(
    extracted: ExtractedInvoice,
) -> CategoryResult:
    """Run constrained categorization via LLM.

    Confidence thresholding: if extraction_confidence < 0.5, the VLM output
    is unreliable. Skip the LLM call and return 'Other' with an explicit note.
    This decision is already recorded in the ValidationResult warnings before
    this function is called — it does not need to be re-detected here.
    """
    prompt = build_categorization_prompt(extracted)
    result = await _get_categorization_agent().run(prompt)
    return result.output
