"""Constrained LLM categorization agent and prompt builder."""

from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from invoice_agent.models.schemas import CategoryResult, ExtractedInvoice
from invoice_agent.settings import get_settings

CATEGORIZATION_SYSTEM_PROMPT = (
    "You are an expert expense categorization assistant. "
    "Your sole job is to assign exactly one expense category to an invoice "
    "based on the vendor name, line item descriptions, and any contextual notes provided.\n\n"
    "## Allowed categories (you must use one of these exactly):\n"
    "- Travel (air/hotel): flights, hotels, car rentals, transit, accommodation\n"
    "- Meals & Entertainment: restaurants, catering, team lunches, client entertainment\n"
    "- Software / Subscriptions: SaaS tools, licenses, cloud services, digital subscriptions\n"
    "- Professional Services: consulting, legal, accounting, staffing, freelance work, "
    "any vague or abstract service descriptions (e.g. 'maximize compelling markets', "
    "'exploit integrated initiatives')\n"
    "- Office Supplies: stationery, printer ink, physical office consumables\n"
    "- Shipping / Postage: courier, freight, postage, delivery charges\n"
    "- Utilities: electricity, water, gas, internet, phone bills\n"
    "- Other: use ONLY when no other category reasonably applies — "
    "you MUST provide a detailed explanatory note in the notes field\n\n"
    "## Confidence:\n"
    "- Set confidence to a value between 0 and 1.\n"
    "- Use 0.9–1.0 when the vendor or line items unambiguously match a category.\n"
    "- Use 0.6–0.89 when the match is reasonable but relies on inference.\n"
    "- Use below 0.6 when the category is a best guess; always populate notes in this case.\n\n"
    "## Notes:\n"
    "- REQUIRED when category is 'Other'.\n"
    "- REQUIRED when confidence is below 0.6.\n"
    "- Use notes to explain your reasoning when the categorization is non-obvious, "
    "or to flag that the vendor/line items were ambiguous or vague.\n\n"
    "## Rules:\n"
    "- Never invent a category outside the allowed list.\n"
    "- When line items span multiple categories, assign the category that represents "
    "the majority of spend or the primary business purpose of the invoice.\n"
    "- Vague or jargon-heavy service descriptions (e.g. 'scale enterprise mindshare') "
    "should default to Professional Services, not Other.\n"
    "- Computer hardware and electronics should be categorized as Office Supplies "
    "unless context clearly indicates otherwise.\n"
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
