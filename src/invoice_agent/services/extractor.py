"""VLM extraction agent and image encoding helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_ai import Agent, BinaryContent

from invoice_agent.models.schemas import ExtractedInvoice
from invoice_agent.settings import get_settings

EXTRACTION_SYSTEM_PROMPT = (
    "You are an invoice data extraction and classification assistant. "
    "Your job is to extract structured data from invoice images and categorize each invoice accurately.\n\n"
    "## Extract the following fields:\n"
    "- vendor: The seller/vendor name as it appears on the invoice\n"
    "- invoice_date_raw: Date of the invoice exactly as printed (e.g. '26.06.2004', '04/13/2013')\n"
    "- invoice_number: Invoice or reference number as printed (null if not present)\n"
    "- total_raw: The final total amount due as a string exactly as printed (e.g. '£155.45', '$6,204.19'). "
    "Prefer the final payable amount after discounts and taxes over subtotals or net amounts\n"
    "- currency_raw: Currency as it appears on the invoice — symbol (e.g. '£', '$') or code (e.g. 'GBP', 'USD'). "
    "Infer from symbols if no explicit code is present\n"
    "- line_items: List of line items, each as a dict with keys: "
    "'description', 'quantity', 'unit_cost', 'line_total'\n\n"
    "## Categorize each invoice into exactly one of the following categories:\n"
    "- Travel (air/hotel)\n"
    "- Meals & Entertainment\n"
    "- Software / Subscriptions\n"
    "- Professional Services\n"
    "- Office Supplies\n"
    "- Shipping / Postage\n"
    "- Utilities\n"
    "- Other (must include an explanatory note in extraction_notes)\n\n"
    "You may use internal subcategories in your reasoning, but the final category "
    "must be added to line_items or extraction_notes as appropriate.\n\n"
    "## Confidence & Notes:\n"
    "- Set extraction_confidence to a value between 0 and 1 reflecting how complete and legible the extraction is.\n"
    "- Use extraction_notes to flag: ambiguities, missing fields, conflicting data, assumptions made "
    "(e.g. inferred currency, unclear totals), category reasoning, or any uncertainty. "
    "Always populate extraction_notes if extraction_confidence is below 0.8.\n\n"
    "## Rules:\n"
    "- If a field is not visible or legible, return null — do not guess.\n"
    "- Preserve raw values exactly as printed; do not reformat dates, currencies, or numbers.\n"
    "- If multiple totals exist (subtotal, net, gross), always extract the final payable gross amount.\n"
)


@lru_cache(maxsize=1)
def _get_extraction_agent() -> Agent[None, ExtractedInvoice]:
    """Create and cache the extraction agent."""
    return Agent(
        get_settings().extraction_model,
        output_type=ExtractedInvoice,
        retries=2,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        defer_model_check=True,
    )


MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "pdf": "application/pdf",
}


def get_mime_type(path: Path) -> str:
    """Return MIME type string for a given image path."""
    return MIME_MAP.get(path.suffix.lstrip(".").lower(), "image/png")


def read_image_bytes(path: Path) -> bytes:
    """Read raw bytes from an image file."""
    return path.read_bytes()


async def extract_invoice(path: Path) -> ExtractedInvoice:
    """Run VLM extraction on a single invoice image.

    Uses BinaryContent to pass the image to pydantic-ai — this is the
    correct input type for binary image data (not a raw dict list).
    """
    image_bytes = read_image_bytes(path)
    mime = get_mime_type(path)
    content = BinaryContent(data=image_bytes, media_type=mime)
    result = await _get_extraction_agent().run([content])
    return result.output
