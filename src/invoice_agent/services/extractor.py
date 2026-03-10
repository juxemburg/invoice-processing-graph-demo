"""VLM extraction agent and image encoding helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_ai import Agent, BinaryContent

from invoice_agent.models.schemas import ExtractedInvoice
from invoice_agent.settings import get_settings

EXTRACTION_SYSTEM_PROMPT = (
    "You are an invoice data extraction assistant. "
    "Extract all invoice fields from the provided image. "
    "If a field is not visible or legible, return null. "
    "Set extraction_confidence to a value between 0 and 1 reflecting "
    "how complete and clear the extraction is."
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
