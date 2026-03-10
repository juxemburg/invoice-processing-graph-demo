"""Pure normalization functions. No LLM calls, no I/O, no graph state."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Optional


def parse_amount(raw: str | None) -> Optional[Decimal]:
    """Parse a raw amount string to Decimal.

    Handles:
    - Currency symbols: $, £, €, ¥
    - US format: "1,234.56"
    - EU format with dot thousands: "1.234,56"
    - EU format with space thousands: "1 394,67" (as in batch1-0001.jpg)
    - Negative in parentheses: "(500.00)"

    Returns None if the string is None or unparseable.
    """
    if not raw:
        return None

    s = raw.strip()

    # Negative parentheses: (500.00) → -500.00
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = "-" + s[1:-1]

    # Strip currency symbols and all whitespace (including non-breaking \u00a0)
    s = re.sub(r"[£$€¥\s\u00a0]", "", s)

    if not s or s == "-":
        return None

    # Detect EU format: last separator is a comma with exactly 2 decimal digits
    if re.search(r",\d{2}$", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        # US format or plain number: remove thousands commas
        s = s.replace(",", "")

    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def parse_date(raw: str | None) -> Optional[str]:
    """Parse a raw date string to ISO 8601 (YYYY-MM-DD).

    ASSUMPTION: For slash-separated dates (e.g. 04/03/2013), MM/DD/YYYY is
    tried before DD/MM/YYYY. This means 04/03/2013 parses as April 3rd, not
    March 4th. This assumption is documented in the output's
    issues_and_assumptions section.

    Returns None if no format matches.
    """
    if not raw:
        return None

    formats = [
        "%d.%m.%Y",  # 26.06.2004 — unambiguous, try first
        "%Y-%m-%d",  # 2024-01-15 — unambiguous
        "%m/%d/%Y",  # 04/13/2013 — US, assumed over DD/MM for slash dates
        "%d/%m/%Y",  # 13/04/2013 — EU slash (fallback only)
        "%B %d, %Y",  # January 15, 2024
        "%b %d, %Y",  # Jan 15, 2024
        "%d %B %Y",  # 15 January 2024
        "%d %b %Y",  # 15 Jan 2024
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def infer_currency(currency_raw: str | None, total_raw: str | None) -> Optional[str]:
    """Infer ISO 4217 currency code from raw currency hint and/or total string.

    Returns None if currency cannot be determined.
    """
    combined = f"{currency_raw or ''} {total_raw or ''}".upper()
    if "GBP" in combined or "£" in combined:
        return "GBP"
    if "EUR" in combined or "€" in combined:
        return "EUR"
    if "USD" in combined or "$" in combined:
        return "USD"
    return None
