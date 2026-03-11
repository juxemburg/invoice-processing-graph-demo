"""Pure scoring functions for invoice-agent evaluation. No I/O."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, TypedDict

# ── Node name mapping ────────────────────────────────────────────────────────

_NODE_KEY_MAP: dict[str, str] = {
    "ExtractNode": "extract",
    "ValidateNode": "validate",
    "NormalizeNode": "normalize",
    "CategorizeNode": "categorize",
    "AggregateNode": "aggregate",
    "ReportNode": "report",
}


class SpanEntry(TypedDict):
    """A parsed span output paired with its Langfuse observation ID."""

    id: str
    output: dict[str, Any]


def parse_span_outputs(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Group observation outputs by node name, preserving observation IDs.

    Sorts observations by startTime, parses JSON output fields,
    and groups them into lists keyed by short node names.
    Non-node spans (e.g. "agent run", "chat claude-haiku") are ignored.

    Returns a dict like::

        {
            "extract": [SpanEntry, ...],
            "validate": [SpanEntry, ...],
            "normalize": [SpanEntry, ...],
            "categorize": [SpanEntry, ...],
            "aggregate": SpanEntry | None,
            "report": SpanEntry | None,
        }
    """
    sorted_obs = sorted(observations, key=lambda o: o.get("startTime", ""))

    result: dict[str, Any] = {
        "extract": [],
        "validate": [],
        "normalize": [],
        "categorize": [],
        "aggregate": None,
        "report": None,
    }

    for obs in sorted_obs:
        name = obs.get("name", "")
        key = _NODE_KEY_MAP.get(name)
        if key is None:
            continue

        raw_output = obs.get("output")
        if raw_output is None:
            continue

        if isinstance(raw_output, str):
            try:
                parsed = json.loads(raw_output)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            parsed = raw_output

        entry: SpanEntry = {"id": obs.get("id", ""), "output": parsed}

        if key in ("aggregate", "report"):
            result[key] = entry
        else:
            result[key].append(entry)

    return result


# ── Per-span scoring ─────────────────────────────────────────────────────────


def score_extract_span(output: dict[str, Any]) -> tuple[float, str]:
    """Score a single ExtractNode span output.

    Weighted fields:
    - vendor present & non-empty (1.0)
    - total_raw present & non-empty (1.0)
    - invoice_number present (0.5)
    - extraction_confidence >= 0.5 (1.0)

    Returns (score, detail_string). Score is weighted avg out of 3.5.
    """
    total_weight = 3.5
    points = 0.0
    issues: list[str] = []

    # vendor (weight 1.0)
    vendor = output.get("vendor")
    if vendor and str(vendor).strip():
        points += 1.0
    else:
        issues.append("missing vendor")

    # total_raw (weight 1.0)
    total_raw = output.get("total_raw")
    if total_raw and str(total_raw).strip():
        points += 1.0
    else:
        issues.append("missing total_raw")

    # invoice_number (weight 0.5)
    inv_num = output.get("invoice_number")
    if inv_num is not None:
        points += 0.5
    else:
        issues.append("missing invoice_number")

    # extraction_confidence (weight 1.0)
    confidence = output.get("extraction_confidence")
    if confidence is not None and confidence >= 0.5:
        points += 1.0
    else:
        issues.append("low confidence")

    score = round(points / total_weight, 4)
    if issues:
        detail = ", ".join(issues)
    else:
        detail = "All fields present"
    return (score, detail)


# ── Reconciliation score ─────────────────────────────────────────────────────


def score_aggregate_span(
    aggregate_output: dict[str, Any],
    normalize_outputs: list[dict[str, Any]],
    categorize_outputs: list[dict[str, Any]],
) -> tuple[float, str]:
    """Score the AggregateNode span via reconciliation checks.

    Three checks, each worth 1/3:
    1. sum of total_normalized == total_spend
    2. sum of spend_by_category values == total_spend
    3. set of categories in categorize_outputs == set of spend_by_category keys

    Returns (score, detail_string).
    """
    checks_passed = 0
    details: list[str] = []

    total_spend = Decimal(str(aggregate_output.get("total_spend", "0")))

    # Check 1: sum of normalize totals == total_spend
    invoice_sum = Decimal("0")
    for norm in normalize_outputs:
        raw = norm.get("total_normalized")
        if raw is not None:
            try:
                invoice_sum += Decimal(str(raw))
            except (InvalidOperation, ValueError):
                pass
    if invoice_sum == total_spend:
        checks_passed += 1
        details.append("invoice sum OK")
    else:
        details.append(f"invoice sum mismatch: {invoice_sum} != {total_spend}")

    # Check 2: sum of spend_by_category values == total_spend
    spend_by_cat = aggregate_output.get("spend_by_category", {})
    cat_sum = Decimal("0")
    for v in spend_by_cat.values():
        cat_sum += Decimal(str(v))
    if cat_sum == total_spend:
        checks_passed += 1
        details.append("category sum OK")
    else:
        details.append(f"category sum mismatch: {cat_sum} != {total_spend}")

    # Check 3: category keys match
    cat_keys_from_outputs = {
        c.get("category") for c in categorize_outputs if c.get("category")
    }
    cat_keys_from_agg = set(spend_by_cat.keys())
    if cat_keys_from_outputs == cat_keys_from_agg:
        checks_passed += 1
        details.append("category keys OK")
    else:
        details.append("category key mismatch")

    score = checks_passed / 3.0
    return (round(score, 4), f"{checks_passed}/3 checks passed: {'; '.join(details)}")


# ── Trace-level aggregation ─────────────────────────────────────────────────


def aggregate_trace_scores(
    extract_span_scores: list[float],
    aggregate_span_score: float | None,
    judge_scores: list[float],
) -> dict[str, tuple[float, str]]:
    """Aggregate span-level scores into trace-level dimension scores.

    Returns dict with keys: schema, reconciliation, category_quality,
    composite. Each maps to (score, detail/label).

    - schema = mean of extract_span_scores (0.0 if empty)
    - reconciliation = aggregate_span_score (0.0 if None)
    - category_quality = mean of judge_scores (0.0 if empty)
    - composite = 0.35*schema + 0.35*recon + 0.30*cat_quality
    """
    schema = (
        round(sum(extract_span_scores) / len(extract_span_scores), 4)
        if extract_span_scores
        else 0.0
    )
    reconciliation = aggregate_span_score if aggregate_span_score is not None else 0.0
    category_quality = (
        round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else 0.0
    )

    composite_score, composite_label = compute_composite_score(
        schema, reconciliation, category_quality
    )

    return {
        "schema": (schema, f"mean of {len(extract_span_scores)} extract scores"),
        "reconciliation": (
            reconciliation,
            "from aggregate span" if aggregate_span_score is not None else "no data",
        ),
        "category_quality": (
            category_quality,
            f"mean of {len(judge_scores)} judge scores",
        ),
        "composite": (composite_score, composite_label),
    }


# ── Composite score ──────────────────────────────────────────────────────────


_THRESHOLDS: list[tuple[float, str]] = [
    (0.8, "PASS"),
    (0.7, "WARN"),
    (0.5, "INVESTIGATE"),
]


def compute_composite_score(
    schema: float,
    reconciliation: float,
    category_quality: float,
) -> tuple[float, str]:
    """Weighted composite: 0.35*schema + 0.35*recon + 0.30*cat_quality.

    Thresholds: >=0.8 PASS, >=0.7 WARN, >=0.5 INVESTIGATE, <0.5 BLOCK.
    """
    score = 0.35 * schema + 0.35 * reconciliation + 0.30 * category_quality
    score = round(score, 4)

    label = "BLOCK"
    for threshold, name in _THRESHOLDS:
        if score >= threshold:
            label = name
            break

    return (score, label)
