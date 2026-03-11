"""Unit tests for invoice_agent.services.eval_scoring."""

from __future__ import annotations

from typing import Any

from invoice_agent.services.eval_scoring import (
    aggregate_trace_scores,
    compute_composite_score,
    parse_span_outputs,
    score_aggregate_span,
    score_extract_span,
)

# ── Fixture factories ────────────────────────────────────────────────────────


def make_extract_output(
    vendor: str = "ACME Corp",
    total_raw: str = "$100.00",
    confidence: float = 0.9,
    line_items_count: int = 1,
    invoice_number: str | None = "INV-001",
) -> dict[str, Any]:
    """Create a mock ExtractNode output dict."""
    return {
        "vendor": vendor,
        "total_raw": total_raw,
        "extraction_confidence": confidence,
        "line_items": [{"desc": "item"}] * line_items_count,
        "invoice_number": invoice_number,
    }


def make_validate_output(
    is_valid: bool = True,
    missing: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Create a mock ValidateNode output dict."""
    return {
        "is_valid": is_valid,
        "missing_fields": missing or [],
        "warnings": warnings or [],
    }


def make_normalize_output(
    total_raw: str = "$100.00",
    total_normalized: str | None = "100.00",
    date_raw: str = "01/15/2024",
    date_normalized: str | None = "2024-01-15",
    currency_raw: str = "USD",
    currency_normalized: str | None = "USD",
) -> dict[str, Any]:
    """Create a mock NormalizeNode output dict."""
    return {
        "total_raw": total_raw,
        "total_normalized": total_normalized,
        "date_raw": date_raw,
        "date_normalized": date_normalized,
        "currency_raw": currency_raw,
        "currency_normalized": currency_normalized,
    }


def make_categorize_output(
    category: str = "Office Supplies",
    confidence: float = 0.9,
    notes: str | None = None,
) -> dict[str, Any]:
    """Create a mock CategorizeNode output dict."""
    return {
        "category": category,
        "confidence": confidence,
        "notes": notes,
    }


def make_aggregate_output(
    total_spend: str = "300.00",
    spend_by_category: dict[str, str] | None = None,
    invoice_count: int = 3,
    failed_count: int = 0,
) -> dict[str, Any]:
    """Create a mock AggregateNode output dict."""
    return {
        "total_spend": total_spend,
        "spend_by_category": spend_by_category or {"Office Supplies": "300.00"},
        "invoice_count": invoice_count,
        "failed_count": failed_count,
    }


def make_judge_score(
    value: float = 0.8,
    observation_id: str = "obs-1",
    name: str = "categorizer_evaluator_v2",
) -> dict[str, Any]:
    """Create a mock Langfuse judge score dict."""
    return {
        "name": name,
        "value": value,
        "observationId": observation_id,
    }


def make_observation(
    name: str = "ExtractNode",
    output_dict: dict[str, Any] | None = None,
    obs_id: str = "obs-1",
    start_time: str = "2024-01-01T00:00:00Z",
) -> dict[str, Any]:
    """Create a mock Langfuse observation dict."""
    import json

    return {
        "id": obs_id,
        "name": name,
        "output": json.dumps(output_dict) if output_dict is not None else None,
        "startTime": start_time,
    }


# ── TestParseSpanOutputs ─────────────────────────────────────────────────────


class TestParseSpanOutputs:
    def test_groups_observations_by_node_name(self):
        # Arrange
        obs = [
            make_observation(
                "ExtractNode", {"vendor": "A"}, start_time="2024-01-01T00:00:01Z"
            ),
            make_observation(
                "ValidateNode", {"is_valid": True}, start_time="2024-01-01T00:00:02Z"
            ),
            make_observation(
                "NormalizeNode",
                {"total_normalized": "100"},
                start_time="2024-01-01T00:00:03Z",
            ),
            make_observation(
                "CategorizeNode",
                {"category": "Other"},
                start_time="2024-01-01T00:00:04Z",
            ),
        ]
        # Act
        result = parse_span_outputs(obs)
        # Assert
        assert len(result["extract"]) == 1
        assert len(result["validate"]) == 1
        assert len(result["normalize"]) == 1
        assert len(result["categorize"]) == 1

    def test_parses_json_output_strings(self):
        # Arrange
        obs = [make_observation("ExtractNode", {"vendor": "ACME"})]
        # Act
        result = parse_span_outputs(obs)
        # Assert
        assert result["extract"][0]["output"]["vendor"] == "ACME"

    def test_skips_null_outputs(self):
        # Arrange
        obs = [
            {
                "id": "1",
                "name": "ExtractNode",
                "output": None,
                "startTime": "2024-01-01T00:00:00Z",
            }
        ]
        # Act
        result = parse_span_outputs(obs)
        # Assert
        assert result["extract"] == []

    def test_ignores_non_node_spans(self):
        # Arrange
        obs = [
            make_observation("agent run", {"some": "data"}),
            make_observation("chat claude-haiku", {"msg": "hi"}),
            make_observation("ExtractNode", {"vendor": "ACME"}),
        ]
        # Act
        result = parse_span_outputs(obs)
        # Assert
        assert len(result["extract"]) == 1
        assert result["aggregate"] is None
        assert result["report"] is None

    def test_preserves_observation_ids(self):
        # Arrange
        obs = [
            make_observation(
                "ExtractNode",
                {"vendor": "A"},
                obs_id="ext-1",
                start_time="2024-01-01T00:00:01Z",
            ),
            make_observation(
                "AggregateNode",
                {"total_spend": "100"},
                obs_id="agg-1",
                start_time="2024-01-01T00:00:02Z",
            ),
        ]
        # Act
        result = parse_span_outputs(obs)
        # Assert
        assert result["extract"][0]["id"] == "ext-1"
        assert result["aggregate"]["id"] == "agg-1"


# ── TestScoreExtractSpan ─────────────────────────────────────────────────────


class TestScoreExtractSpan:
    def test_perfect_score(self):
        # Arrange — all fields present, confidence >= 0.5
        output = make_extract_output()
        # Act
        score, detail = score_extract_span(output)
        # Assert
        assert score == 1.0
        assert detail == "All fields present"

    def test_missing_vendor_reduces_score(self):
        # Arrange — vendor is None
        output = make_extract_output(vendor="")
        # Act
        score, detail = score_extract_span(output)
        # Assert
        assert score < 1.0
        assert "missing vendor" in detail

    def test_missing_total_raw_reduces_score(self):
        # Arrange — total_raw is None
        output = make_extract_output(total_raw="")
        # Act
        score, detail = score_extract_span(output)
        # Assert
        assert score < 1.0
        assert "missing total_raw" in detail

    def test_missing_invoice_number_half_weight(self):
        # Arrange — with and without invoice_number
        with_num = make_extract_output(invoice_number="INV-1")
        without_num = make_extract_output(invoice_number=None)
        # Act
        score_with, _ = score_extract_span(with_num)
        score_without, _ = score_extract_span(without_num)
        # Assert — difference ≈ 0.5/3.5 ≈ 0.143
        diff = score_with - score_without
        assert 0.14 < diff < 0.15

    def test_low_confidence_reduces_score(self):
        # Arrange — extraction_confidence = 0.3
        output = make_extract_output(confidence=0.3)
        # Act
        score, detail = score_extract_span(output)
        # Assert
        assert score < 1.0
        assert "low confidence" in detail

    def test_all_missing_returns_zero(self):
        # Arrange — empty dict
        output: dict[str, Any] = {}
        # Act
        score, _ = score_extract_span(output)
        # Assert
        assert score == 0.0


# ── TestScoreAggregateSpan ──────────────────────────────────────────────────


class TestScoreAggregateSpan:
    def test_perfect_reconciliation(self):
        # Arrange
        agg = make_aggregate_output(
            total_spend="200.00",
            spend_by_category={
                "Office Supplies": "100.00",
                "Travel (air/hotel)": "100.00",
            },
        )
        norms = [
            make_normalize_output(total_normalized="100.00"),
            make_normalize_output(total_normalized="100.00"),
        ]
        cats = [
            make_categorize_output(category="Office Supplies"),
            make_categorize_output(category="Travel (air/hotel)"),
        ]
        # Act
        score, detail = score_aggregate_span(agg, norms, cats)
        # Assert
        assert score == 1.0
        assert "3/3" in detail

    def test_invoice_sum_mismatch(self):
        # Arrange
        agg = make_aggregate_output(
            total_spend="200.00",
            spend_by_category={"Office Supplies": "200.00"},
        )
        norms = [make_normalize_output(total_normalized="150.00")]
        cats = [make_categorize_output(category="Office Supplies")]
        # Act
        score, detail = score_aggregate_span(agg, norms, cats)
        # Assert
        assert score < 1.0
        assert "invoice sum mismatch" in detail

    def test_category_sum_mismatch(self):
        # Arrange
        agg = make_aggregate_output(
            total_spend="200.00",
            spend_by_category={"Office Supplies": "150.00"},
        )
        norms = [make_normalize_output(total_normalized="200.00")]
        cats = [make_categorize_output(category="Office Supplies")]
        # Act
        score, detail = score_aggregate_span(agg, norms, cats)
        # Assert
        assert score < 1.0
        assert "category sum mismatch" in detail

    def test_category_key_mismatch(self):
        # Arrange
        agg = make_aggregate_output(
            total_spend="100.00",
            spend_by_category={"Travel (air/hotel)": "100.00"},
        )
        norms = [make_normalize_output(total_normalized="100.00")]
        cats = [make_categorize_output(category="Office Supplies")]
        # Act
        score, detail = score_aggregate_span(agg, norms, cats)
        # Assert
        assert score < 1.0
        assert "category key mismatch" in detail

    def test_decimal_precision(self):
        # Arrange — values with many decimal places
        agg = make_aggregate_output(
            total_spend="100.10",
            spend_by_category={"Office Supplies": "100.10"},
        )
        norms = [make_normalize_output(total_normalized="100.10")]
        cats = [make_categorize_output(category="Office Supplies")]
        # Act
        score, _ = score_aggregate_span(agg, norms, cats)
        # Assert
        assert score == 1.0


# ── TestAggregateTraceScores ────────────────────────────────────────────────


class TestAggregateTraceScores:
    def test_all_perfect(self):
        # Arrange
        result = aggregate_trace_scores([1.0, 1.0], 1.0, [0.9, 0.8])
        # Assert
        assert result["schema"][0] == 1.0
        assert result["reconciliation"][0] == 1.0
        assert result["category_quality"][0] == 0.85
        # composite = 0.35*1.0 + 0.35*1.0 + 0.30*0.85 = 0.955
        assert result["composite"][0] == 0.955

    def test_empty_extracts_schema_zero(self):
        # Arrange
        result = aggregate_trace_scores([], 1.0, [0.8])
        # Assert
        assert result["schema"][0] == 0.0

    def test_no_aggregate_recon_zero(self):
        # Arrange
        result = aggregate_trace_scores([1.0], None, [0.8])
        # Assert
        assert result["reconciliation"][0] == 0.0

    def test_no_judge_scores_cat_quality_zero(self):
        # Arrange
        result = aggregate_trace_scores([1.0], 1.0, [])
        # Assert
        assert result["category_quality"][0] == 0.0

    def test_composite_uses_correct_weights(self):
        # Arrange — known values: schema=0.8, recon=0.6, cat=0.5
        result = aggregate_trace_scores([0.8], 0.6, [0.5])
        # Assert — composite = 0.35*0.8 + 0.35*0.6 + 0.30*0.5 = 0.28+0.21+0.15 = 0.64
        assert result["composite"][0] == 0.64


# ── TestComputeCompositeScore ────────────────────────────────────────────────


class TestComputeCompositeScore:
    def test_weighted_formula(self):
        # Act
        score, _ = compute_composite_score(1.0, 1.0, 1.0)
        # Assert
        assert score == 1.0

    def test_pass_threshold(self):
        # Act
        _, label = compute_composite_score(0.9, 0.9, 0.9)
        # Assert
        assert label == "PASS"

    def test_warn_threshold(self):
        # Act — 0.35*0.7 + 0.35*0.8 + 0.30*0.6 = 0.245+0.28+0.18 = 0.705
        _, label = compute_composite_score(0.7, 0.8, 0.6)
        # Assert
        assert label == "WARN"

    def test_investigate_threshold(self):
        # Act — 0.35*0.5 + 0.35*0.6 + 0.30*0.5 = 0.175+0.21+0.15 = 0.535
        _, label = compute_composite_score(0.5, 0.6, 0.5)
        # Assert
        assert label == "INVESTIGATE"

    def test_block_threshold(self):
        # Act
        _, label = compute_composite_score(0.2, 0.3, 0.1)
        # Assert
        assert label == "BLOCK"

    def test_weights_sum_to_one(self):
        # Assert — weights 0.35 + 0.35 + 0.30 = 1.0
        assert 0.35 + 0.35 + 0.30 == 1.0
        # Also verify via computation
        score, _ = compute_composite_score(1.0, 1.0, 1.0)
        assert score == 1.0
