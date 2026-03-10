"""Integration tests for graph node state transitions.

No real API calls — tests verify that nodes write correct data to state
and return the correct next node type.
"""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from invoice_agent.models.schemas import (
    ExtractedInvoice,
    ValidationResult,
)
from invoice_agent.nodes.pipeline import (
    AggregateNode,
    CategorizeNode,
    ExtractNode,
    GraphState,
    NormalizeNode,
    ProcessNextInvoice,
    ValidateNode,
)


@pytest.fixture
def fake_ctx(tmp_path: Path):
    """Minimal context object wrapping a GraphState with two fake paths."""
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    img1.write_bytes(b"fake")
    img2.write_bytes(b"fake")
    state = GraphState(image_paths=[img1, img2])

    @dataclass
    class Ctx:
        state: GraphState

    return Ctx(state=state)


class TestProcessNextInvoice:
    async def test_returns_extract_node_with_path_when_invoices_remain(self, fake_ctx):
        # Arrange
        node = ProcessNextInvoice()
        # Act
        result = await node.run(fake_ctx)
        # Assert
        assert isinstance(result, ExtractNode)
        assert result.path == fake_ctx.state.image_paths[0]

    async def test_returns_aggregate_node_when_index_past_end(self, fake_ctx):
        # Arrange
        fake_ctx.state.current_index = 2
        node = ProcessNextInvoice()
        # Act
        result = await node.run(fake_ctx)
        # Assert
        assert isinstance(result, AggregateNode)


class TestExtractNode:
    async def test_successful_extraction_transitions_to_validate(self, fake_ctx):
        # Arrange
        extracted = ExtractedInvoice(
            vendor="ACME",
            total_raw="$100.00",
            extraction_confidence=0.9,
        )
        node = ExtractNode(path=fake_ctx.state.image_paths[0])
        # Act
        with patch(
            "invoice_agent.nodes.pipeline.extract_invoice",
            new=AsyncMock(return_value=extracted),
        ):
            result = await node.run(fake_ctx)
        # Assert
        assert isinstance(result, ValidateNode)
        assert result.extracted == extracted

    async def test_extraction_failure_appends_failed_invoice_and_advances(
        self, fake_ctx
    ):
        # Arrange
        node = ExtractNode(path=fake_ctx.state.image_paths[0])
        # Act
        with patch(
            "invoice_agent.nodes.pipeline.extract_invoice",
            new=AsyncMock(side_effect=Exception("API timeout")),
        ):
            result = await node.run(fake_ctx)
        # Assert — invoice appended as failed, index incremented
        assert isinstance(result, ProcessNextInvoice)
        assert len(fake_ctx.state.processed) == 1
        assert fake_ctx.state.processed[0].extraction_failed is True
        assert fake_ctx.state.current_index == 1


class TestValidateNode:
    async def test_valid_extraction_transitions_to_normalize(self, fake_ctx):
        # Arrange
        extracted = ExtractedInvoice(
            vendor="ACME",
            total_raw="$100.00",
            extraction_confidence=0.9,
        )
        node = ValidateNode(path=fake_ctx.state.image_paths[0], extracted=extracted)
        # Act
        result = await node.run(fake_ctx)
        # Assert
        assert isinstance(result, NormalizeNode)
        assert result.extracted == extracted

    async def test_invalid_extraction_appends_failed_invoice_and_advances(
        self, fake_ctx
    ):
        # Arrange — missing vendor and total
        extracted = ExtractedInvoice(
            vendor=None, total_raw=None, extraction_confidence=0.9
        )
        node = ValidateNode(path=fake_ctx.state.image_paths[0], extracted=extracted)
        # Act
        result = await node.run(fake_ctx)
        # Assert
        assert isinstance(result, ProcessNextInvoice)
        assert len(fake_ctx.state.processed) == 1
        assert fake_ctx.state.processed[0].extraction_failed is True
        assert fake_ctx.state.current_index == 1


class TestNormalizeNode:
    async def test_writes_parsed_fields_to_next_node(self, fake_ctx):
        # Arrange
        extracted = ExtractedInvoice(
            vendor="ACME",
            total_raw="£155.45",
            invoice_date_raw="26.06.2004",
            currency_raw="GBP",
            extraction_confidence=0.9,
        )
        validation = ValidationResult(is_valid=True, missing_fields=[], warnings=[])
        node = NormalizeNode(
            path=fake_ctx.state.image_paths[0],
            extracted=extracted,
            validation=validation,
        )
        # Act
        result = await node.run(fake_ctx)
        # Assert
        assert isinstance(result, CategorizeNode)
        assert result.normalized.total == Decimal("155.45")
        assert result.normalized.invoice_date == "2004-06-26"
        assert result.normalized.currency == "GBP"
