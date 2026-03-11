"""Unit tests for the tracing module."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from invoice_agent.models.schemas import (
    ExtractedInvoice,
    NormalizedFields,
    ValidationResult,
)
from invoice_agent.tracing import _build_node_input, init_tracing, traced_node


class TestInitTracing:
    """Tests for init_tracing() function."""

    def test_noop_when_langfuse_not_configured(self):
        """init_tracing does nothing when Langfuse keys are absent."""
        import invoice_agent.tracing as mod

        mod._initialized = False  # reset

        with patch("invoice_agent.tracing.get_settings") as mock_settings:
            mock_settings.return_value.langfuse_enabled = False
            init_tracing()
            # No Langfuse imports or calls should happen

        mod._initialized = False  # reset for other tests

    def test_idempotent_on_second_call(self):
        """Calling init_tracing twice only initializes once."""
        import invoice_agent.tracing as mod

        mod._initialized = True  # simulate already initialized

        with patch("invoice_agent.tracing.get_settings") as mock_settings:
            init_tracing()
            mock_settings.assert_not_called()

        mod._initialized = False  # reset for other tests


class TestTracedNode:
    """Tests for the traced_node decorator."""

    @pytest.mark.asyncio
    async def test_passthrough_when_langfuse_disabled(self):
        """Decorator calls original function directly when tracing off."""
        mock_run = AsyncMock(return_value="next_node")

        @traced_node
        async def run(self, ctx):  # type: ignore[override]
            return await mock_run(self, ctx)

        with patch("invoice_agent.tracing.get_settings") as mock_settings:
            mock_settings.return_value.langfuse_enabled = False
            result = await run(MagicMock(), MagicMock())

        assert result == "next_node"
        mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_creates_span_when_langfuse_enabled(self):
        """Decorator creates a Langfuse span when tracing is on."""
        mock_run = AsyncMock(return_value="next_node")

        class FakeNode:
            @traced_node
            async def run(self, ctx):  # type: ignore[override]
                return await mock_run(self, ctx)

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        with (
            patch("invoice_agent.tracing.get_settings") as mock_settings,
            patch("langfuse.get_client") as mock_get_client,
        ):
            mock_settings.return_value.langfuse_enabled = True
            mock_get_client.return_value.start_as_current_observation.return_value = (
                mock_span
            )

            node = FakeNode()
            result = await node.run(MagicMock())

        assert result == "next_node"
        mock_get_client.return_value.start_as_current_observation.assert_called_once()


# ── Helper stubs for _build_node_input tests ─────────────────────────────────


@dataclass
class _SimpleNode:
    """Node with only simple types and a Path."""

    path: Path = field(default_factory=Path)
    name: str = ""
    count: int = 0


@dataclass
class _NodeWithBaseModel:
    """Node carrying a Pydantic BaseModel field."""

    path: Path = field(default_factory=Path)
    extracted: ExtractedInvoice = field(default_factory=ExtractedInvoice)


@dataclass
class _NodeWithDataclass:
    """Node carrying a stdlib dataclass field."""

    path: Path = field(default_factory=Path)
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult(True, [], [])
    )


@dataclass
class _MixedNode:
    """Node with Path + BaseModel + dataclass + str fields."""

    path: Path = field(default_factory=Path)
    extracted: ExtractedInvoice = field(default_factory=ExtractedInvoice)
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult(True, [], [])
    )
    label: str = ""


# ── _build_node_input tests ──────────────────────────────────────────────────


class TestBuildNodeInput:
    """Tests for _build_node_input."""

    def test_returns_none_for_non_dataclass(self) -> None:
        """Non-dataclass objects return None."""
        assert _build_node_input("not a node") is None

    def test_simple_fields_only(self) -> None:
        """Path and simple types are serialized as before."""
        node = _SimpleNode(
            path=Path("/tmp/fake.jpg"),
            name="test",
            count=42,
        )
        with patch("os.path.getsize", return_value=1024):
            with patch.object(Path, "exists", return_value=True):
                result = _build_node_input(node)

        assert result is not None
        assert result["file"] == "fake.jpg"
        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["mime_type"] == "image/jpeg"

    def test_basemodel_serialization(self) -> None:
        """Pydantic BaseModel fields are serialized via model_dump."""
        extracted = ExtractedInvoice(
            vendor="Acme Corp",
            total_raw="150.00",
            extraction_confidence=0.9,
        )
        node = _NodeWithBaseModel(
            path=Path("/tmp/inv.png"),
            extracted=extracted,
        )
        with patch("os.path.getsize", return_value=2048):
            with patch.object(Path, "exists", return_value=True):
                result = _build_node_input(node)

        assert result is not None
        assert "extracted" in result
        assert result["extracted"]["vendor"] == "Acme Corp"
        assert result["extracted"]["total_raw"] == "150.00"
        assert result["extracted"]["extraction_confidence"] == 0.9

    def test_dataclass_serialization(self) -> None:
        """Stdlib dataclass fields are serialized via asdict."""
        validation = ValidationResult(
            is_valid=False,
            missing_fields=["vendor"],
            warnings=["low confidence"],
        )
        node = _NodeWithDataclass(
            path=Path("/tmp/inv.jpg"),
            validation=validation,
        )
        with patch("os.path.getsize", return_value=512):
            with patch.object(Path, "exists", return_value=True):
                result = _build_node_input(node)

        assert result is not None
        assert "validation" in result
        assert result["validation"]["is_valid"] is False
        assert result["validation"]["missing_fields"] == ["vendor"]
        assert result["validation"]["warnings"] == ["low confidence"]

    def test_mixed_fields(self) -> None:
        """Node with Path + BaseModel + dataclass + str all serialized."""
        extracted = ExtractedInvoice(vendor="Test Vendor")
        validation = ValidationResult(True, [], [])
        node = _MixedNode(
            path=Path("/tmp/test.pdf"),
            extracted=extracted,
            validation=validation,
            label="batch-1",
        )
        with patch("os.path.getsize", return_value=4096):
            with patch.object(Path, "exists", return_value=True):
                result = _build_node_input(node)

        assert result is not None
        assert result["file"] == "test.pdf"
        assert result["mime_type"] == "application/pdf"
        assert result["extracted"]["vendor"] == "Test Vendor"
        assert result["validation"]["is_valid"] is True
        assert result["label"] == "batch-1"

    def test_normalized_fields_dataclass(self) -> None:
        """NormalizedFields (with Decimal) serializes correctly."""

        @dataclass
        class _NodeWithNormalized:
            normalized: NormalizedFields

        node = _NodeWithNormalized(
            normalized=NormalizedFields(
                total=Decimal("250.50"),
                invoice_date="2025-03-15",
                currency="USD",
            )
        )
        result = _build_node_input(node)

        assert result is not None
        assert result["normalized"]["total"] == Decimal("250.50")
        assert result["normalized"]["invoice_date"] == "2025-03-15"
        assert result["normalized"]["currency"] == "USD"
