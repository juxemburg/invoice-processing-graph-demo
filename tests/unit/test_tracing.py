"""Unit tests for the tracing module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from invoice_agent.tracing import init_tracing, traced_node


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
