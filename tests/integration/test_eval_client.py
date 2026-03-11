"""Integration tests for invoice_agent.services.eval_client.

All tests mock the Langfuse clients to avoid real API calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from invoice_agent.services.eval_client import fetch_trace_data, push_scores


class TestFetchTraceData:
    @patch("invoice_agent.services.eval_client._get_api_client")
    def test_returns_structured_dict(self, mock_get_api: MagicMock):
        # Arrange
        mock_api = MagicMock()
        mock_get_api.return_value = mock_api

        mock_api.trace.get.return_value = SimpleNamespace(
            input={"invoice_count": 3, "files": ["a.jpg"]},
            output={"total_spend": "300.00"},
            session_id="run_20240101",
            scores=None,
        )
        mock_api.observations.get_many.return_value = SimpleNamespace(data=[])

        # Act
        result = fetch_trace_data("trace-123")

        # Assert
        assert "trace" in result
        assert "observations" in result
        assert "scores" in result
        assert result["trace"]["session_id"] == "run_20240101"
        mock_api.trace.get.assert_called_once_with("trace-123")

    @patch("invoice_agent.services.eval_client._get_api_client")
    def test_parses_observations_and_scores(self, mock_get_api: MagicMock):
        # Arrange
        mock_api = MagicMock()
        mock_get_api.return_value = mock_api

        from datetime import datetime, timezone

        obs_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_obs = SimpleNamespace(
            id="obs-1",
            name="ExtractNode",
            output='{"vendor": "ACME"}',
            start_time=obs_time,
        )
        mock_api.trace.get.return_value = SimpleNamespace(
            input={},
            output={},
            session_id="run_1",
            scores=[
                SimpleNamespace(
                    name="categorizer_evaluator_v2",
                    value=0.8,
                    observation_id="obs-1",
                ),
            ],
        )
        mock_api.observations.get_many.return_value = SimpleNamespace(data=[mock_obs])

        # Act
        result = fetch_trace_data("trace-456")

        # Assert
        assert len(result["observations"]) == 1
        assert result["observations"][0]["name"] == "ExtractNode"
        assert len(result["scores"]) == 1
        assert result["scores"][0]["name"] == "categorizer_evaluator_v2"
        assert result["scores"][0]["value"] == 0.8


class TestPushScores:
    @patch("invoice_agent.services.eval_client.Langfuse")
    def test_calls_create_score_per_dimension(self, mock_langfuse_cls: MagicMock):
        # Arrange
        mock_lf = MagicMock()
        mock_langfuse_cls.return_value = mock_lf

        scores = {
            "schema": (0.92, "All fields present"),
            "reconciliation": (1.0, "3/3 checks passed"),
        }

        # Act
        push_scores("trace-789", scores)

        # Assert
        assert mock_lf.create_score.call_count == 2
        calls = mock_lf.create_score.call_args_list
        assert calls[0].kwargs["name"] == "schema"
        assert calls[0].kwargs["value"] == 0.92
        assert calls[1].kwargs["name"] == "reconciliation"

    @patch("invoice_agent.services.eval_client.Langfuse")
    def test_calls_flush_once(self, mock_langfuse_cls: MagicMock):
        # Arrange
        mock_lf = MagicMock()
        mock_langfuse_cls.return_value = mock_lf

        scores = {"schema": (0.9, "OK"), "recon": (1.0, "OK")}

        # Act
        push_scores("trace-abc", scores)

        # Assert
        mock_lf.flush.assert_called_once()

    @patch("invoice_agent.services.eval_client.Langfuse")
    def test_passes_observation_id_when_provided(self, mock_langfuse_cls: MagicMock):
        # Arrange
        mock_lf = MagicMock()
        mock_langfuse_cls.return_value = mock_lf

        scores = {"extraction_completeness": (0.85, "missing invoice_number")}

        # Act
        push_scores("trace-123", scores, observation_id="obs-123")

        # Assert
        call_kwargs = mock_lf.create_score.call_args.kwargs
        assert call_kwargs["observation_id"] == "obs-123"
        assert call_kwargs["trace_id"] == "trace-123"

    @patch("invoice_agent.services.eval_client.Langfuse")
    def test_omits_observation_id_when_none(self, mock_langfuse_cls: MagicMock):
        # Arrange
        mock_lf = MagicMock()
        mock_langfuse_cls.return_value = mock_lf

        scores = {"schema": (0.9, "OK")}

        # Act
        push_scores("trace-456", scores)

        # Assert
        call_kwargs = mock_lf.create_score.call_args.kwargs
        assert "observation_id" not in call_kwargs
