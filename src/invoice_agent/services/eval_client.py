"""Langfuse fetch/push wrappers for the evaluation pipeline."""

from __future__ import annotations

from typing import Any

from langfuse import Langfuse
from langfuse.api.client import LangfuseAPI

from invoice_agent.settings import get_settings


def _get_api_client() -> LangfuseAPI:
    """Create a LangfuseAPI REST client from settings."""
    settings = get_settings()
    return LangfuseAPI(
        base_url=settings.langfuse_host,
        username=settings.langfuse_public_key or "",
        password=settings.langfuse_secret_key or "",
    )


def fetch_trace_data(trace_id: str) -> dict[str, Any]:
    """Fetch trace, observations, and scores for a given trace ID.

    Returns::

        {
            "trace": {"input": ..., "output": ..., "session_id": ...},
            "observations": [...],
            "scores": [...],
        }
    """
    api = _get_api_client()

    # Fetch the trace
    trace = api.trace.get(trace_id)

    trace_data: dict[str, Any] = {
        "input": trace.input,
        "output": trace.output,
        "session_id": trace.session_id,
    }

    # Fetch observations for this trace
    obs_response = api.observations.get_many(
        trace_id=trace_id,
        fields="core,basic,io",
    )
    observations: list[dict[str, Any]] = []
    for obs in obs_response.data:
        observations.append(
            {
                "id": obs.id,
                "name": obs.name,
                "output": obs.output,
                "startTime": (obs.start_time.isoformat() if obs.start_time else ""),
            }
        )

    # Extract scores from the trace (list of ScoreV1 objects)
    scores: list[dict[str, Any]] = []
    if trace.scores:
        for s in trace.scores:
            scores.append(
                {
                    "name": s.name,
                    "value": s.value,
                    "observationId": s.observation_id,
                }
            )

    return {
        "trace": trace_data,
        "observations": observations,
        "scores": scores,
    }


def push_scores(
    trace_id: str,
    scores: dict[str, tuple[float, str]],
    observation_id: str | None = None,
) -> None:
    """Push evaluation scores to Langfuse.

    If observation_id is provided, scores are linked to that span.
    Otherwise they are linked to the trace only.
    """
    settings = get_settings()
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key or "",
        secret_key=settings.langfuse_secret_key or "",
        host=settings.langfuse_host,
    )
    for name, (value, comment) in scores.items():
        kwargs: dict[str, Any] = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "comment": comment,
            "data_type": "NUMERIC",
        }
        if observation_id is not None:
            kwargs["observation_id"] = observation_id
        langfuse.create_score(**kwargs)
    langfuse.flush()


def fetch_recent_traces(n: int = 5) -> list[dict[str, Any]]:
    """Fetch the N most recent invoice-agent-run traces.

    Returns a list of dicts with keys:
    trace_id, session_id, timestamp, invoice_count.
    """
    api = _get_api_client()
    response = api.trace.list(
        name="invoice-agent-run",
        limit=n,
        order_by="timestamp.desc",
    )

    results: list[dict[str, Any]] = []
    for trace in response.data:
        invoice_count = 0
        if isinstance(trace.input, dict):
            invoice_count = trace.input.get("invoice_count", 0)

        results.append(
            {
                "trace_id": trace.id,
                "session_id": trace.session_id,
                "timestamp": (trace.timestamp.isoformat() if trace.timestamp else ""),
                "invoice_count": invoice_count,
            }
        )

    return results
