"""Langfuse tracing initialization and graph-node span decorator."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from invoice_agent.settings import get_settings

logger = logging.getLogger(__name__)

_initialized = False

F = TypeVar("F", bound=Callable[..., Any])


def init_tracing() -> None:
    """Initialize Langfuse tracing and pydantic-ai instrumentation.

    Safe to call multiple times — subsequent calls are no-ops.
    When Langfuse keys are not configured, this function does nothing.
    Must be called BEFORE any pydantic-ai Agent is instantiated.
    """
    global _initialized  # noqa: PLW0603
    if _initialized:
        return
    _initialized = True

    settings = get_settings()
    if not settings.langfuse_enabled:
        logger.debug("Langfuse keys not configured — tracing disabled")
        return

    # Import langfuse AFTER env vars are loaded (settings ensures this)
    import os

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key or "")
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

    from langfuse import get_client
    from pydantic_ai import Agent
    from pydantic_ai.models.instrumented import InstrumentationSettings

    # Initialize Langfuse client (reads env vars automatically)
    get_client()

    # Enable pydantic-ai auto-instrumentation for all agents
    # Exclude binary content (invoice images) for performance
    Agent.instrument_all(InstrumentationSettings(include_binary_content=False))

    logger.info("Langfuse tracing initialized (host=%s)", settings.langfuse_host)


def traced_node(func: F) -> F:
    """Decorator for graph node run() methods.

    Creates a Langfuse span for each node execution. When Langfuse
    is not configured, the decorator is a transparent pass-through.
    """

    @functools.wraps(func)
    async def wrapper(self: Any, ctx: Any) -> Any:
        settings = get_settings()
        if not settings.langfuse_enabled:
            return await func(self, ctx)

        from langfuse import get_client

        langfuse = get_client()
        node_name = type(self).__name__

        # Build metadata from node fields (e.g. path for ExtractNode)
        metadata: dict[str, str] = {}
        if hasattr(self, "path") and self.path:
            metadata["invoice_file"] = str(self.path.name)

        with langfuse.start_as_current_observation(
            as_type="span",
            name=node_name,
            metadata=metadata,
        ):
            return await func(self, ctx)

    return wrapper  # type: ignore[return-value]
