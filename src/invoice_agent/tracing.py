"""Langfuse tracing initialization and graph-node span decorator."""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from invoice_agent.settings import get_settings

logger = logging.getLogger(__name__)

_initialized = False

F = TypeVar("F", bound=Callable[..., Any])

# OTel context captured inside the root span so child spans nest correctly.
_root_otel_context: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "root_otel_context", default=None
)

# Current Langfuse span reference, used by log_span_output.
_current_span: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_span", default=None
)


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

    os.environ.setdefault("OTEL_SERVICE_NAME", "invoice-agent")
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


def capture_root_context() -> None:
    """Capture current OTel context as root for child spans.

    Call this inside the root Langfuse span (in cli.py) so that
    traced_node spans nest under the root trace.
    """
    from opentelemetry import context as otel_context

    _root_otel_context.set(otel_context.get_current())


def log_span_output(output: dict[str, Any]) -> None:
    """Log output data on the current traced span.

    No-op if tracing is disabled or no span is active.
    """
    span = _current_span.get()
    if span is not None:
        span.update(output=output)


_MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pdf": "application/pdf",
}


def _build_node_input(node: Any) -> dict[str, Any] | None:
    """Build span input from node dataclass fields.

    Serializes Pydantic BaseModel fields via ``.model_dump()`` and
    stdlib dataclass fields via ``dataclasses.asdict()``.  Simple
    types (str, int, float, bool) and Path names are included
    directly.  For Path fields pointing to existing files, adds
    mime_type and size_kb for diagnostics.
    """
    if not hasattr(node, "__dataclass_fields__"):
        return None
    inp: dict[str, Any] = {}
    for name in node.__dataclass_fields__:
        val = getattr(node, name, None)
        if isinstance(val, Path):
            inp["file"] = str(val.name)
            if val.exists():
                suffix = val.suffix.lower()
                inp["mime_type"] = _MIME_MAP.get(suffix, "application/octet-stream")
                inp["size_kb"] = round(os.path.getsize(val) / 1024, 1)
        elif isinstance(val, BaseModel):
            inp[name] = val.model_dump()
        elif dataclasses.is_dataclass(val) and not isinstance(val, type):
            inp[name] = dataclasses.asdict(val)
        elif isinstance(val, (str, int, float, bool)):
            inp[name] = val
    return inp or None


def traced_node(func: F) -> F:
    """Decorator for graph node run() methods.

    Creates a Langfuse span for each node execution. When Langfuse
    is not configured, the decorator is a transparent pass-through.
    Restores the root OTel context so spans nest under the root trace.
    """

    @functools.wraps(func)
    async def wrapper(self: Any, ctx: Any) -> Any:
        settings = get_settings()
        if not settings.langfuse_enabled:
            return await func(self, ctx)

        from langfuse import get_client
        from opentelemetry import context as otel_context

        langfuse = get_client()
        node_name = type(self).__name__

        # Build metadata from node fields (e.g. path for ExtractNode)
        metadata: dict[str, str] = {}
        if hasattr(self, "path") and self.path:
            metadata["invoice_file"] = str(self.path.name)

        # Restore root context so this span nests under the root trace
        root_ctx = _root_otel_context.get()
        token = None
        if root_ctx is not None:
            token = otel_context.attach(root_ctx)

        try:
            with langfuse.start_as_current_observation(
                as_type="span",
                name=node_name,
                input=_build_node_input(self),
                metadata=metadata,
            ) as span:
                _current_span.set(span)
                try:
                    return await func(self, ctx)
                finally:
                    _current_span.set(None)
        finally:
            if token is not None:
                otel_context.detach(token)

    return wrapper  # type: ignore[return-value]
