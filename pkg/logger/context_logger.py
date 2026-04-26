"""Context-aware structured logger for Dapr Agents.

Provides automatic correlation ID injection, log level management,
and structured output compatible with observability platforms.
"""

import logging
import sys
import threading
from contextlib import contextmanager
from typing import Any, Optional


class ContextLogger:
    """Thread-safe contextual logger with auto-injected metadata."""

    _ctx = threading.local()

    def __init__(self, name: str = "dapr.agents", level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            self._logger.addHandler(handler)

    @classmethod
    def set_context(cls, **kwargs):
        """Set thread-local context key-value pairs."""
        if not hasattr(cls._ctx, "data"):
            cls._ctx.data = {}
        cls._ctx.data.update(kwargs)

    @classmethod
    def clear_context(cls):
        """Clear thread-local context."""
        cls._ctx.data = {}

    @classmethod
    @contextmanager
    def bind(cls, **kwargs):
        """Context manager for temporary context binding."""
        old = getattr(cls._ctx, "data", {}).copy()
        cls.set_context(**kwargs)
        try:
            yield
        finally:
            cls._ctx.data = old

    def _format(self, msg: str, extra: Optional[dict] = None) -> str:
        ctx = getattr(self._ctx, "data", {})
        parts = [msg]
        if ctx:
            pairs = [f"{k}={v}" for k, v in ctx.items()]
            parts.append(" ".join(pairs))
        if extra:
            pairs = [f"{k}={v}" for k, v in extra.items()]
            parts.append(" ".join(pairs))
        return " | ".join(parts)

    def debug(self, msg: str, **extra):
        self._logger.debug(self._format(msg, extra))

    def info(self, msg: str, **extra):
        self._logger.info(self._format(msg, extra))

    def warning(self, msg: str, **extra):
        self._logger.warning(self._format(msg, extra))

    def error(self, msg: str, **extra):
        self._logger.error(self._format(msg, extra))


log = ContextLogger()
