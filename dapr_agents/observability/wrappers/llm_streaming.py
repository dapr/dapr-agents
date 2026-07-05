#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Streaming instrumentation helpers for the LLM observability wrapper.

Extracted from ``llm.py`` to keep the streaming span lifecycle (start span →
pass chunks through → accumulate → set attributes at stream end → close)
separate from the non-streaming request path. ``LLMWrapper`` mixes
:class:`LLMStreamingMixin` in; the mixin relies on a handful of general span
helpers the host wrapper provides (see the class docstring for the contract).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from ..constants import (
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    Status,
    StatusCode,
    safe_json_dumps,
)
from dapr_agents.types.message import LLMChatResponse
from dapr_agents.types.streaming import AssistantMessageAccumulator

logger = logging.getLogger(__name__)


class _AccumulatorBearingIterator:
    """Wraps a chunk iterator and carries a shared ``AssistantMessageAccumulator``.

    ``StreamEmitter.consume`` detects the attached ``_dapr_accumulator``
    and reuses it instead of running a parallel accumulation pass, halving
    per-chunk CPU work when both the observability wrapper and the
    streaming emitter are active on the same iterator.

    Forwards the full generator protocol (``send``, ``throw``, ``close``)
    so callers that drive the underlying generator with values or inject
    exceptions continue to work unchanged.
    """

    __slots__ = ("_inner", "_dapr_accumulator")

    def __init__(self, inner: Any, accumulator: Any) -> None:
        self._inner = inner
        self._dapr_accumulator = accumulator

    def __iter__(self):  # noqa: D401 - trivial
        return self

    def __next__(self):
        return next(self._inner)

    def send(self, value):
        return self._inner.send(value)

    def throw(self, *args, **kwargs):
        return self._inner.throw(*args, **kwargs)

    def close(self):
        return self._inner.close()


class _AsyncAccumulatorBearingIterator:
    """Async-iterator counterpart to :class:`_AccumulatorBearingIterator`.

    Forwards the async-generator protocol (``asend``, ``athrow``,
    ``aclose``) so callers that drive the underlying async generator
    with values or inject exceptions continue to work unchanged.
    """

    __slots__ = ("_inner", "_dapr_accumulator")

    def __init__(self, inner: Any, accumulator: Any) -> None:
        self._inner = inner
        self._dapr_accumulator = accumulator

    def __aiter__(self):  # noqa: D401 - trivial
        return self

    async def __anext__(self):
        return await self._inner.__anext__()

    async def asend(self, value):
        return await self._inner.asend(value)

    async def athrow(self, *args, **kwargs):
        return await self._inner.athrow(*args, **kwargs)

    async def aclose(self):
        return await self._inner.aclose()


class LLMStreamingMixin:
    """Streaming-call instrumentation for :class:`LLMWrapper`.

    Owns the streaming branches (``stream=True``) of the sync and async
    execution paths plus the chunk-accumulation helpers. Mixed into
    ``LLMWrapper`` so the streaming span lifecycle lives apart from the
    non-streaming request path.

    Contract — the host wrapper must provide (all defined on ``LLMWrapper``):

    - ``self._tracer`` — the OpenTelemetry tracer.
    - ``self._record_span_error(span, exc)`` — attach an error to a span.
    - ``self._set_output_attributes(span, result)`` — non-streaming attribute
      path, reused when a provider falls back to a full response.
    - ``self._set_serialized_output(span, payload)`` — serialize the raw
      output payload onto the span.
    """

    async def _wrap_async_streaming(
        self,
        *,
        wrapped: Any,
        args: Any,
        kwargs: Any,
        span_name: str,
        attributes: Dict[str, Any],
    ) -> Any:
        """Instrument an ``async``-returning streaming LLM call."""

        span = self._tracer.start_span(span_name, attributes=attributes)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            self._record_span_error(span, exc)
            span.end()
            raise

        if self._looks_like_non_streaming_result(result, async_mode=True):
            try:
                self._set_output_attributes(span, result)
                span.set_status(Status(StatusCode.OK))
                return result
            finally:
                span.end()

        accumulator = AssistantMessageAccumulator()

        async def _gen():
            try:
                async for chunk in result:
                    self._safe_ingest(accumulator, chunk)
                    yield chunk
            except GeneratorExit:
                # Iterator was garbage-collected or the caller cancelled
                # mid-stream (e.g., an HTTP client disconnected). Record as
                # a cancellation and end the span so we don't leak it.
                self._record_span_error(
                    span, asyncio.CancelledError("stream cancelled")
                )
                span.end()
                raise
            except Exception as exc:
                self._record_span_error(span, exc)
                span.end()
                raise
            else:
                self._finalize_stream_span(span, accumulator)

        return _AsyncAccumulatorBearingIterator(_gen(), accumulator)

    def _handle_streaming_execution(
        self,
        wrapped: Any,
        args: Any,
        kwargs: Any,
        span_name: str,
        attributes: Dict[str, Any],
        instance: Any = None,
        messages: Any = None,
    ) -> Any:
        """Instrument a streaming LLM call so span attributes are set at stream end.

        The wrapped call returns an iterator. We start the span, pass every
        chunk through to the caller, accumulate into a final ``AssistantMessage``,
        then set the usual output attributes on the span and close it inside a
        ``finally`` block. If the underlying call raises before returning an
        iterator, the span records the error and propagates.

        Falls back to a ``LLMChatResponse``-style result when the provider
        downgraded to non-streaming (e.g. Dapr Conversation API fallback); in
        that case the usual ``_set_output_attributes`` handles the span.

        The accumulator is attached to the returned iterator as
        ``_dapr_accumulator`` so ``StreamEmitter.consume`` can reuse the
        same instance and avoid a second ingest pass per chunk.
        """

        span = self._tracer.start_span(span_name, attributes=attributes)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            self._record_span_error(span, exc)
            span.end()
            raise

        if self._looks_like_non_streaming_result(result, async_mode=False):
            # Provider fell back to non-streaming (e.g. Dapr Conversation API
            # before native streaming shipped). Use the existing attribute path.
            try:
                self._set_output_attributes(span, result)
                span.set_status(Status(StatusCode.OK))
                return result
            finally:
                span.end()

        accumulator = AssistantMessageAccumulator()

        def _wrap_iterator(inner):
            try:
                for chunk in inner:
                    self._safe_ingest(accumulator, chunk)
                    yield chunk
            except GeneratorExit:
                # Consumer cancelled or garbage-collected the iterator
                # before exhaustion — end the span so we don't leak it.
                self._record_span_error(
                    span, asyncio.CancelledError("stream cancelled")
                )
                span.end()
                raise
            except Exception as exc:
                self._record_span_error(span, exc)
                span.end()
                raise
            else:
                self._finalize_stream_span(span, accumulator)

        # Wrap the generator in a small iterator class that supports
        # attribute assignment so ``StreamEmitter.consume`` can detect the
        # pre-populated accumulator and skip its own ingest pass.
        return _AccumulatorBearingIterator(_wrap_iterator(result), accumulator)

    # -- shared helpers for streaming branches ----------------------------

    @staticmethod
    def _looks_like_non_streaming_result(result: Any, *, async_mode: bool) -> bool:
        """Return True when ``result`` is a provider fallback response rather
        than a chunk iterator.

        The Dapr Conversation API (and structured-output paths) returns a
        full ``LLMChatResponse`` even when ``stream=True`` was requested.
        Detect that shape by (a) explicit class match and (b) absence of the
        appropriate iterator protocol for the current execution mode.
        """
        if isinstance(result, LLMChatResponse):
            return True
        iter_attr = "__aiter__" if async_mode else "__iter__"
        return not hasattr(result, iter_attr)

    @staticmethod
    def _safe_ingest(accumulator: Any, chunk: Any) -> None:
        """Feed a chunk into an accumulator; log and continue on failures."""
        try:
            accumulator.ingest(chunk)
        except Exception as acc_exc:  # noqa: BLE001 - never break the stream
            logger.debug("LLMWrapper accumulator ingest failed: %s", acc_exc)

    def _finalize_stream_span(self, span: Any, accumulator: Any) -> None:
        """Set end-of-stream attributes and close the span."""
        try:
            final_message = accumulator.assistant_message()
            self._set_streaming_output_attributes(
                span,
                assistant_message=final_message,
                metadata=accumulator.last_metadata,
                finish_reason=accumulator.finish_reason,
            )
            span.set_status(Status(StatusCode.OK))
        finally:
            span.end()

    def _set_streaming_output_attributes(
        self,
        span: Any,
        *,
        assistant_message: Dict[str, Any],
        metadata: Dict[str, Any],
        finish_reason: Any,
    ) -> None:
        """Translate an accumulated streaming result into the same attributes
        the non-streaming path emits."""

        try:
            self._set_serialized_output(
                span,
                {"results": [{"message": assistant_message}], "metadata": metadata},
            )
            out_msg = {
                "role": assistant_message.get("role", "assistant"),
                "content": assistant_message.get("content") or "",
            }
            if "tool_calls" in assistant_message:
                out_msg["tool_calls"] = assistant_message["tool_calls"]
            span.set_attribute(GEN_AI_OUTPUT_MESSAGES, safe_json_dumps([out_msg]))
            if finish_reason:
                span.set_attribute(
                    GEN_AI_RESPONSE_FINISH_REASONS,
                    safe_json_dumps([str(finish_reason)]),
                )
            usage = metadata.get("usage") if metadata else None
            if isinstance(usage, dict):
                if "prompt_tokens" in usage:
                    span.set_attribute(
                        GEN_AI_USAGE_INPUT_TOKENS, int(usage["prompt_tokens"])
                    )
                if "completion_tokens" in usage:
                    span.set_attribute(
                        GEN_AI_USAGE_OUTPUT_TOKENS, int(usage["completion_tokens"])
                    )
        except Exception as exc:  # noqa: BLE001 - never break the stream
            logger.debug("LLMWrapper streaming attribute set failed: %s", exc)
