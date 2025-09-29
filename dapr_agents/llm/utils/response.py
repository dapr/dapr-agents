import logging
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel

from dapr_agents.llm.utils.stream import StreamHandler
from dapr_agents.llm.utils.structure import StructureHandler
from dapr_agents.types.message import (
    LLMChatCandidateChunk,
    LLMChatResponse,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ResponseHandler:
    """
    Handles both streaming and non-streaming chat completions from various LLM providers.
    """

    @staticmethod
    def process_response(
        response: Any,
        llm_provider: str,
        response_format: Optional[Type[T]] = None,
        structured_mode: Literal["json", "function_call"] = "json",
        stream: bool = False,
        on_chunk: Optional[Callable[[LLMChatCandidateChunk], None]] = None,
    ) -> Union[
        Iterator[LLMChatCandidateChunk],  # when streaming
        LLMChatResponse,  # non‑stream + no format
        T,  # non‑stream + single structured format
        list[T],  # non‑stream + list structured format
    ]:
        """
        Process a chat completion.

        - **Streaming** (`stream=True`):
          Yields `LLMChatCandidateChunk` via `StreamHandler`, honoring `on_chunk` / `on_final`.
        - **Non-streaming** (`stream=False`):
          1. Normalize provider envelope → `LLMChatResponse`.
          2. If no `response_format` requested, return that `LLMChatResponse`.
          3. Otherwise, extract the first assistant message, parse & validate it
             against your Pydantic `response_format`, and return the model (or list).

        Args:
            response:         Raw API return (stream iterator or full response object).
            llm_provider:     e.g. `"openai"`.
            response_format:  Optional Pydantic model (or `List[Model]`) for structured output.
            structured_mode:  `"json"` or `"function_call"` (only non-stream).
            stream:           Whether this is a streaming call.
            on_chunk:         Callback on every partial `LLMChatCandidateChunk`.

        Returns:
            • **streaming**: `Iterator[LLMChatCandidateChunk]`
            • **non-stream + no format**: full `LLMChatResponse`
            • **non-stream + format**: validated Pydantic model instance or `List[...]`
        """
        provider = llm_provider.lower()

        # ─── Streaming ─────────────────────────────────────────────────────────
        if stream:
            return StreamHandler.process_stream(
                stream=response,
                llm_provider=provider,
                on_chunk=on_chunk,
            )
        else:
            # ─── Non‑streaming ─────────────────────────────────────────────────────
            # 1) Normalize full response → LLMChatResponse
            if provider in ("openai", "nvidia"):
                from dapr_agents.llm.openai.utils import process_openai_chat_response

                llm_resp: LLMChatResponse = process_openai_chat_response(response)
            elif provider == "huggingface":
                from dapr_agents.llm.huggingface.utils import process_hf_chat_response

                llm_resp = process_hf_chat_response(response)
            elif provider == "dapr":
                from dapr_agents.llm.dapr.utils import process_dapr_chat_response

                llm_resp = process_dapr_chat_response(response)
            else:
                # if you add more providers, handle them here
                llm_resp = response  # type: ignore

            # 2) If no structured format requested, return the full response
            if response_format is None:
                return llm_resp

            # 3) They did request a Pydantic model → extract first assistant message
            first_candidate = next(iter(llm_resp.results), None)
            if not first_candidate:
                raise ValueError("No candidates in LLMChatResponse")
            assistant = first_candidate.message

            # 3a) Get the raw JSON or function‐call payload
            raw = StructureHandler.extract_structured_response(
                message=assistant,
                llm_provider=llm_provider,
                structured_mode=structured_mode,
            )

            # 3b) Wrap List[Model] → IterableModel if needed
            fmt = StructureHandler.normalize_iterable_format(response_format)
            # 3c) Ensure exactly one Pydantic model inside
            model_cls = StructureHandler.resolve_response_model(fmt)
            if model_cls is None:
                raise TypeError(
                    f"Cannot resolve a Pydantic model from {response_format!r}"
                )

            # 3d) Validate JSON/dict → Pydantic
            validated = StructureHandler.validate_response(raw, fmt)
            logger.info("Structured output successfully validated.")

            # 3e) Return the validated model (don't unwrap iterable models)
            return validated
