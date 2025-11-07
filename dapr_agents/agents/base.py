from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, Coroutine

from dapr_agents.agents.components import AgentComponents
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    AgentExecutionConfig,
    WorkflowGrpcOptions,
    DEFAULT_AGENT_WORKFLOW_BUNDLE,
)
from dapr_agents.agents.prompting import AgentProfileConfig, PromptingAgentBase
from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.utils.defaults import get_default_llm
from dapr_agents.memory import ConversationDaprStateMemory, ConversationListMemory
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.storage.daprstores.stateservice import StateStoreError
from dapr_agents.tool.base import AgentTool
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.types import AssistantMessage, ToolExecutionRecord, UserMessage

logger = logging.getLogger(__name__)


class AgentBase(AgentComponents):
    """
    Base class for agent behavior.

    Responsibilities:
    - Profile/prompt wiring (system prompt, instructions, style, template).
    - LLM client wiring.
    - Tool exposure and execution adapter.
    - Conversation memory management (configurable; defaults provided).

    Infrastructure (pub/sub, durable state, registry) is provided by `AgentComponents`.
    """

    def __init__(
        self,
        *,
        # Profile / prompt
        profile: Optional[AgentProfileConfig] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[Iterable[str]] = None,
        style_guidelines: Optional[Iterable[str]] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[PromptTemplateBase] = None,
        # Components (infrastructure)
        pubsub: Optional[AgentPubSubConfig] = None,
        state: Optional[AgentStateConfig] = None,
        registry: Optional[AgentRegistryConfig] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
        max_etag_attempts: int = 10,
        # Memory / runtime
        memory: Optional[AgentMemoryConfig] = None,
        llm: Optional[ChatClientBase] = None,
        tools: Optional[Iterable[Any]] = None,
        # Metadata
        agent_metadata: Optional[Dict[str, Any]] = None,
        workflow_grpc: Optional[WorkflowGrpcOptions] = None,
        # Execution
        execution: Optional[AgentExecutionConfig] = None,
    ) -> None:
        """
        Initialize an agent with behavior + infrastructure.

        Args:
            profile: Base profile config (name/role/goal/prompts). Optional if
                individual fields are provided below.
            name: Agent name (required if `profile` is omitted).
            role: Agent role (e.g., "Assistant").
            goal: High-level agent objective.
            instructions: Additional instruction strings for the prompt.
            style_guidelines: Style directives for the prompt.
            system_prompt: System prompt override.
            prompt_template: Optional explicit prompt template instance.

            pubsub: Pub/Sub config used by `AgentComponents`.
            state: Durable state config used by `AgentComponents`.
            registry: Team registry config used by `AgentComponents`.
            execution: Execution dials for the agent run.
            base_metadata: Default Dapr state metadata used by `AgentComponents`.
            max_etag_attempts: Concurrency retry count for registry mutations.

            memory: Memory backend configuration. If omitted and a state store
                is configured, a Dapr-backed conversation memory is created by default.
            llm: Chat client. Defaults to `get_default_llm()`.
            tools: Optional tool callables or `AgentTool` instances.

            agent_metadata: Extra metadata to store in the registry.
            workflow_grpc: Optional gRPC overrides for the workflow runtime channel.
        """
        # Resolve and validate profile (ensures non-empty name).
        resolved_profile = self._build_profile(
            base_profile=profile,
            name=name,
            role=role,
            goal=goal,
            instructions=instructions,
            style_guidelines=style_guidelines,
            system_prompt=system_prompt,
        )
        self.profile = resolved_profile
        self.name = resolved_profile.name  # type: ignore[assignment]

        # Wire infrastructure via AgentComponents.
        super().__init__(
            name=self.name,
            pubsub=pubsub,
            state=state,
            registry=registry,
            base_metadata=base_metadata,
            max_etag_attempts=max_etag_attempts,
            default_bundle=DEFAULT_AGENT_WORKFLOW_BUNDLE,
            workflow_grpc_options=workflow_grpc,
        )

        # -----------------------------
        # Memory wiring
        # -----------------------------
        self._memory = memory or AgentMemoryConfig()
        if self._memory.store is None and state is not None:
            # Auto-provision a Dapr-backed memory if we have a state store.
            self._memory.store = ConversationDaprStateMemory(  # type: ignore[union-attr]
                store_name=state.store.store_name,
                session_id=f"{self.name}-session",
            )
        self.memory = self._memory.store or ConversationListMemory()

        # -----------------------------
        # Prompting helper
        # -----------------------------
        self.prompting_helper = PromptingAgentBase(
            name=self.name,
            role=resolved_profile.role or "Assistant",
            goal=resolved_profile.goal or "Help users accomplish their tasks.",
            instructions=list(resolved_profile.instructions),
            style_guidelines=list(resolved_profile.style_guidelines),
            system_prompt=resolved_profile.system_prompt,
            template_format=resolved_profile.template_format,
            include_chat_history=True,
            prompt_template=prompt_template,
            profile=resolved_profile,
        )
        # Keep profile config synchronized with helper defaults.
        if self.profile.name is None:
            self.profile.name = self.prompting_helper.name
        if self.profile.role is None:
            self.profile.role = self.prompting_helper.role
        if self.profile.goal is None:
            self.profile.goal = self.prompting_helper.goal

        self.prompt_template = self.prompting_helper.prompt_template
        self._text_formatter = self.prompting_helper.text_formatter

        # -----------------------------
        # LLM wiring
        # -----------------------------
        self.llm: ChatClientBase = llm or get_default_llm()
        if self.llm:
            self.llm.prompt_template = self.prompt_template

        # -----------------------------
        # Tools
        # -----------------------------
        self.tools: List[Any] = list(tools or [])
        self.tool_executor = AgentToolExecutor(tools=list(self.tools))
        self.tool_history: List[ToolExecutionRecord] = []

        # -----------------------------
        # Execution config
        # -----------------------------
        self.execution = execution or AgentExecutionConfig()
        try:
            self.execution.max_iterations = max(1, int(self.execution.max_iterations))
        except Exception:
            self.execution.max_iterations = 10
        if not self.tools:
            if self.execution.tool_choice is not None:
                logger.debug(
                    "No tools configured for agent '%s'; ignoring tool_choice=%r.",
                    self.name,
                    self.execution.tool_choice,
                )
            self.execution.tool_choice = None
        elif self.execution.tool_choice is None:
            self.execution.tool_choice = "auto"

        # -----------------------------
        # Load durable state (from AgentComponents)
        # -----------------------------
        try:
            self.load_state()
        except Exception:  # noqa: BLE001
            logger.warning("Agent failed to load persisted state; starting fresh.")

        # -----------------------------
        # Agent metadata & registry registration (from AgentComponents)
        # -----------------------------
        base_meta: Dict[str, Any] = {
            "name": self.name,
            "orchestrator": False,
            "role": self.prompting_helper.role,
            "goal": self.prompting_helper.goal,
            "instructions": list(self.prompting_helper.instructions),
        }
        if self.pubsub is not None:
            base_meta["topic_name"] = self.agent_topic_name
            base_meta["pubsub_name"] = self.message_bus_name

        merged_meta = {**base_meta, **(agent_metadata or {})}
        self.agent_metadata = merged_meta
        if self.registry_state is not None:
            try:
                self.register_agentic_system(metadata=merged_meta)
            except StateStoreError:
                logger.warning(
                    "Could not register agent metadata; registry unavailable."
                )
        else:
            logger.debug(
                "Registry configuration not provided; skipping agent registration."
            )

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------
    @property
    def text_formatter(self) -> ColorTextFormatter:
        """Formatter used for human-friendly console output."""
        return self._text_formatter

    @text_formatter.setter
    def text_formatter(self, formatter: ColorTextFormatter) -> None:
        """Override the default text formatter and keep the helper in sync."""
        self._text_formatter = formatter
        if hasattr(self, "prompting_helper"):
            self.prompting_helper._text_formatter = formatter

    def print_interaction(
        self, source_agent_name: str, target_agent_name: str, message: str
    ) -> None:
        """
        Print a formatted interaction between two agents.

        Args:
            source_agent_name: Sender name.
            target_agent_name: Recipient name.
            message: Message content.
        """
        separator = "-" * 80
        parts = [
            (source_agent_name, "dapr_agents_mustard"),
            (" -> ", "dapr_agents_teal"),
            (f"{target_agent_name}\n\n", "dapr_agents_mustard"),
            (message + "\n\n", None),
            (separator + "\n", "dapr_agents_teal"),
        ]
        self._text_formatter.print_colored_text(parts)

    # ------------------------------------------------------------------
    # Prompting & memory utilities
    # ------------------------------------------------------------------
    def build_initial_messages(
        self,
        user_input: Optional[Union[str, Dict[str, Any]]] = None,
        **extra_variables: Any,
    ) -> List[Dict[str, Any]]:
        """
        Build the initial message list for an LLM call.

        Args:
            user_input: Optional user message or structured payload.
            **extra_variables: Extra template variables for the prompt template.

        Returns:
            List of message dictionaries ready for an LLM chat API.
        """
        return self.prompting_helper.build_initial_messages(
            user_input,
            chat_history=self.get_chat_history()
            if self.prompting_helper.include_chat_history
            else None,
            **extra_variables,
        )

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history from the configured memory backend.

        Returns:
            A list of message-like dictionaries in normalized form.
        """
        try:
            history = self.memory.get_messages()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory get_messages failed: %s", exc)
            return []

        normalized: List[Dict[str, Any]] = []
        for entry in history:
            if hasattr(entry, "model_dump"):
                normalized.append(entry.model_dump())
            elif isinstance(entry, dict):
                normalized.append(dict(entry))
        return normalized

    def reset_memory(self) -> None:
        """Clear all stored conversation messages."""
        if self.memory:
            self.memory.reset_memory()

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Return the last message stored in memory, if any."""
        history = self.get_chat_history()
        return dict(history[-1]) if history else None

    def get_last_user_message(
        self, messages: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Return the most recent message authored by the user from a sequence.

        Args:
            messages: Message sequence from which to extract the last user message.

        Returns:
            The last user message as a dict, or None if not present.
        """
        match = self._get_last_user_message(messages)
        if not match:
            return None
        result = dict(match)
        content = result.get("content")
        if isinstance(content, str):
            result["content"] = content.strip()
        return result

    def get_last_message_if_user(
        self, messages: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Return the last message only if it is authored by the user.

        Args:
            messages: Message sequence.

        Returns:
            The last message as a dict if its role is 'user'; otherwise None.
        """
        if messages and messages[-1].get("role") == "user":
            msg = dict(messages[-1])
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = content.strip()
            return msg
        return None

    def get_llm_tools(self) -> List[Union[AgentTool, Dict[str, Any]]]:
        """
        Convert configured tools into LLM-friendly tool specs.

        Returns:
            List of `AgentTool` or tool-spec dicts.
        """
        llm_tools: List[Union[AgentTool, Dict[str, Any]]] = []
        for tool in self.tools:
            if isinstance(tool, AgentTool):
                llm_tools.append(tool)
            elif callable(tool):
                try:
                    llm_tools.append(AgentTool.from_func(tool))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to convert callable to AgentTool: %s", exc)
        return llm_tools

    def _build_profile(
        self,
        *,
        base_profile: Optional[AgentProfileConfig],
        name: Optional[str],
        role: Optional[str],
        goal: Optional[str],
        instructions: Optional[Iterable[str]],
        style_guidelines: Optional[Iterable[str]],
        system_prompt: Optional[str],
    ) -> AgentProfileConfig:
        """
        Construct a concrete AgentProfileConfig from a base profile and field overrides.

        Args:
            base_profile: Optional starting profile to clone (avoids mutating the caller’s).
            name: Name override.
            role: Role override.
            goal: Goal/mission override.
            instructions: Additional instruction strings.
            style_guidelines: Prompt style directives.
            system_prompt: System prompt override.

        Returns:
            A fully-populated AgentProfileConfig with a non-empty name.

        Raises:
            ValueError: If the resulting profile has an empty name.
        """
        # Clone the base profile to avoid external side effects.
        if base_profile is not None:
            profile = AgentProfileConfig(
                name=base_profile.name,
                role=base_profile.role,
                goal=base_profile.goal,
                instructions=list(base_profile.instructions),
                style_guidelines=list(base_profile.style_guidelines),
                system_prompt=base_profile.system_prompt,
                template_format=base_profile.template_format,
                modules=tuple(base_profile.modules),
                module_overrides=dict(base_profile.module_overrides),
            )
        else:
            profile = AgentProfileConfig()

        # Apply field-level overrides when provided.
        if name is not None:
            profile.name = name
        if role is not None:
            profile.role = role
        if goal is not None:
            profile.goal = goal
        if instructions is not None:
            profile.instructions = list(instructions)
        if style_guidelines is not None:
            profile.style_guidelines = list(style_guidelines)
        if system_prompt is not None:
            profile.system_prompt = system_prompt

        # Durable agents require a concrete name for state/memory/registry keys.
        if not profile.name or not profile.name.strip():
            raise ValueError(
                "Durable agents require a non-empty name "
                "(provide name= or profile.name)."
            )

        return profile

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _run_asyncio_task(coro: Coroutine[Any, Any, Any]) -> Any:
        """
        Execute an async coroutine from a synchronous context, creating a fresh loop if needed.

        Args:
            coro: The coroutine to execute.

        Returns:
            Any: The result of the coroutine execution.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    @staticmethod
    def _serialize_message(message: Any) -> Dict[str, Any]:
        """
        Convert a message-like object into a plain dict for history persistence.

        Args:
            message: Pydantic model, dict, or object exposing `model_dump`.

        Returns:
            Normalized dictionary representation.

        Raises:
            TypeError: When the input type is unsupported.
        """
        if hasattr(message, "model_dump"):
            return message.model_dump()
        if isinstance(message, dict):
            return dict(message)
        if hasattr(message, "__dict__"):
            return dict(message.__dict__)
        raise TypeError(
            f"Unsupported message type for serialization: {type(message)!r}"
        )

    def _get_last_user_message(
        self, messages: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find the last user-role message from the given sequence."""
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        return None

    # ------------------------------------------------------------------
    # State-aware message helpers (use AgentComponents' state model)
    # ------------------------------------------------------------------
    def _construct_messages_with_instance_history(
        self, instance_id: str
    ) -> List[Dict[str, Any]]:
        """
        Build a conversation history combining persistent memory and per-instance messages.

        Args:
            instance_id: Workflow instance identifier.

        Returns:
            Combined message history excluding system messages from instance timeline.
        """
        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None

        instance_messages: List[Dict[str, Any]] = []
        if entry and hasattr(entry, "messages"):
            for msg in getattr(entry, "messages"):
                serialized = self._serialize_message(msg)
                if serialized.get("role") != "system":
                    instance_messages.append(serialized)

        persistent_memory: List[Dict[str, Any]] = []
        try:
            for msg in self.memory.get_messages():
                try:
                    persistent_memory.append(self._serialize_message(msg))
                except TypeError:
                    logger.debug(
                        "Unsupported memory message type %s; skipping.", type(msg)
                    )
        except Exception:  # noqa: BLE001
            logger.debug("Unable to load persistent memory.", exc_info=True)

        history: List[Dict[str, Any]] = []
        history.extend(persistent_memory)
        history.extend(instance_messages)
        return history

    def _sync_system_messages_with_state(
        self,
        instance_id: str,
        all_messages: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Persist the latest set of system messages into the instance state.

        Args:
            instance_id: Workflow instance id.
            all_messages: Complete message list to scan for system-role messages.
        """
        # Delegate to AgentComponents logic.
        self.sync_system_messages(instance_id=instance_id, all_messages=all_messages)

    def _process_user_message(
        self,
        instance_id: str,
        task: Optional[str],
        user_message_copy: Optional[Dict[str, Any]],
    ) -> None:
        """
        Append a user message into the instance timeline and memory, and persist state.

        Args:
            instance_id: Workflow instance id.
            task: Optional task string; if missing, no-op.
            user_message_copy: Message dict to append.
        """
        if not task or not user_message_copy:
            return

        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None
        if entry is None or not hasattr(entry, "messages"):
            return

        # Use configured coercer / message model
        message_model = (
            self._message_coercer(user_message_copy)  # type: ignore[attr-defined]
            if getattr(self, "_message_coercer", None)
            else self._message_dict_to_message_model(user_message_copy)
        )
        entry.messages.append(message_model)  # type: ignore[attr-defined]
        if hasattr(entry, "last_message"):
            entry.last_message = message_model  # type: ignore[attr-defined]

        session_id = getattr(getattr(self, "memory", None), "session_id", None)
        if session_id is not None and hasattr(entry, "session_id"):
            entry.session_id = str(session_id)  # type: ignore[attr-defined]

        self.memory.add_message(
            UserMessage(content=user_message_copy.get("content", ""))
        )
        self.save_state()

    def _save_assistant_message(
        self, instance_id: str, assistant_message: Dict[str, Any]
    ) -> None:
        """
        Append an assistant message into the instance timeline and memory, and persist state.

        Args:
            instance_id: Workflow instance id.
            assistant_message: Assistant message dict (will be tagged with agent name).
        """
        assistant_message["name"] = self.name

        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None
        if entry is None or not hasattr(entry, "messages"):
            return

        message_id = assistant_message.get("id")
        if message_id and any(
            getattr(msg, "id", None) == message_id for msg in getattr(entry, "messages")
        ):
            return

        message_model = (
            self._message_coercer(assistant_message)  # type: ignore[attr-defined]
            if getattr(self, "_message_coercer", None)
            else self._message_dict_to_message_model(assistant_message)
        )
        entry.messages.append(message_model)  # type: ignore[attr-defined]
        if hasattr(entry, "last_message"):
            entry.last_message = message_model  # type: ignore[attr-defined]

        self.memory.add_message(AssistantMessage(**assistant_message))
        self.save_state()

    # ------------------------------------------------------------------
    # Small convenience wrappers
    # ------------------------------------------------------------------
    def list_team_agents(
        self, *, team: Optional[str] = None, include_self: bool = True
    ) -> Dict[str, Any]:
        """
        Convenience wrapper over `get_agents_metadata`.

        Args:
            team: Team override.
            include_self: If True, include this agent in the results.

        Returns:
            Mapping of agent name to metadata.
        """
        return self.get_agents_metadata(
            exclude_self=not include_self,
            exclude_orchestrator=False,
            team=team,
        )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_datetime(value: Optional[Any]) -> datetime:
        """Coerce strings/None to a timezone-aware UTC datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.now(timezone.utc)
