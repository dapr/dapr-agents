from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from dapr_agents.agents.configs import AgentProfileConfig, PromptSection
from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.chat import ChatPromptTemplate
from dapr_agents.types import MessagePlaceHolder

logger = logging.getLogger(__name__)

_JINJA_PLACEHOLDER_PATTERN = re.compile(r"(?<!\{)\{\s*(\w+)\s*\}(?!\})")


def _ensure_jinja_placeholders(text: str) -> str:
    """Convert single-brace placeholders to Jinja without touching existing Jinja blocks."""
    return _JINJA_PLACEHOLDER_PATTERN.sub(r"{{\1}}", text)


@dataclass
class PromptSpec:
    """Declarative description used to build a system prompt."""

    name: str = "Agent"
    role: str = "Assistant"
    goal: str = "Help users accomplish their tasks."
    instructions: List[str] = field(default_factory=list)
    style_guidelines: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    template_format: str = "jinja2"

    def build_sections(self) -> List[PromptSection]:
        sections: List[PromptSection] = [
            PromptSection(title="Name", lines=["Your name is {{name}}."]),
            PromptSection(title="Role", lines=["You are {{role}}."]),
            PromptSection(title="Goal", lines=["Your goal is {{goal}}."]),
        ]

        if self.instructions:
            sections.append(
                PromptSection(title="Primary Instructions", lines=self.instructions)
            )

        if self.style_guidelines:
            sections.append(
                PromptSection(title="Communication Style", lines=self.style_guidelines)
            )

        return sections

    def render_system_prompt(self) -> str:
        if self.system_prompt:
            return self._apply_template_format(self.system_prompt)

        date_line = datetime.now().strftime("%B %d, %Y")
        header = f"# Today's date is: {date_line}"
        rendered_sections: List[str] = []
        for section in self.build_sections():
            rendered = section.render(self.template_format)
            if rendered:
                rendered_sections.append(rendered)
        sections = "\n\n".join(rendered_sections)
        prompt = f"{header}\n\n{sections}".strip()
        return self._apply_template_format(prompt)

    def _apply_template_format(self, text: str) -> str:
        if self.template_format == "jinja2":
            return _ensure_jinja_placeholders(text)
        return text


class PromptTemplateFactory:
    """Utility for constructing chat prompt templates from PromptSpec objects."""

    @staticmethod
    def build(
        spec: PromptSpec,
        *,
        template_format: Optional[str] = None,
        include_placeholders: Sequence[str] = ("chat_history",),
        extra_messages: Optional[Iterable[Tuple[str, str]]] = None,
        extra_sections: Optional[Iterable[PromptSection]] = None,
    ) -> ChatPromptTemplate:
        template_format = template_format or spec.template_format
        system_message = spec.render_system_prompt()

        messages: List[Union[Tuple[str, str], Dict[str, Any], MessagePlaceHolder]] = [
            ("system", system_message)
        ]

        if extra_sections:
            for section in extra_sections:
                rendered = section.render(template_format)
                if rendered:
                    messages.append(("system", rendered))

        for placeholder in include_placeholders:
            messages.append(MessagePlaceHolder(variable_name=placeholder))

        if extra_messages:
            messages.extend(extra_messages)

        template = ChatPromptTemplate.from_messages(
            messages=messages,
            template_format=template_format,
        )
        template.input_variables = sorted(set(template.input_variables + ["name"]))
        return template


PROMPT_MODULE_REGISTRY: Dict[
    str, Callable[["AgentProfileConfig"], Optional[PromptSection]]
] = {}


def register_prompt_module(
    name: str,
    factory: Callable[["AgentProfileConfig"], Optional[PromptSection]],
) -> None:
    """Register a callable that returns an additional prompt section."""
    PROMPT_MODULE_REGISTRY[name] = factory


def prompt_spec_from_profile(
    profile: AgentProfileConfig,
    *,
    default_name: str,
) -> Tuple[PromptSpec, List[PromptSection]]:
    """Build a PromptSpec and collect additional sections from the profile configuration."""

    spec = PromptSpec(
        name=profile.name or default_name,
        role=profile.role or "Assistant",
        goal=profile.goal or "Help users accomplish their tasks.",
        instructions=list(profile.instructions),
        style_guidelines=list(profile.style_guidelines),
        system_prompt=profile.system_prompt,
        template_format=profile.template_format,
    )

    extra_sections: List[PromptSection] = []

    for module_name in profile.modules:
        if module_name in profile.module_overrides:
            section = profile.module_overrides[module_name]
        else:
            factory = PROMPT_MODULE_REGISTRY.get(module_name)
            section = factory(profile) if factory else None
        if section:
            extra_sections.append(section)

    return spec, extra_sections


@dataclass
class PromptingAgentBase:
    """
    Lightweight helper focused on prompt construction and history injection.

    This class intentionally omits workflow, tool-execution, and LLM wiring; higher-level
    agent implementations can compose this helper alongside their own runtime concerns.
    """

    name: str = "Durable Agent"
    role: str = "Assistant"
    goal: str = "Deliver helpful responses."
    instructions: List[str] = field(default_factory=list)
    style_guidelines: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    template_format: str = "jinja2"
    include_chat_history: bool = True
    prompt_template: Optional[PromptTemplateBase] = None
    profile: Optional[AgentProfileConfig] = None

    def __post_init__(self) -> None:
        self._text_formatter = ColorTextFormatter()

        spec: PromptSpec
        extra_sections: List[PromptSection] = []
        if self.profile is not None:
            spec, extra_sections = prompt_spec_from_profile(
                self.profile,
                default_name=self.name,
            )
            # adopt values from spec so other consumers see resolved persona
            self.name = spec.name
            self.role = spec.role
            self.goal = spec.goal
            self.instructions = list(spec.instructions)
            self.style_guidelines = list(spec.style_guidelines)
            self.system_prompt = spec.system_prompt
            self.template_format = spec.template_format
        else:
            spec = self.to_prompt_spec()

        if self.prompt_template is None:
            placeholders = ["chat_history"] if self.include_chat_history else []
            self.prompt_template = PromptTemplateFactory.build(
                spec,
                include_placeholders=placeholders,
                extra_sections=extra_sections,
            )

        self._prefill_prompt_variables()

    def to_prompt_spec(self) -> PromptSpec:
        return PromptSpec(
            name=self.name,
            role=self.role,
            goal=self.goal,
            instructions=list(self.instructions),
            style_guidelines=list(self.style_guidelines),
            system_prompt=self.system_prompt,
            template_format=self.template_format,
        )

    def rebuild_prompt_template(
        self,
        *,
        spec: Optional[PromptSpec] = None,
        include_placeholders: Optional[Sequence[str]] = None,
    ) -> None:
        extra_sections: List[PromptSection] = []
        if spec is None:
            if self.profile is not None:
                spec, extra_sections = prompt_spec_from_profile(
                    self.profile,
                    default_name=self.name,
                )
            else:
                spec = self.to_prompt_spec()
        placeholders = (
            include_placeholders
            if include_placeholders is not None
            else (["chat_history"] if self.include_chat_history else [])
        )
        self.prompt_template = PromptTemplateFactory.build(
            spec,
            include_placeholders=placeholders,
            extra_sections=extra_sections,
        )
        self._prefill_prompt_variables()

    def _prefill_prompt_variables(self) -> None:
        if not self.prompt_template:
            return

        variables: Dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "goal": self.goal,
        }
        if self.instructions:
            variables["instructions"] = "\n".join(self.instructions)
        if self.style_guidelines:
            variables["style_guidelines"] = "\n".join(self.style_guidelines)

        self.prompt_template = self.prompt_template.pre_fill_variables(**variables)

    def build_initial_messages(
        self,
        user_input: Optional[Union[str, Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        **extra_variables: Any,
    ) -> List[Dict[str, Any]]:
        if not self.prompt_template:
            raise ValueError("Prompt template has not been initialised.")

        variables = dict(extra_variables)
        if self.include_chat_history and chat_history is not None:
            variables.setdefault("chat_history", chat_history)

        messages = self.prompt_template.format_prompt(**variables)

        if isinstance(user_input, str):
            messages.append({"role": "user", "content": user_input})
        elif isinstance(user_input, dict):
            messages.append(user_input)

        return messages

    @property
    def text_formatter(self) -> ColorTextFormatter:
        return self._text_formatter
