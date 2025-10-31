from pydantic import BaseModel, Field
from typing import Optional, Literal, Any, Dict, List, Union, Callable
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.chat import ChatPromptTemplate
from dapr_agents.types.message import MessagePlaceHolder
from dapr_agents.prompt.agent_prompt_context import Context
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


class Prompt(BaseModel):
    """
    Prompt is the base class for agent prompt fields.
    """

    system_prompt: Optional[str] = Field(
        default=None,
        description="A custom system prompt, overriding name, role, goal, and instructions.",
    )
    template: Optional[PromptTemplateBase] = Field(
        default=None, description="The prompt template for the agent."
    )
    # TODO: we should have a system_template, prompt_template, and response_template, or better separation here.
    # If we have something like a customer service agent, we want diff templates for different types of interactions.
    # In future, we could also have a way to dynamically change the template based on the context of the interaction.
    template_format: Literal["f-string", "jinja2"] = Field(
        default="jinja2",
        description="The format used for rendering the prompt template.",
    )
    context: Optional[Context] = Field(default=None, description="Prompt context")

    def model_post_init(self, __context: Any) -> None:
        if self.context is None:
            self.context = Context()
        self.template = self._init()
        self._validate_prompt_template()
        self.prefill_agent_attributes()

    def _validate_prompt_template(self) -> None:
        """
        Ensures chat_history is always available, injects any declared attributes,
        and warns if the user set attributes that aren't in the template.
        """
        if not self.template:
            return

        # Always make chat_history available
        vars_set = set(self.template.input_variables) | {"chat_history"}

        # Inject any attributes the template declares
        valid_attrs, unused_attrs = self._collect_template_attrs()
        vars_set |= set(valid_attrs.keys())
        self.template.input_variables = list(vars_set)

        if unused_attrs:
            logger.warning(
                "Agent attributes set but not referenced in prompt_template: "
                f"{', '.join(unused_attrs)}. Consider adding them to input_variables."
            )

    def prefill_agent_attributes(self) -> None:
        """
        Pre-fill prompt_template with agent attributes if specified in `input_variables`.
        Uses _collect_template_attrs to avoid duplicate logic and ensure consistency.
        """
        if not self.template:
            return

        # Re-use our helper to split valid vs. unused
        valid_attrs, unused_attrs = self._collect_template_attrs()

        if unused_attrs:
            logger.warning(
                "Agent attributes set but not used in prompt_template: "
                f"{', '.join(unused_attrs)}. Consider adding them to input_variables."
            )

        if valid_attrs:
            self.template = self.template.pre_fill_variables(**valid_attrs)
            logger.debug(f"Pre-filled template with: {list(valid_attrs.keys())}")
        else:
            logger.debug("No prompt_template variables needed pre-filling.")

    def _init(self) -> PromptTemplateBase:
        """
        Determines which prompt template to use for the agent:
        1. If the user supplied one, use it.
        2. Else if the LLM client already has one, adopt that.
        3. Else generate a system_prompt and ChatPromptTemplate from agent attributes.

        Returns:
            PromptTemplateBase: The selected or constructed prompt template.
        """
        if self.template:
            logger.info("Using provided prompt template")
            return self.template

        # Build from system_prompt or attributes
        if not self.system_prompt:
            logger.debug("Constructing system_prompt from attributes")
            self.system_prompt = self.construct_system_prompt()

        logger.debug("Building ChatPromptTemplate from system_prompt")
        return self.construct_template()

    def _collect_template_attrs(self) -> tuple[Dict[str, str], List[str]]:
        """
        Collect agent attributes for prompt template pre-filling and warn about unused ones.
        - valid: attributes set on self and declared in prompt_template.input_variables.
        - unused: attributes set on self but not present in the template.
        Returns:
            (valid, unused): Tuple of dict of valid attrs and list of unused attr names.
        """
        attrs = ["name", "role", "goal", "instructions"]
        valid: Dict[str, str] = {}
        unused: List[str] = []
        if not self.template or not hasattr(self.template, "input_variables"):
            return valid, attrs  # No template, all attrs are unused
        original = set(self.template.input_variables)

        for attr in attrs:
            val = getattr(self.context, attr, None) if self.context else None
            if val is None:
                continue
            if attr in original:
                # Only join instructions if it's a list and the template expects it
                if attr == "instructions" and isinstance(val, list):
                    valid[attr] = "\n".join(val)
                else:
                    valid[attr] = str(val)
            else:
                unused.append(attr)
        return valid, unused

    def construct_system_prompt(self) -> str:
        """
        Build the system prompt for the agent using a single template string.
        - Fills in the current date.
        - Leaves placeholders for name, role, goal, and instructions as variables (instructions only if set).
        - Converts placeholders to Jinja2 syntax if requested.

        Returns:
            str: The formatted system prompt string.
        """
        # Default f-string template; placeholders will be swapped to Jinja if needed.
        default_sys_prompt = """
            # Today's date is: {date}

            ## Name
            Your name is {name}.

            ## Role
            Your role is {role}.

            ## Goal
            {goal}.

            ## Instructions
            {instructions}.
            """.strip()

        # Only fill in the date; leave all other placeholders as variables
        has_instructions = bool(self.context and self.context.instructions)
        instructions_placeholder = "{instructions}" if has_instructions else ""
        filled = default_sys_prompt.format(
            date=(
                self.context.date
                if self.context and self.context.date
                else datetime.now().strftime("%B %d, %Y")
            ),
            name="{name}",
            role="{role}",
            goal="{goal}",
            instructions=instructions_placeholder,
        )

        # If using Jinja2, swap braces for all placeholders
        if self.template_format == "jinja2":
            # Replace every {foo} with {{foo}}
            return re.sub(r"\{(\w+)\}", r"{{\1}}", filled)
        else:
            return filled

    def construct_template(self) -> ChatPromptTemplate:
        """
        Constructs a ChatPromptTemplate that includes the system prompt and a placeholder for chat history.
        Ensures that the template is flexible and adaptable to dynamically handle pre-filled variables.

        Returns:
            ChatPromptTemplate: A formatted prompt template for the agent.
        """
        # Construct the system prompt if not provided
        system_prompt = self.system_prompt or self.construct_system_prompt()

        # Create the template with placeholders for system message and chat history
        return ChatPromptTemplate.from_messages(
            messages=[
                ("system", system_prompt),
                MessagePlaceHolder(variable_name="chat_history"),
            ],
            template_format=self.template_format,
        )

    def prefill_template(self, **kwargs: Union[str, Callable[[], str]]) -> None:
        """
        Pre-fills the prompt template with specified variables, updating input variables if applicable.

        Args:
            **kwargs: Variables to pre-fill in the prompt template. These can be strings or callables
                    that return strings.

        Notes:
            - Existing pre-filled variables will be overwritten by matching keys in `kwargs`.
            - This method does not affect the `chat_history` which is dynamically updated.
        """
        if not self.template:
            raise ValueError(
                "Prompt template must be initialized before pre-filling variables."
            )

        self.template = self.template.pre_fill_variables(**kwargs)
        logger.debug(f"Pre-filled prompt template with variables: {kwargs.keys()}")
