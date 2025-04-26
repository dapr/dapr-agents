from dapr_agents.types.message import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from dapr_agents.prompt.utils.fstring import extract_fstring_variables
from dapr_agents.prompt.utils.jinja import extract_jinja_variables
from typing import Dict, Any, Tuple, Optional, Union, List, Literal
from jinja2 import Template, TemplateError
from pathlib import Path
import yaml
import re
import os
import json
import logging

logger = logging.getLogger(__name__)


class RoleMap:
    """
    A utility class to map roles to message types.
    This ensures consistency in how roles like 'system', 'user', etc., are handled.
    """

    _ROLE_MAP = {
        "system": SystemMessage,
        "user": UserMessage,
        "assistant": AssistantMessage,
        "tool": ToolMessage,
    }

    @classmethod
    def get_message_class(cls, role: str) -> Union[BaseMessage, None]:
        """Get the message class for a given role."""
        role = role.lower()
        return cls._ROLE_MAP.get(role, None)

    @classmethod
    def add_custom_role(cls, role: str, message_class: BaseMessage):
        """Add custom roles dynamically."""
        cls._ROLE_MAP[role.lower()] = message_class


class PromptyHelper:
    """
    Utility class for handling operations related to Prompty files,
    including parsing frontmatter, file loading, environment variable resolution,
    and preparing inputs for Prompty templates.
    """

    @staticmethod
    def parse_prompty_content(
        prompty_source: Union[Path, str],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Extract YAML frontmatter and markdown content from a Prompty source, which can be a file path or raw content.

        Args:
            prompty_source (Union[Path, str]): A file path to the Prompty file or the inline content as a string.

        Returns:
            Tuple[Dict[str, Any], str]: A tuple with parsed YAML metadata and markdown content.

        Raises:
            ValueError: If the frontmatter format is invalid.
        """
        # Check if `prompty_source` is a file path and read from it
        if isinstance(prompty_source, Path) and prompty_source.exists():
            with open(prompty_source, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = prompty_source  # Treat as inline content

        pattern = r"-{3,}\n(.*?)\n-{3,}\n(.*)"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            yaml_frontmatter = match.group(1)
            markdown_content = match.group(2)
            yaml_metadata = yaml.safe_load(yaml_frontmatter)
            return yaml_metadata, markdown_content
        else:
            raise ValueError(
                "Invalid Prompty content format: could not extract frontmatter."
            )

    @staticmethod
    def resolve_env(
        value: Any, parent: Optional[Path] = None, env_error: bool = True
    ) -> Any:
        """
        Resolve environment variables or file references from the input value.
        If the value is not a string (e.g., a list or dict), return it as-is.
        """
        # If the value is not a string, just return it (no need to resolve env variables)
        if not isinstance(value, str):
            return value

        # Handle string values (resolve environment variables or file references)
        if value.startswith("${") and value.endswith("}"):
            variable = value[2:-1].split(":")

            # Handle environment variables ${env:VAR_NAME} or legacy ${VAR_NAME}
            if variable[0] == "env" and len(variable) > 1:
                result = PromptyHelper.process_env(
                    variable[1], env_error, variable[2] if len(variable) > 2 else None
                )

            # Handle file references ${file:path/to/file}
            elif variable[0] == "file" and len(variable) > 1:
                if parent:
                    result = PromptyHelper.process_file(variable[1], parent)
                else:
                    raise ValueError(f"File parent path not provided for {variable[1]}")

            # Fallback for legacy environment variables
            else:
                result = PromptyHelper.process_env(variable[0], env_error=False)
                if not result and len(variable) > 1:
                    result = variable[1]  # Use default value
                elif not result and env_error:
                    raise ValueError(f"Variable {variable[0]} not found in environment")

            return result

        # If it's not an environment variable reference, just return the value
        return value

    @staticmethod
    def process_env(
        var_name: str, env_error: bool = True, default_value: Optional[str] = None
    ) -> Any:
        value = os.getenv(var_name, default_value)
        if value is None and env_error:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return value

    @staticmethod
    def process_file(path: str, parent: Path) -> Any:
        """
        Process file resolution and return its contents as JSON.
        """
        full_path = parent / Path(path).resolve()
        if not full_path.exists():
            raise FileNotFoundError(f"File {full_path} not found.")

        with open(full_path, "r") as file:
            items = json.load(file)
            if isinstance(items, list):
                return [PromptyHelper.normalize(value, parent) for value in items]
            elif isinstance(items, dict):
                return {
                    key: PromptyHelper.normalize(value, parent)
                    for key, value in items.items()
                }
            return items

    @staticmethod
    def normalize(
        attribute: Any, parent: Path = Path().resolve(), env_error=True
    ) -> Any:
        """
        Normalize the attribute by resolving environment variables, file references, and default values.

        Args:
            attribute (Any): The attribute to normalize.
            parent (Path): The base path for file references, defaulting to the current working directory.
            env_error (bool): Whether to raise an error if an environment variable is missing.

        Returns:
            Any: The normalized attribute value.
        """
        if isinstance(attribute, str):
            attribute: str = attribute.strip()
            if attribute.startswith("${") and attribute.endswith("}"):
                variable = attribute[2:-1].split(":")
                if variable[0] == "env" and len(variable) > 1:
                    return PromptyHelper.process_env(
                        variable[1],
                        env_error,
                        variable[2] if len(variable) > 2 else None,
                    )
                elif variable[0] == "file" and len(variable) > 1:
                    return PromptyHelper.process_file(variable[1], parent)
                else:
                    v = PromptyHelper.process_env(variable[0], False)
                    if len(v) == 0:
                        if len(variable) > 1:
                            return variable[1]
                        elif env_error:
                            raise ValueError(
                                f"Variable {variable[0]} not found in environment"
                            )
                    return v
            elif (
                attribute.startswith("file:")
                and Path(parent / attribute.split(":")[1]).exists()
            ):
                return PromptyHelper.process_file(attribute.split(":")[1], parent)
            return attribute
        elif isinstance(attribute, list):
            return [PromptyHelper.normalize(value, parent) for value in attribute]
        elif isinstance(attribute, dict):
            return {
                key: PromptyHelper.normalize(value, parent)
                for key, value in attribute.items()
            }
        return attribute

    @staticmethod
    def prepare_inputs(
        inputs: Dict[str, Any],
        prompty_inputs: Dict[str, Any],
        prompty_sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare the inputs by merging them with the default template values and resolving environment variables.
        """
        prepared_inputs = {}

        # Mapping from string types to Python types
        type_mapping = {
            "string": str,
            "number": (int, float),
            "array": list,
            "object": dict,
            "boolean": bool,
        }

        # Merge provided inputs with sample data in the template, if any.
        merged_inputs = {**prompty_sample, **inputs}

        # First, check for any user-provided inputs not defined in the template
        for key in inputs:
            if key not in prompty_inputs:
                # Issue a warning for undefined inputs
                logger.warning(
                    f"Input '{key}' is not defined in the template. It will be included as-is."
                )
                # Include the user-provided input as-is
                prepared_inputs[key] = inputs[key]

        # Process the defined template inputs
        for key, settings in prompty_inputs.items():
            # Check if the key exists in the provided inputs or merged inputs (user + sample)
            if key in merged_inputs:
                expected_type = settings.get("type", None)
                if isinstance(expected_type, str):
                    expected_type = type_mapping.get(expected_type.lower(), None)

                # Optionally validate types if defined in `settings`
                if expected_type and not isinstance(merged_inputs[key], expected_type):
                    raise TypeError(
                        f"Input '{key}' must be of type {expected_type.__name__}, but got {type(merged_inputs[key]).__name__}"
                    )
                prepared_inputs[key] = PromptyHelper.resolve_env(merged_inputs[key])

            # Check if there is a default value in the settings
            elif "default" in settings:
                prepared_inputs[key] = settings["default"]

            # Raise an error if the input is required and not provided
            elif settings.get("required", False):
                raise ValueError(f"Input '{key}' is required but not provided.")

            # If it's optional and no default or input is provided, assign None
            else:
                prepared_inputs[key] = None

        # process any additional keys that exist in merged_inputs but not in prompty_inputs
        for key in merged_inputs:
            if key not in prepared_inputs:
                # Log a warning if the input is not defined in the template
                logger.warning(
                    f"Input '{key}' is not defined in the template but will be included as-is."
                )
                prepared_inputs[key] = merged_inputs[key]

        return prepared_inputs

    @staticmethod
    def render_content(content: str, inputs: Dict[str, Any]) -> str:
        """
        Render the Prompty content using Jinja2 with the provided inputs.
        """
        try:
            template = Template(content)
            rendered_content = template.render(inputs)
        except TemplateError as e:
            raise ValueError(f"Template rendering error: {e}")

        return rendered_content

    @staticmethod
    def to_prompt(
        content: str,
        inputs: Dict[str, Any] = None,
        api_type: Literal["chat", "completion"] = "chat",
    ) -> Union[str, List[BaseMessage]]:
        """
        Parse the content of the Prompty template using the provided inputs.
        Return either a formatted string (for completion APIs) or a list of messages (for chat APIs).
        """
        # Render the template content using Jinja2
        if not inputs:
            rendered_content = content
        else:
            rendered_content = PromptyHelper.render_content(content, inputs)

        if api_type == "chat":
            # For chat models, split the content into role-based messages
            return PromptyHelper.parse_as_messages(rendered_content)
        else:
            # For completion models, return the entire rendered content as a string
            return rendered_content

    @staticmethod
    def parse_as_messages(content: str) -> List[BaseMessage]:
        """
        Parse the rendered content into a list of messages based on roles.
        If no roles are detected, return an empty list.

        Args:
            content (str): The content string to parse.

        Returns:
            List[BaseMessage]: List of messages with appropriate roles or an empty list if no roles are found.
        """
        chunks = PromptyHelper.parse_role_content(content)

        # Immediately return an empty list if no role-based chunks were found
        if not chunks:
            logger.info("No role-based content found; returning an empty list.")
            return []

        # Parse the content based on detected roles
        messages = []
        role = None

        for chunk in chunks:
            if chunk.lower() in RoleMap._ROLE_MAP.keys():
                role = chunk  # Assign role if chunk matches a defined role
            elif (
                role
            ):  # If a role is set, treat this chunk as the content for that role
                messages.append(PromptyHelper.to_message(role, chunk))
                role = None
            else:
                raise ValueError(f"Unexpected content without a role: {chunk}")

        return messages

    @staticmethod
    def parse_role_content(content: str) -> List[str]:
        """
        Parse the formatted content into chunks based on role delimiters.
        Only returns chunks if role-based content is detected.
        """
        # Regex to split on role definitions, ensuring we capture role-based content
        role_pattern = (
            r"(?i)^\s*#?\s*(" + "|".join(RoleMap._ROLE_MAP.keys()) + r")\s*:\s*\n"
        )

        # Check if any role-based pattern exists in the content
        if not re.search(role_pattern, content, flags=re.MULTILINE):
            # Return an empty list if no role-based pattern is found
            return []

        # Split content by role pattern if a role is detected
        chunks = re.split(role_pattern, content, flags=re.MULTILINE)

        # Filter out any empty or whitespace-only chunks
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    @staticmethod
    def to_message(role: str, content: str) -> BaseMessage:
        """
        Parse a single chunk of content into a message object.
        """
        role = role.strip().lower()
        content = content.strip()

        logger.debug(f"Parsing role: '{role}', content: '{content[:30]}...'")

        # Map role to a message class
        message_class = RoleMap.get_message_class(role)
        if not message_class:
            raise ValueError(f"Invalid message role: '{role}'")

        if not content:
            raise ValueError(f"Content missing for role: '{role}'")

        return message_class(content=content)

    @staticmethod
    def extract_placeholders_from_content(
        content: str, template_format: Literal["f-string", "jinja2"] = "jinja2"
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """
        Extract undeclared variables from the content based on the specified template format.

        Args:
            content (str): The content of the Prompty template from which to extract placeholders.
            template_format (Literal["f-string", "jinja2"]): The format of the template, either "f-string" or "jinja2". Default is "jinja2".

        Returns:
            Union[List[str], Tuple[List[str], List[str]]]:
                - For "f-string" format: A list of variable names.
                - For "jinja2" format: A list of undeclared variables.
        """
        if template_format == "jinja2":
            return extract_jinja_variables(content)
        elif template_format == "f-string":
            return extract_fstring_variables(content)
        else:
            raise ValueError(f"Unsupported template format: {template_format}")
