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

import logging
from string import Formatter
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple, cast
from jinja2 import Environment, nodes
from jinja2.meta import find_undeclared_variables

logger = logging.getLogger(__name__)

TemplateFormat = Literal["f-string", "jinja2"]
_UNSET: Any = object()


def _noop_filter(value: Any, *args: Any, **kwargs: Any) -> Any:
    return value


def _noop_test(value: Any, *args: Any, **kwargs: Any) -> bool:
    return True


def render_fstring(template: str, **kwargs: Any) -> str:
    """
    Render an f-string style template by formatting it with the provided variables.

    Args:
        template (str): The f-string style template.
        **kwargs: Variables to be used for formatting the template.

    Returns:
        str: The rendered template string with variables replaced.
    """
    return Formatter().vformat(template, (), kwargs)


def extract_fstring_variables(template: str) -> List[str]:
    """
    Extract variables from an f-string style template.

    Args:
        template (str): The f-string style template.

    Returns:
        List[str]: A list of variable names found in the template.
    """
    variables: List[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise ValueError(
                "Positional placeholders (e.g. '{}' or '{0}') are not supported; use named fields like '{name}'."
            )
        variables.append(field_name)
    return variables


def render_jinja(
    template: str,
    environment: Optional[Environment] = cast(Optional[Environment], _UNSET),
    **kwargs: Any,
) -> str:
    """
    Render a Jinja2 template using the provided variables.

    Args:
        template (str): The Jinja2 template string.
        environment (Optional[Environment]): Optional custom Jinja2 Environment.
        **kwargs: Variables to be used in rendering the template.

    Returns:
        str: The rendered template string.
    """
    if environment is _UNSET:
        env = Environment()
    elif isinstance(environment, Environment):
        env = environment
    elif environment is None:
        if "environment" in extract_jinja_variables(template):
            kwargs = {"environment": None, **kwargs}
        env = Environment()
    else:
        kwargs = {"environment": environment, **kwargs}
        env = Environment()
    return env.from_string(template).render(kwargs)


def extract_jinja_variables(
    template: str, environment: Optional[Environment] = None
) -> List[str]:
    """
    Extract undeclared variables from a Jinja2 template.

    Args:
        template (str): The Jinja2 template string.
        environment (Optional[Environment]): Optional custom Jinja2 Environment.

    Returns:
        List[str]: A list of undeclared variable names in the template.
    """
    if environment is not None:
        env = environment.overlay()
        env.filters = dict(environment.filters)
        env.tests = dict(environment.tests)
    else:
        env = Environment()
    parsed_content = env.parse(template)

    # Register stubs for any undeclared filters or tests in the template AST
    # so find_undeclared_variables does not fail with TemplateAssertionError.
    for node in parsed_content.find_all(nodes.Filter):
        if node.name not in env.filters:
            env.filters[node.name] = _noop_filter
    for node in parsed_content.find_all(nodes.Test):
        if node.name not in env.tests:
            env.tests[node.name] = _noop_test

    undeclared_variables = find_undeclared_variables(parsed_content)
    return sorted(undeclared_variables)


render_fstring_template = render_fstring
render_jinja_template = render_jinja

DEFAULT_FORMATTER_MAPPING: Dict[str, Callable[..., str]] = {
    "f-string": render_fstring,
    "jinja2": render_jinja,
}

DEFAULT_VARIABLE_EXTRACTOR_MAPPING: Dict[str, Callable[..., List[str]]] = {
    "f-string": extract_fstring_variables,
    "jinja2": extract_jinja_variables,
}


class TemplateEngine:
    """
    Encapsulated template engine providing unified formatting and variable extraction
    for prompt templates (supporting f-string and jinja2).
    """

    SUPPORTED_FORMATS: Tuple[str, ...] = ("f-string", "jinja2")

    @classmethod
    def render_mapping(
        cls,
        template: str,
        variables: Mapping[str, Any],
        template_format: str = "f-string",
        environment: Optional[Environment] = None,
    ) -> str:
        """
        Render a template string from a variable mapping without keyword-argument collisions.

        Args:
            template (str): The template string to format.
            variables (Mapping[str, Any]): Mapping of placeholder names to values.
            template_format (str): The template format ('f-string' or 'jinja2').
            environment (Optional[Environment]): Optional Jinja2 environment (for jinja2 format).

        Returns:
            str: The formatted string.

        Raises:
            ValueError: If the template format is unsupported.
        """
        formatter = DEFAULT_FORMATTER_MAPPING.get(template_format)
        if not formatter:
            raise ValueError(f"Unsupported template format: {template_format}")
        if formatter is render_fstring:
            return Formatter().vformat(template, (), variables)
        if formatter is render_jinja:
            env = environment if environment is not None else Environment()
            return env.from_string(template).render(dict(variables))
        if template_format == "jinja2" and environment is not None:
            return formatter(template, environment=environment, **variables)
        return formatter(template, **variables)

    @classmethod
    def render(
        cls,
        /,
        template: str,
        template_format: str = "f-string",
        environment: Optional[Environment] = cast(Optional[Environment], _UNSET),
        **kwargs: Any,
    ) -> str:
        """
        Render a template string using the specified format.

        Args:
            template (str): The template string to format.
            template_format (str): The template format ('f-string' or 'jinja2').
            environment (Optional[Environment]): Optional Jinja2 environment (for jinja2 format).
            **kwargs: Variables to populate placeholders.

        Returns:
            str: The formatted string.

        Raises:
            ValueError: If the template format is unsupported.
        """
        if template_format not in DEFAULT_FORMATTER_MAPPING:
            raise ValueError(f"Unsupported template format: {template_format}")
        resolved_env: Optional[Environment]
        if environment is _UNSET:
            resolved_env = None
        elif isinstance(environment, Environment):
            resolved_env = environment
        elif environment is None:
            if "environment" in cls.extract_variables(template, template_format):
                kwargs = {"environment": None, **kwargs}
            resolved_env = None
        else:
            kwargs = {"environment": environment, **kwargs}
            resolved_env = None
        return cls.render_mapping(
            template,
            kwargs,
            template_format=template_format,
            environment=resolved_env,
        )

    @classmethod
    def extract_variables(
        cls,
        template: str,
        template_format: str = "f-string",
        environment: Optional[Environment] = None,
    ) -> List[str]:
        """
        Extract placeholder variable names from a template string.

        Args:
            template (str): The template string.
            template_format (str): The template format ('f-string' or 'jinja2').
            environment (Optional[Environment]): Optional Jinja2 environment (for jinja2 format).

        Returns:
            List[str]: A list of extracted variable names.

        Raises:
            ValueError: If the template format is unsupported.
        """
        extractor = DEFAULT_VARIABLE_EXTRACTOR_MAPPING.get(template_format)
        if not extractor:
            raise ValueError(f"Unsupported template format: {template_format}")
        if template_format == "jinja2" and environment is not None:
            return extractor(template, environment=environment)
        return extractor(template)


__all__ = [
    "TemplateEngine",
    "TemplateFormat",
    "render_fstring",
    "extract_fstring_variables",
    "render_jinja",
    "extract_jinja_variables",
    "render_fstring_template",
    "render_jinja_template",
    "DEFAULT_FORMATTER_MAPPING",
    "DEFAULT_VARIABLE_EXTRACTOR_MAPPING",
]
