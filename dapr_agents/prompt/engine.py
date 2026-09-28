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
from typing import Any, Callable, Dict, List, Literal, Tuple
from jinja2 import Environment, Template
from jinja2.meta import find_undeclared_variables

logger = logging.getLogger(__name__)

TemplateFormat = Literal["f-string", "jinja2"]


def render_fstring(template: str, **kwargs: Any) -> str:
    """
    Render an f-string style template by formatting it with the provided variables.

    Args:
        template (str): The f-string style template.
        **kwargs: Variables to be used for formatting the template.

    Returns:
        str: The rendered template string with variables replaced.
    """
    return template.format(**kwargs)


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


def render_jinja(template: str, **kwargs: Any) -> str:
    """
    Render a Jinja2 template using the provided variables.

    Args:
        template (str): The Jinja2 template string.
        **kwargs: Variables to be used in rendering the template.

    Returns:
        str: The rendered template string.
    """
    return Template(template).render(**kwargs)


def extract_jinja_variables(template: str) -> List[str]:
    """
    Extract undeclared variables from a Jinja2 template.

    Args:
        template (str): The Jinja2 template string.

    Returns:
        List[str]: A list of undeclared variable names in the template.
    """
    environment = Environment()
    parsed_content = environment.parse(template)
    undeclared_variables = find_undeclared_variables(parsed_content)
    return sorted(undeclared_variables)


render_fstring_template = render_fstring
render_jinja_template = render_jinja

DEFAULT_FORMATTER_MAPPING: Dict[str, Callable[..., str]] = {
    "f-string": render_fstring,
    "jinja2": render_jinja,
}

DEFAULT_VARIABLE_EXTRACTOR_MAPPING: Dict[str, Callable[[str], List[str]]] = {
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
    def render(
        cls, template: str, template_format: str = "f-string", **kwargs: Any
    ) -> str:
        """
        Render a template string using the specified format.

        Args:
            template (str): The template string to format.
            template_format (str): The template format ('f-string' or 'jinja2').
            **kwargs: Variables to populate placeholders.

        Returns:
            str: The formatted string.

        Raises:
            ValueError: If the template format is unsupported.
        """
        formatter = DEFAULT_FORMATTER_MAPPING.get(template_format)
        if not formatter:
            raise ValueError(f"Unsupported template format: {template_format}")
        return formatter(template, **kwargs)

    @classmethod
    def extract_variables(
        cls, template: str, template_format: str = "f-string"
    ) -> List[str]:
        """
        Extract placeholder variable names from a template string.

        Args:
            template (str): The template string.
            template_format (str): The template format ('f-string' or 'jinja2').

        Returns:
            List[str]: A list of extracted variable names.

        Raises:
            ValueError: If the template format is unsupported.
        """
        extractor = DEFAULT_VARIABLE_EXTRACTOR_MAPPING.get(template_format)
        if not extractor:
            raise ValueError(f"Unsupported template format: {template_format}")
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
