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

"""Compatibility shim for string prompt utilities. Use dapr_agents.prompt.engine instead."""

from typing import Any, List

from dapr_agents.prompt.engine import (
    DEFAULT_FORMATTER_MAPPING,
    DEFAULT_VARIABLE_EXTRACTOR_MAPPING,
    TemplateEngine,
)


class StringPromptHelper:
    """
    Utility class for handling string-based operations, such as template formatting,
    extracting variables, and normalizing input data.
    """

    @staticmethod
    def format_content(content: str, template_format: str, **kwargs: Any) -> str:
        """
        Apply template formatting to the content string using the specified format.

        Args:
            content (str): The content string to format.
            template_format (str): Template format ('f-string' or 'jinja2').
            **kwargs: Variables for populating placeholders within the content.

        Returns:
            str: The formatted content.
        """
        return TemplateEngine.render(content, template_format, **kwargs)

    @staticmethod
    def extract_variables(template: str, template_format: str) -> List[str]:
        """
        Extract variables from the template content based on the template format.

        Args:
            template (str): The template content string.
            template_format (str): Template format ('f-string' or 'jinja2').

        Returns:
            List[str]: A list of extracted variable names.
        """
        return TemplateEngine.extract_variables(template, template_format)


__all__ = [
    "StringPromptHelper",
    "DEFAULT_FORMATTER_MAPPING",
    "DEFAULT_VARIABLE_EXTRACTOR_MAPPING",
]
