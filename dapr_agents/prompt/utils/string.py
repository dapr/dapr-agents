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

from dapr_agents.prompt.utils.jinja import (
    render_jinja_template,
    extract_jinja_variables,
)
from dapr_agents.prompt.utils.fstring import (
    render_fstring_template,
    extract_fstring_variables,
)
from typing import Any, Dict

DEFAULT_FORMATTER_MAPPING = {
    "f-string": render_fstring_template,
    "jinja2": render_jinja_template,
}

DEFAULT_VARIABLE_EXTRACTOR_MAPPING = {
    "f-string": extract_fstring_variables,
    "jinja2": extract_jinja_variables,
}


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
        formatter = DEFAULT_FORMATTER_MAPPING.get(template_format)
        if not formatter:
            raise ValueError(f"Unsupported template format: {template_format}")
        return formatter(content, **kwargs)

    @staticmethod
    def extract_variables(template: str, template_format: str) -> Dict[str, Any]:
        """
        Extract variables from the template content based on the template format.

        Args:
            template (str): The template content string.
            template_format (str): Template format ('f-string' or 'jinja2').

        Returns:
            Dict[str, Any]: A dictionary of extracted variables.
        """
        extractor = DEFAULT_VARIABLE_EXTRACTOR_MAPPING.get(template_format)
        if not extractor:
            raise ValueError(f"Unsupported template format: {template_format}")
        return extractor(template)
