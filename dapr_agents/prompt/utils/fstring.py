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

from typing import Any, List
from string import Formatter


def render_fstring_template(template: str, **kwargs: Any) -> str:
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
    # Use the stdlib string.Formatter, which shares str.format() semantics, so
    # extraction matches how render_fstring_template renders the template. This
    # correctly treats escaped braces ({{ and }}) as literal characters instead
    # of variables, and drops any format spec (e.g. {value:>10} -> "value").
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
