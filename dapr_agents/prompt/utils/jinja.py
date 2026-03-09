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

from jinja2 import Environment, Template
from jinja2.meta import find_undeclared_variables
from typing import List, Any


def render_jinja_template(template: str, **kwargs: Any) -> str:
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
    Extract undeclared variables from a Jinja2 template. These variables represent placeholders
    that need to be filled in during rendering.

    Args:
        template (str): The Jinja2 template string.

    Returns:
        List[str]: A list of undeclared variable names in the template.
    """
    environment = Environment()
    parsed_content = environment.parse(template)

    # Extract all undeclared variables (placeholders)
    undeclared_variables = find_undeclared_variables(parsed_content)

    return list(undeclared_variables)
