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
