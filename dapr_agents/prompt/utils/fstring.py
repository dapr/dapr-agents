from typing import Any, List
import re


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
    # Find all occurrences of {variable_name} in the template
    # This will match {name}, {role}, etc. even when they're part of sentences
    matches = re.findall(r"\{([^{}]+)\}", template)
    return matches
