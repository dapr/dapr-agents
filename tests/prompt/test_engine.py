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

"""Unit tests for the encapsulated TemplateEngine module."""

import pytest

from dapr_agents.prompt.engine import (
    TemplateEngine,
    extract_fstring_variables,
    extract_jinja_variables,
    render_fstring,
    render_jinja,
)


class TestTemplateEngineRender:
    """Tests for TemplateEngine.render() and underlying renderers."""

    def test_render_fstring_simple(self):
        result = TemplateEngine.render("Hello {name}!", "f-string", name="World")
        assert result == "Hello World!"

    def test_render_fstring_escaped_braces(self):
        result = TemplateEngine.render(
            'JSON {{"key": "{val}"}}', "f-string", val="data"
        )
        assert result == 'JSON {"key": "data"}'

    def test_render_jinja_simple(self):
        result = TemplateEngine.render("Hello {{ name }}!", "jinja2", name="World")
        assert result == "Hello World!"

    def test_render_jinja_conditionals(self):
        template = (
            "{% if is_admin %}Admin: {{ name }}{% else %}User: {{ name }}{% endif %}"
        )
        admin_res = TemplateEngine.render(
            template, "jinja2", is_admin=True, name="Alice"
        )
        assert admin_res == "Admin: Alice"
        user_res = TemplateEngine.render(template, "jinja2", is_admin=False, name="Bob")
        assert user_res == "User: Bob"

    def test_render_unsupported_format_raises(self):
        with pytest.raises(
            ValueError, match="Unsupported template format: unsupported"
        ):
            TemplateEngine.render("Hello", template_format="unsupported")


class TestTemplateEngineExtractVariables:
    """Tests for TemplateEngine.extract_variables() and underlying extractors."""

    def test_extract_fstring_variables(self):
        vars_found = TemplateEngine.extract_variables(
            "Hello {first_name}, {last_name}!", "f-string"
        )
        assert vars_found == ["first_name", "last_name"]

    def test_extract_fstring_ignores_escaped_braces(self):
        vars_found = TemplateEngine.extract_variables(
            '{{"literal": true}} and {real_var}', "f-string"
        )
        assert vars_found == ["real_var"]

    def test_extract_fstring_rejects_positional(self):
        with pytest.raises(ValueError, match="Positional placeholders"):
            TemplateEngine.extract_variables("Hello {}!", "f-string")

        with pytest.raises(ValueError, match="Positional placeholders"):
            TemplateEngine.extract_variables("Hello {0}!", "f-string")

    def test_extract_jinja_variables(self):
        vars_found = TemplateEngine.extract_variables(
            "Hello {{ user.name }}, you have {{ count }} items.", "jinja2"
        )
        assert sorted(vars_found) == ["count", "user"]

    def test_extract_jinja_variables_with_undeclared_custom_filter_and_test(self):
        template = (
            "Hello {{ name | custom_filter('arg') }}! "
            "{% if score is passing %}Pass: {{ score }}{% endif %}"
        )
        vars_found = TemplateEngine.extract_variables(template, "jinja2")
        assert vars_found == ["name", "score"]

    def test_extract_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported template format: mustache"):
            TemplateEngine.extract_variables("{{name}}", template_format="mustache")

    def test_extract_variables_empty_string(self):
        assert TemplateEngine.extract_variables("", "f-string") == []
        assert TemplateEngine.extract_variables("", "jinja2") == []


class TestDirectFunctions:
    """Tests for direct render and extract functions."""

    def test_render_fstring_direct(self):
        assert render_fstring("{a}+{b}={c}", a=1, b=2, c=3) == "1+2=3"

    def test_render_jinja_direct(self):
        assert render_jinja("{{ items | join(', ') }}", items=["a", "b"]) == "a, b"

    def test_render_jinja_with_custom_environment(self):
        from jinja2 import Environment

        env = Environment()
        env.filters["shout"] = lambda s: s.upper() + "!"
        result = TemplateEngine.render(
            "Hello {{ name | shout }}", "jinja2", environment=env, name="world"
        )
        assert result == "Hello WORLD!"

    def test_render_empty_string(self):
        assert TemplateEngine.render("", "f-string") == ""
        assert TemplateEngine.render("", "jinja2") == ""

    def test_extract_fstring_variables_direct(self):
        assert extract_fstring_variables("{x} and {y}") == ["x", "y"]

    def test_extract_jinja_variables_direct(self):
        assert extract_jinja_variables("{{ x }} and {{ y }}") == ["x", "y"]

    def test_extract_jinja_variables_does_not_mutate_caller_environment(self):
        from jinja2 import Environment

        caller_env = Environment()
        original_filters = dict(caller_env.filters)
        original_tests = dict(caller_env.tests)

        vars_found = extract_jinja_variables(
            "{{ name | undeclared_filter }} {% if x is undeclared_test %}{{ x }}{% endif %}",
            environment=caller_env,
        )
        assert vars_found == ["name", "x"]
        assert "undeclared_filter" not in caller_env.filters
        assert "undeclared_test" not in caller_env.tests
        assert caller_env.filters == original_filters
        assert caller_env.tests == original_tests

    def test_render_mapping_reserved_parameter_names(self):
        variables = {
            "template": "T",
            "template_format": "F",
            "environment": "prod",
            "cls": "MyClass",
            "self": "me",
        }
        fstring_res = TemplateEngine.render_mapping(
            "{template}:{template_format}:{environment}:{cls}:{self}",
            variables,
            "f-string",
        )
        assert fstring_res == "T:F:prod:MyClass:me"

        jinja_res = TemplateEngine.render_mapping(
            "{{ template }}:{{ template_format }}:{{ environment }}:{{ cls }}",
            variables,
            "jinja2",
        )
        assert jinja_res == "T:F:prod:MyClass"

    def test_render_and_render_jinja_handle_colliding_kwargs(self):
        assert (
            TemplateEngine.render("Env: {environment}", "f-string", environment="prod")
            == "Env: prod"
        )
        assert (
            TemplateEngine.render(
                "Env: {{ environment }}", "jinja2", environment="prod"
            )
            == "Env: prod"
        )
        assert (
            TemplateEngine.render("Env: {{ environment }}", "jinja2", environment=None)
            == "Env: None"
        )
        assert render_jinja("Env: {{ environment }}", environment="prod") == "Env: prod"
        assert render_jinja("Env: {{ environment }}", environment=None) == "Env: None"
        assert (
            TemplateEngine.render(
                "Cls: {cls}, Self: {self}", "f-string", **{"cls": "C", "self": "S"}
            )
            == "Cls: C, Self: S"
        )
