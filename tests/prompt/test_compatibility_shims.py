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

"""Tests for backward-compatible imports and re-exports in prompt modules and shims."""


def test_utils_string_reexports():
    """Verify that utils/string.py re-exports expected helpers and mappings."""
    from dapr_agents.prompt.engine import (
        DEFAULT_FORMATTER_MAPPING as ENGINE_FORMATTERS,
        DEFAULT_VARIABLE_EXTRACTOR_MAPPING as ENGINE_EXTRACTORS,
        extract_fstring_variables as engine_extract_fstring,
        extract_jinja_variables as engine_extract_jinja,
        render_fstring_template as engine_render_fstring,
        render_jinja_template as engine_render_jinja,
    )
    from dapr_agents.prompt.utils.string import (
        DEFAULT_FORMATTER_MAPPING,
        DEFAULT_VARIABLE_EXTRACTOR_MAPPING,
        StringPromptHelper,
        extract_fstring_variables,
        extract_jinja_variables,
        render_fstring_template,
        render_jinja_template,
    )

    assert render_fstring_template is engine_render_fstring
    assert extract_fstring_variables is engine_extract_fstring
    assert render_jinja_template is engine_render_jinja
    assert extract_jinja_variables is engine_extract_jinja
    assert DEFAULT_FORMATTER_MAPPING is ENGINE_FORMATTERS
    assert DEFAULT_VARIABLE_EXTRACTOR_MAPPING is ENGINE_EXTRACTORS

    # Test execution through shim
    assert render_fstring_template("Hello {name}", name="world") == "Hello world"
    assert extract_fstring_variables("Hello {name}") == ["name"]
    assert render_jinja_template("Hello {{ name }}", name="world") == "Hello world"
    assert extract_jinja_variables("Hello {{ name }}") == ["name"]
    assert (
        StringPromptHelper.format_content("Hi {v}", "f-string", v="there") == "Hi there"
    )


def test_utils_chat_reexports():
    """Verify that utils/chat.py re-exports expected helpers, message classes, and mappings."""
    from dapr_agents.prompt.chat import ChatPromptHelper as ChatPromptHelperCore
    from dapr_agents.prompt.engine import (
        DEFAULT_FORMATTER_MAPPING as ENGINE_FORMATTERS,
        DEFAULT_VARIABLE_EXTRACTOR_MAPPING as ENGINE_EXTRACTORS,
        extract_fstring_variables as engine_extract_fstring,
        extract_jinja_variables as engine_extract_jinja,
        render_fstring_template as engine_render_fstring,
        render_jinja_template as engine_render_jinja,
    )
    from dapr_agents.prompt.utils.chat import (
        DEFAULT_FORMATTER_MAPPING,
        DEFAULT_VARIABLE_EXTRACTOR_MAPPING,
        AssistantMessage,
        BaseMessage,
        ChatPromptHelper,
        MessagePlaceHolder,
        SystemMessage,
        ToolMessage,
        UserMessage,
        extract_fstring_variables,
        extract_jinja_variables,
        render_fstring_template,
        render_jinja_template,
    )
    from dapr_agents.types.message import (
        AssistantMessage as CoreAssistantMessage,
        BaseMessage as CoreBaseMessage,
        MessagePlaceHolder as CoreMessagePlaceHolder,
        SystemMessage as CoreSystemMessage,
        ToolMessage as CoreToolMessage,
        UserMessage as CoreUserMessage,
    )

    assert render_fstring_template is engine_render_fstring
    assert extract_fstring_variables is engine_extract_fstring
    assert render_jinja_template is engine_render_jinja
    assert extract_jinja_variables is engine_extract_jinja
    assert DEFAULT_FORMATTER_MAPPING is ENGINE_FORMATTERS
    assert DEFAULT_VARIABLE_EXTRACTOR_MAPPING is ENGINE_EXTRACTORS
    assert ChatPromptHelper is ChatPromptHelperCore

    assert BaseMessage is CoreBaseMessage
    assert SystemMessage is CoreSystemMessage
    assert UserMessage is CoreUserMessage
    assert AssistantMessage is CoreAssistantMessage
    assert ToolMessage is CoreToolMessage
    assert MessagePlaceHolder is CoreMessagePlaceHolder

    # Test instantiation of re-exported message classes
    sys_msg = SystemMessage(content="system instructions")
    assert sys_msg.role == "system"
    user_msg = UserMessage(content="user question")
    assert user_msg.role == "user"


def test_prompt_string_reexports():
    """Verify that prompt/string.py re-exports StringPromptHelper."""
    from dapr_agents.prompt.string import StringPromptHelper, StringPromptTemplate
    from dapr_agents.prompt.utils.string import (
        StringPromptHelper as UtilStringPromptHelper,
    )

    assert StringPromptHelper is UtilStringPromptHelper
    assert (
        StringPromptHelper.format_content("{greeting}", "f-string", greeting="hello")
        == "hello"
    )
    tmpl = StringPromptTemplate.from_template("Hello {name}")
    assert tmpl.format_prompt(name="world") == "Hello world"


def test_prompt_chat_reexports():
    """Verify that prompt/chat.py re-exports message classes and helpers."""
    from dapr_agents.prompt.chat import (
        AssistantMessage,
        BaseMessage,
        ChatPromptHelper,
        ChatPromptTemplate,
        MessagePlaceHolder,
        SystemMessage,
        ToolMessage,
        UserMessage,
        extract_fstring_variables,
        extract_jinja_variables,
        render_fstring_template,
        render_jinja_template,
    )
    from dapr_agents.types.message import (
        AssistantMessage as CoreAssistantMessage,
        BaseMessage as CoreBaseMessage,
        MessagePlaceHolder as CoreMessagePlaceHolder,
        SystemMessage as CoreSystemMessage,
        ToolMessage as CoreToolMessage,
        UserMessage as CoreUserMessage,
    )

    assert BaseMessage is CoreBaseMessage
    assert SystemMessage is CoreSystemMessage
    assert UserMessage is CoreUserMessage
    assert AssistantMessage is CoreAssistantMessage
    assert ToolMessage is CoreToolMessage
    assert MessagePlaceHolder is CoreMessagePlaceHolder
    assert callable(render_fstring_template)
    assert callable(extract_fstring_variables)
    assert callable(render_jinja_template)
    assert callable(extract_jinja_variables)
    assert ChatPromptTemplate is not None
    assert ChatPromptHelper is not None


def test_utils_fstring_reexports():
    """Verify that utils/fstring.py re-exports expected helpers."""
    from dapr_agents.prompt.engine import (
        extract_fstring_variables as engine_extract_fstring,
        render_fstring as engine_render_fstring_fn,
        render_fstring_template as engine_render_fstring,
    )
    from dapr_agents.prompt.utils.fstring import (
        extract_fstring_variables,
        render_fstring,
        render_fstring_template,
    )

    assert render_fstring is engine_render_fstring_fn
    assert render_fstring_template is engine_render_fstring
    assert extract_fstring_variables is engine_extract_fstring

    assert render_fstring_template("Hello {name}", name="world") == "Hello world"
    assert extract_fstring_variables("Hello {name}") == ["name"]


def test_utils_jinja_reexports():
    """Verify that utils/jinja.py re-exports expected helpers."""
    from dapr_agents.prompt.engine import (
        extract_jinja_variables as engine_extract_jinja,
        render_jinja as engine_render_jinja_fn,
        render_jinja_template as engine_render_jinja,
    )
    from dapr_agents.prompt.utils.jinja import (
        extract_jinja_variables,
        render_jinja,
        render_jinja_template,
    )

    assert render_jinja is engine_render_jinja_fn
    assert render_jinja_template is engine_render_jinja
    assert extract_jinja_variables is engine_extract_jinja

    assert render_jinja_template("Hello {{ name }}", name="world") == "Hello world"
    assert extract_jinja_variables("Hello {{ name }}") == ["name"]
