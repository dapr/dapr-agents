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

"""Unit tests for ChatPromptTemplate.format_prompt() method."""

import pytest
from dapr_agents.prompt.chat import ChatPromptTemplate
from dapr_agents.types.message import (
    UserMessage,
    SystemMessage,
    AssistantMessage,
    MessagePlaceHolder,
)


class TestChatPromptTemplateFormatPrompt:
    """Tests for ChatPromptTemplate.format_prompt() method."""

    def test_format_prompt_with_tuple_messages(self):
        """Test format_prompt with simple tuple messages (role, content)."""
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("user", "Hello!"),
            ]
        )
        result = template.format_prompt()

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello!"

    def test_format_prompt_with_dict_messages(self):
        """Test format_prompt with dictionary messages."""
        template = ChatPromptTemplate.from_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        )
        result = template.format_prompt()

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "What is 2+2?"

    def test_format_prompt_with_base_message_objects(self):
        """Test format_prompt with BaseMessage instances."""
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a bot."),
                UserMessage(content="Test message"),
            ]
        )
        result = template.format_prompt()

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a bot."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Test message"

    def test_format_prompt_with_f_string_variables(self):
        """Test format_prompt with f-string template variable substitution."""
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a {role}"),
                ("user", "My name is {name}"),
            ],
            template_format="f-string",
        )
        result = template.format_prompt(role="assistant", name="Alice")

        assert len(result) == 2
        assert result[0]["content"] == "You are a assistant"
        assert result[1]["content"] == "My name is Alice"

    def test_format_prompt_with_prefilled_variables(self):
        """Test format_prompt respects pre_filled_variables."""
        template = ChatPromptTemplate.from_messages(
            [
                ("user", "Hello {name}!"),
            ],
            template_format="f-string",
        )

        # Pre-fill the name variable
        prefilled_template = template.pre_fill_variables(name="Bob")
        result = prefilled_template.format_prompt()

        assert len(result) == 1
        assert result[0]["content"] == "Hello Bob!"

    def test_format_prompt_raises_on_undeclared_variables(self):
        """Test that format_prompt raises ValueError for undeclared variables."""
        template = ChatPromptTemplate.from_messages(
            [
                ("user", "Hello"),
            ]
        )

        # Passing undeclared variable should raise
        with pytest.raises(ValueError, match="Undeclared variables were passed"):
            template.format_prompt(unexpected_var="value")

    def test_format_prompt_logs_missing_variables(self, caplog):
        """Test that format_prompt logs info about missing variables before raising."""
        template = ChatPromptTemplate.from_messages(
            [
                ("user", "Name: {name}"),
            ],
            template_format="f-string",
        )

        # Missing the required variable — logs info then raises KeyError during formatting
        with caplog.at_level("INFO"):
            with pytest.raises(KeyError):
                template.format_prompt()

        assert "Some input variables were not provided" in caplog.text
        assert "name" in caplog.text

    def test_format_prompt_with_message_placeholder(self):
        """Test format_prompt with MessagePlaceHolder for dynamic message lists."""
        template = ChatPromptTemplate(
            input_variables=["history"],
            messages=[
                ("system", "You are helpful."),
                MessagePlaceHolder(variable_name="history"),
                ("user", "What happened before?"),
            ],
        )

        # Provide dynamic messages via placeholder
        history_messages = [
            UserMessage(content="First question"),
            AssistantMessage(content="First answer"),
        ]
        result = template.format_prompt(history=history_messages)

        # Should have: system, history (2 messages), user
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "First question"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"
        assert result[3]["content"] == "What happened before?"

    def test_format_prompt_with_empty_placeholder(self, caplog):
        """Test format_prompt when MessagePlaceHolder variable is not provided."""
        template = ChatPromptTemplate(
            input_variables=[],
            messages=[
                ("system", "Start"),
                MessagePlaceHolder(variable_name="history"),
                ("user", "End"),
            ],
        )

        with caplog.at_level("INFO"):
            result = template.format_prompt()

        # Should skip the placeholder and only have system and user
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "MessagePlaceHolder variable 'history' was not provided" in caplog.text

    def test_format_prompt_with_mixed_message_types(self):
        """Test format_prompt with mixed message types (tuple, dict, BaseMessage)."""
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "System prompt"),
                {"role": "user", "content": "Dict message"},
                SystemMessage(content="Another system message"),
            ]
        )
        result = template.format_prompt()

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System prompt"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "system"
        assert result[2]["content"] == "Another system message"

    def test_format_prompt_preserves_message_order(self):
        """Test that format_prompt preserves message order."""
        messages = [
            ("system", "First"),
            ("user", "Second"),
            ("assistant", "Third"),
            ("user", "Fourth"),
        ]
        template = ChatPromptTemplate.from_messages(messages)
        result = template.format_prompt()

        expected_contents = ["First", "Second", "Third", "Fourth"]
        assert [msg["content"] for msg in result] == expected_contents

    def test_format_prompt_with_template_format_override(self):
        """Test that template_format parameter overrides template default."""
        template = ChatPromptTemplate.from_messages(
            [
                ("user", "Hello {name}"),
            ],
            template_format="jinja2",  # Default format differs from override
        )

        # Should use the overridden template_format, not the default
        result = template.format_prompt(template_format="f-string", name="Alice")
        assert result[0]["content"] == "Hello Alice"

    def test_format_prompt_handles_multiple_variables_in_message(self):
        """Test format_prompt with multiple variables in same message."""
        template = ChatPromptTemplate.from_messages(
            [
                ("user", "{greeting} {name}, you are {age} years old"),
            ],
            template_format="f-string",
        )
        result = template.format_prompt(greeting="Hi", name="Charlie", age="25")

        assert result[0]["content"] == "Hi Charlie, you are 25 years old"

    def test_format_prompt_returns_dict_representation(self):
        """Test that format_prompt returns messages as dictionaries (not BaseMessage objects)."""
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="Test"),
            ]
        )
        result = template.format_prompt()

        # Should be dict, not BaseMessage instance
        assert isinstance(result[0], dict)
        assert "role" in result[0]
        assert "content" in result[0]
