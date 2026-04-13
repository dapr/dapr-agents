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

import json
from typing import Optional

import chainlit as cl
from dotenv import load_dotenv

from dapr_agents import DurableAgent, AgentRunner
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.tool.mcp.client import MCPClient
from get_schema import get_table_schema_as_dict

load_dotenv()

instructions = [
    "You are an assistant designed to translate human readable text to postgresql queries. "
    "Your primary goal is to provide accurate SQL queries based on the user request. "
    "If something is unclear or you need more context, ask thoughtful clarifying questions."
]

# Initialized once on first chat session
_agent: Optional[DurableAgent] = None
_runner: Optional[AgentRunner] = None
_mcp_client: Optional[MCPClient] = None
_table_info: dict = {}
_initialized = False


def _extract_content(result: str | None) -> str:
    """Extract the content field from a serialized workflow result."""
    if not result:
        return ""
    try:
        parsed = json.loads(result)
        return parsed.get("content", result)
    except (json.JSONDecodeError, AttributeError):
        return result


async def _ensure_initialized() -> tuple[DurableAgent, AgentRunner, dict]:
    """Initialize agent, runner, MCP client, and schema exactly once."""
    global _agent, _runner, _mcp_client, _table_info, _initialized

    if _initialized:
        return _agent, _runner, _table_info

    # Connect to MCP server and load tools.
    _mcp_client = MCPClient()
    await _mcp_client.connect_sse(
        server_name="local",
        url="http://0.0.0.0:8000/sse",
    )
    tools = _mcp_client.get_all_tools()

    _agent = DurableAgent(
        name="SQL",
        role="Database Expert",
        instructions=instructions,
        tools=tools,
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(store_name="conversationstore"),
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflowstatestore"),
        ),
        registry=AgentRegistryConfig(
            store=StateStoreService(store_name="registrystatestore"),
            team_name="default",
        ),
        execution=AgentExecutionConfig(max_iterations=3),
    )

    _runner = AgentRunner()
    _runner.workflow(_agent)

    _table_info = get_table_schema_as_dict()
    _initialized = True

    return _agent, _runner, _table_info


@cl.on_chat_start
async def on_chat_start():
    _, _, table_info = await _ensure_initialized()

    if table_info:
        await cl.Message(
            content="Database connection successful. Ask me anything."
        ).send()
    else:
        await cl.Message(content="Database connection failed.").send()


@cl.on_chat_end
async def on_chat_end():
    global _runner, _agent, _initialized
    if _runner and _agent:
        _runner.shutdown(_agent)
    _initialized = False


@cl.on_message
async def on_message(message: cl.Message):
    agent, runner, table_info = await _ensure_initialized()

    if not table_info:
        await cl.Message(
            content="Agent is not ready yet. Please try again in a moment."
        ).send()
        return

    prompt = create_prompt_for_llm(table_info, message.content)

    try:
        result = await runner.run(agent, {"task": prompt})
        await cl.Message(content=_extract_content(result)).send()
    except Exception as exc:
        await cl.Message(
            content=f"Sorry, something went wrong while processing your request: {exc}"
        ).send()


def create_prompt_for_llm(schema_data, user_question):
    prompt = "Here is the schema for the tables in the database:\n\n"

    # Add schema information to the prompt
    for table, columns in schema_data.items():
        prompt += f"Table {table}:\n"
        for col in columns:
            prompt += f"  - {col['column_name']} ({col['data_type']}), Nullable: {col['is_nullable']}, Default: {col['column_default']}\n"

    # Add the user's question for context
    prompt += f"\nUser's question: {user_question}\n"
    prompt += "Generate and execute the postgres SQL query to answer the user's question. Return the results in a table format and provide analysis."

    return prompt
