from typing import Optional

import chainlit as cl
from dotenv import load_dotenv

from dapr_agents import Agent
from dapr_agents.agents.configs import AgentMemoryConfig
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.tool.mcp.client import MCPClient
from dapr_agents.types import AssistantMessage
from get_schema import get_table_schema_as_dict

load_dotenv()

instructions = [
    "You are an assistant designed to translate human readable text to postgresql queries. "
    "Your primary goal is to provide accurate SQL queries based on the user request. "
    "If something is unclear or you need more context, ask thoughtful clarifying questions."
]

agent: Optional[Agent] = None
table_info = {}
mcp_client: Optional[MCPClient] = None


async def _load_mcp_tools() -> list:
    client = MCPClient(persistent_connections=True)
    await client.connect_sse(
        server_name="local",
        url="http://0.0.0.0:8000/sse",
    )
    return client, client.get_all_tools()


@cl.on_chat_start
async def on_chat_start():
    global agent, table_info, mcp_client

    client, tools = await _load_mcp_tools()
    mcp_client = client

    agent = Agent(
        name="SQL",
        role="Database Expert",
        instructions=instructions,
        tools=tools,
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="conversationstore",
                session_id="sql-agent",
            )
        ),
    )

    table_info = get_table_schema_as_dict()
    if table_info:
        await cl.Message(
            content="Database connection successful. Ask me anything."
        ).send()
    else:
        await cl.Message(content="Database connection failed.").send()


@cl.on_chat_end
async def on_chat_end():
    global mcp_client
    if mcp_client:
        try:
            await mcp_client.close()
        except RuntimeError as exc:
            # best-effort; Chainlit may shut down the loop concurrently
            if "Attempted to exit cancel scope" not in str(exc):
                raise
        finally:
            mcp_client = None


@cl.on_message
async def on_message(message: cl.Message):
    if agent is None or not table_info:
        await cl.Message(
            content="Agent is not ready yet. Please try again in a moment."
        ).send()
        return

    prompt = create_prompt_for_llm(table_info, message.content)

    try:
        sql_response: AssistantMessage = await agent.run(prompt)
        await cl.Message(content=sql_response.content).send()

        result_prompt = (
            "Execute the following sql query and always return a table format unless instructed otherwise. "
            "If the user asks a question regarding the data, return the result and formalize an answer "
            "based on inspecting the data: " + (sql_response.content or "")
        )
        result_set: AssistantMessage = await agent.run(result_prompt)
        await cl.Message(content=result_set.content).send()
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
    prompt += "Generate the postgres SQL query to answer the user's question. Return only the query string and nothing else."

    return prompt
