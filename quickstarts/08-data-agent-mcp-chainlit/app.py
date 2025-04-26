import chainlit as cl
from dapr_agents import Agent
from dapr_agents.tool.mcp.client import MCPClient
from dotenv import load_dotenv
from get_schema import get_table_schema_as_dict

load_dotenv()

instructions = [
    "You are an assistant designed to translate human readable text to postgresql queries. "
    "Your primary goal is to provide accurate SQL queries based on the user request. "
    "If something is unclear or you need more context, ask thoughtful clarifying questions."
]

agent = {}

table_info = {}


@cl.on_chat_start
async def start():
    client = MCPClient()
    await client.connect_sse(
        server_name="local",  # Unique name you assign to this server
        url="http://0.0.0.0:8000/sse",  # MCP SSE endpoint
        headers=None,  # Optional HTTP headers if needed
    )

    # See what tools were loaded
    tools = client.get_all_tools()

    global agent
    agent = Agent(
        name="SQL",
        role="Database Expert",
        instructions=instructions,
        tools=tools,
    )

    global table_info
    table_info = get_table_schema_as_dict()

    if table_info:
        await cl.Message(
            content="Database connection successful. Ask me anything."
        ).send()
    else:
        await cl.Message(content="Database connection failed.").send()


@cl.on_message
async def main(message: cl.Message):
    # generate the result set and pass back to the user
    prompt = create_prompt_for_llm(table_info, message.content)
    result = await agent.run(prompt)

    await cl.Message(
        content=result,
    ).send()

    result_set = await agent.run(
        "Execute the following sql query and always return a table format unless instructed otherwise. If the user asks a question regarding the data, return the result and formalize an answer based on inspecting the data: "
        + result
    )
    await cl.Message(
        content=result_set,
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
