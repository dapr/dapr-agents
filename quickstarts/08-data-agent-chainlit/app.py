import chainlit as cl
from dapr_agents import Agent
from dapr_agents.tool.mcp.client import MCPClient
from dotenv import load_dotenv
import psycopg
import os

load_dotenv()

instructions = [
    "You are an assistant designed to translate human readable text to postgresql queries. "
    "Your primary goal is to provide accurate SQL queries based on the user request."
    "If something is unclear or you need more context, ask thoughtful clarifying questions. "
]

agent = {}

table_info = {}

@cl.on_chat_start
async def start():
    client = MCPClient()
    await client.connect_sse(
        server_name="local",  # Unique name you assign to this server
        url="http://0.0.0.0:8000/sse",  # MCP SSE endpoint
        headers=None  # Optional HTTP headers if needed
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
        await cl.Message(
            content="Database connection failed."
        ).send()
        

@cl.on_message
async def main(message: cl.Message):
    # generate the result set and pass back to the user
    prompt = create_prompt_for_llm(table_info, message.content)
    result = await agent.run(prompt)

    await cl.Message(
        content=result,
    ).send()

    result_set = await agent.run("Execute the following sql query: " + result)
    await cl.Message(
        content=result_set,
    ).send()

def get_table_schema_as_dict():
    conn_params = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD")
    }

    schema_data = {}

    try:
        with psycopg.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                # Get all table names with their schemas
                cur.execute("""
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY table_schema, table_name;
                """)
                tables = cur.fetchall()

                for schema, table in tables:
                    schema_data[f'{schema}.{table}'] = []
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position;
                    """, (schema, table))
                    columns = cur.fetchall()

                    for col in columns:
                        schema_data[f'{schema}.{table}'].append({
                            "column_name": col[0],
                            "data_type": col[1],
                            "is_nullable": col[2],
                            "column_default": col[3]
                        })
                        
                    return schema_data

    except Exception as e:
        return False

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
