import os
from dapr_agents import AssistantAgent
from dapr_agents.llm.openai.chat import OpenAIChatClient
from dapr_agents.tool.mcp import MCPClient
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    
    llm = OpenAIChatClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL")
    )

    try:
      client = MCPClient()

      await client.connect_stdio(
          server_name="GitHubMCPServer",
          env= {
              "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
          },
          command="./github-mcp-server",
          args=["stdio"]
      )

      # See what tools were loaded
      tools = client.get_all_tools()
      print("🔧 Tools:", [t.name for t in tools])

    except Exception as e:
        print(f"Error initializing tools from GitHub MCP server: {e}")

    try:
        github_service = AssistantAgent(
            name="Octocat",
            role="GitHub Issues Assistant",
            goal="You are an AI developer assistant using the GitHub MCP server to get an understanding of the context of Issues in order to properly articulate the requirements for implementating the ask.",
            instructions=[
                "You understand that list_issues has the following parameters you need to fill out: " \
                  "owner: Repository owner (string, required) " \
                  "repo: Repository name (string, required) " \
                  "state: Filter by state ('open', 'closed', 'all') (string, optional) " \
                  "labels: Labels to filter by (string[], optional) " \
                  "sort: Sort by ('created', 'updated', 'comments') (string, optional) " \
                  "direction: Sort direction ('asc', 'desc') (string, optional) " \
                  "since: Filter by date (ISO 8601 timestamp) (string, optional) " \
                  "page: Page number (number, optional) " \
                  "perPage: Results per page (number, optional).",
                "You will use 'created' for sorting and 'asc' for direction.",
                "You will use last month as default input to 'since' when getting issues.",
                "You will use 'open' as default input to 'state' when getting issues.",
                "You will use 1 as your default input to 'page' and 25 as default to 'perPage' when getting issues.",
                "When using list_issues you will always set values for: direction, owner, page, perPage, repo, state, sort, since.",
                "You understand that get_issue has the following parameters you need to fill out: " \
                  "owner: Repository owner (string, required) " \
                  "repo: Repository name (string, required) " \
                  "issue_number: Issue number (number, required).",
                "You understand that get_issue_comments has the following parameters you need to fill out: " \
                  "owner: Repository owner (string, required) " \
                  "repo: Repository name (string, required) " \
                  "issue_number: Issue number (number, required). " \
                  "page: Page number (number, optional) " \
                  "perPage: Results per page (number, optional).",
                "When using get_issue_comments you will always set values for: issue_number, owner, page, perPage, repo",
                "You will use 1 as your default input to 'page' and 25 as default to 'perPage' when getting issue comments.",
                "You will use branch main or master as the default input to branch or from_branch.",
                "You are an expert in the GitHub API.",
                "You understand the context of the issue and can provide a summary.",
                "You write clear and concise requirements that are easily testable.",
                "You will outline, in bullet points, the requirements for implementing the ask in your response.",
                "You will not try to implement the requirements, you will only outline them.",
            ],
            tools=tools,
            llm=llm,
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
        )

        await github_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
