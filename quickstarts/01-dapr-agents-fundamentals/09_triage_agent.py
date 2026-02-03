from dapr_agents import DurableAgent, tool
from dapr_agents.agents.configs import AgentMemoryConfig
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.workflow.runners.agent import AgentRunner
from dotenv import load_dotenv

load_dotenv()
llm = DaprChatClient(component_name="llm-provider")


@tool
def get_customer_info(customer_name: str) -> str:
    """Get customer information by name. Returns a simple text description."""
    customers = {
        "alice": "Customer: Alice, Premium Plan, 5 active services",
        "bob": "Customer: Bob, Standard Plan, 2 active services",
        "charlie": "Customer: Charlie, Basic Plan, 1 active service",
    }
    return customers.get(
        customer_name.lower(),
        f"Customer: {customer_name}, Standard Plan, 1 active service",
    )


def main():
    triage_agent = DurableAgent(
        name="triage_agent",
        role="Customer Support Triage Assistant",
        goal="Gather customer information and prepare a triage summary.",
        instructions=[
            "Use the tool to get customer information, then combine it with the issue description.",
        ],
        llm=llm,
        tools=[get_customer_info],
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="agent-memory",
                session_id=f"triage-agent-session",
            )
        ),
    )
    runner = AgentRunner()
    try:
        runner.serve(triage_agent, port=8001)
    finally:
        runner.shutdown(triage_agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down triage agent...")
        exit(0)
