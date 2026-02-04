from dapr_agents import DurableAgent
from dapr_agents.agents.configs import AgentMemoryConfig
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.workflow.runners.agent import AgentRunner
from dotenv import load_dotenv

load_dotenv()
llm = DaprChatClient(component_name="llm-provider")


def main():
    expert_agent = DurableAgent(
        name="expert_agent",
        role="Technical Support Specialist",
        goal="Provide recommendations based on customer context and issue.",
        instructions=[
            "Provide a clear, actionable recommendation to resolve the issue.",
        ],
        llm=llm,
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="agent-memory",
                session_id=f"expert-agent-session",
            )
        ),
    )
    runner = AgentRunner()
    try:
        runner.serve(expert_agent, port=8002)
    finally:
        runner.shutdown(expert_agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down expert agent...")
        exit(0)
