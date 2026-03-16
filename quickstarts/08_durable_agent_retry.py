import asyncio
from dapr_agents.agents import DurableAgent
from dapr_agents.workflow import WorkflowRetryPolicy


# Define policy: 3 attempts with exponential backoff
retry_policy = WorkflowRetryPolicy(
    max_attempts=3,
    initial_interval_ms=1000,
    backoff_coefficient=2.0,
)


# Instantiate the agent directly using the retry policy
agent = DurableAgent(
    name="ResilientAgent",
    retry_policy=retry_policy,
)


async def perform_task():
    """Simulates a task running with the active retry policy."""
    print(f"[{agent.name}] Running task with active retry policy.")
    # Here you would typically use agent.run() or a similar execution method
    return "Task Success"


async def main():
    result = await perform_task()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
