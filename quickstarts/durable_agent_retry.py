"""
Durable Agent Retry Configuration Quickstart.
Demonstrates fault-tolerance using WorkflowRetryPolicy.
"""

import asyncio
from dapr_agents.agents import DurableAgent
from dapr_agents.workflow import WorkflowRetryPolicy

# Define policy: 3 attempts with exponential backoff
retry_policy = WorkflowRetryPolicy(
    max_attempts=3, initial_interval_ms=1000, backoff_coefficient=2.0
)


class ResilientAgent(DurableAgent):
    def __init__(self):
        super().__init__(name="ResilientAgent", retry_policy=retry_policy)

    async def perform_task(self):
        print(f"[{self.name}] Running task with active retry policy.")
        return "Task Success"


async def main():
    agent = ResilientAgent()
    await agent.perform_task()


if __name__ == "__main__":
    asyncio.run(main())
