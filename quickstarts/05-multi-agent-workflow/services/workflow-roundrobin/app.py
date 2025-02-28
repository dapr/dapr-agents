from floki import RoundRobinWorkflowService
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        roundrobin_workflow_service = RoundRobinWorkflowService(
            name="Orchestrator",
            message_bus_name="messagepubsub",
            agents_state_store_name="agentstatestore",
            workflow_state_store_name="workflowstatestore",
            port=8004,
            daprGrpcPort=50004,
            max_iterations=2
        )

        await roundrobin_workflow_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())