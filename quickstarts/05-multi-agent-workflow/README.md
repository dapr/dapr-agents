# Multi-Agent Event-Driven Workflows

This quickstart demonstrates how to create and orchestrate event-driven workflows with multiple autonomous agents using Dapr Agents. You'll learn how to set up agents as services, implement workflow orchestration, and enable real-time agent collaboration through pub/sub messaging.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key
- Dapr CLI and Docker installed

## Environment Setup

```bash
# Create a virtual environment
python3.10 -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file for your API keys:

```env
OPENAI_API_KEY=your_api_key_here
```

2. Make sure Dapr is initialized on your system:

```bash
dapr init
```

3. The quickstart includes the necessary Dapr components in the `components` directory:

- `statestore.yaml`: Agent state configuration
- `pubsub.yaml`: Pub/Sub message bus configuration
- `workflowstate.yaml`: Workflow state configuration

## Project Structure

```
components/                # Dapr configuration files
├── statestore.yaml       # State store configuration
├── pubsub.yaml           # Pub/Sub configuration
└── workflowstate.yaml    # Workflow state configuration
services/                  # Directory for agent services
├── hobbit/               # First agent's service
│   └── app.py           # FastAPI app for hobbit
├── wizard/              # Second agent's service
│   └── app.py           # FastAPI app for wizard
├── elf/                 # Third agent's service
│   └── app.py           # FastAPI app for elf
└── workflow-roundrobin/ # Workflow orchestrator
    └── app.py           # Workflow service
dapr.yaml                # Multi-App Run Template
```

## Examples

### Agent Service Implementation

Each agent is implemented as a separate service. Here's an example for the Hobbit agent:

```python
# services/hobbit/app.py
from floki import Agent, AgentService
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        # Define Agent
        hobbit_agent = Agent(
            role="Hobbit",
            name="Frodo",
            goal="Take the ring to Mordor",
            instructions=["Speak like Frodo"]
        )
        
        # Expose Agent as a Service
        hobbit_service = AgentService(
            agent=hobbit_agent,
            message_bus_name="messagepubsub",
            agents_state_store_name="agentstatestore",
            port=8001,
            daprGrpcPort=50001
        )

        await hobbit_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

Similar implementations exist for the Wizard (Gandalf) and Elf (Legolas) agents.

### Workflow Orchestrator Implementation

The workflow orchestrator manages the interaction between agents:

```python
# services/workflow-roundrobin/app.py
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
```

### Running the Multi-Agent System

The project includes a `dapr.yaml` configuration for running all services:

```yaml
version: 1
common:
  resourcesPath: ./components
  logLevel: info
  appLogDestination: console
  daprdLogDestination: console

apps:
- appId: HobbitApp
  appDirPath: ./services/hobbit/
  appPort: 8001
  command: ["python3", "app.py"]
  daprGRPCPort: 50001

- appId: WizardApp
  appDirPath: ./services/wizard/
  appPort: 8002
  command: ["python3", "app.py"]
  daprGRPCPort: 50002

- appId: ElfApp
  appDirPath: ./services/elf/
  appPort: 8003
  command: ["python3", "app.py"]
  daprGRPCPort: 50003

- appId: WorkflowApp
  appDirPath: ./services/workflow-roundrobin/
  appPort: 8004
  command: ["python3", "app.py"]
  daprGRPCPort: 50004
```

Start all services using the Dapr CLI:

```bash
dapr run -f .
```

Once all services are running, you can start the workflow via HTTP:

```bash
curl -i -X POST http://localhost:8004/RunWorkflow \
    -H "Content-Type: application/json" \
    -d '{"message": "How to get to Mordor? Let's all help!"}'
```

**Expected output:** The agents will engage in a conversation about getting to Mordor, with each agent taking turns to contribute based on their character.

## Key Concepts

- **Agent Service**: Stateful service exposing an agent via API endpoints
- **Pub/Sub Messaging**: Event-driven communication between agents
- **Actor Model**: Stateful agent representation using Dapr Actors
- **Workflow Orchestration**: Coordinating agent interactions
- **Distributed System**: Multiple services working together

## Workflow Types

Dapr Agents supports multiple workflow orchestration patterns:

1. **RoundRobin**: Cycles through agents sequentially
2. **Random**: Selects agents randomly for tasks
3. **LLM-based**: Uses GPT-4o to intelligently select agents based on context

## Dapr Integration

This quickstart showcases several Dapr building blocks:

- **Pub/Sub**: Agent communication via Redis message bus
- **State Management**: Persistence of agent and workflow states
- **Service Invocation**: Direct HTTP communication between services
- **Actors**: Stateful agent representation

## Monitoring and Observability

1. **Console Logs**: Monitor real-time workflow execution
2. **Redis Insights**: View message bus and state data at http://localhost:5540/
3. **Zipkin Tracing**: Access distributed tracing at http://localhost:9411/zipkin/

## Troubleshooting

1. **Service Startup**: If services fail to start, verify Dapr components configuration
2. **Communication Issues**: Check Redis connection and pub/sub setup
3. **Workflow Errors**: Check Zipkin traces for detailed request flows
4. **System Reset**: Clear Redis data through Redis Insights if needed

## Next Steps

After completing this quickstart, you can:

- Add more agents to the workflow
- Switch to another workflow orchestration pattern (Random, LLM-based)
- Extend agents with custom tools
- Deploy to a Kubernetes cluster using Dapr