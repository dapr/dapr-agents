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
└── workflow-random/      # Workflow orchestrator
    └── app.py           # Workflow service
└── workflow-roundrobin/  # Roundrobin orchestrator
    └── app.py           # Workflow service    
└── workflow-llm/         # LLM orchestrator
    └── app.py           # Workflow service        
dapr-random.yaml         # Multi-App Run Template using the random orchestrator
dapr-roundrobin.yaml     # Multi-App Run Template using the roundrobin orchestrator
dapr-llm.yaml            # Multi-App Run Template using the LLM orchestrator
```

## Examples

### Agent Service Implementation

Each agent is implemented as a separate service. Here's an example for the Hobbit agent:

```python
# services/hobbit/app.py
from dapr_agents import Agent, AgentActorService, AssistantAgent
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        hobbit_service = AssistantAgent(name="Frodo", role="Hobbit",
                                      goal="Carry the One Ring to Mount Doom, resisting its corruptive power while navigating danger and uncertainty.",
                                      instructions=[
                                          "Speak like Frodo, with humility, determination, and a growing sense of resolve.",
                                          "Endure hardships and temptations, staying true to the mission even when faced with doubt.",
                                          "Seek guidance and trust allies, but bear the ultimate burden alone when necessary.",
                                          "Move carefully through enemy-infested lands, avoiding unnecessary risks.",
                                          "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task."],
                                      message_bus_name="messagepubsub",
                                      state_store_name="workflowstatestore",
                                      state_key="workflow_state",
                                      agents_registry_store_name="agentstatestore",
                                      agents_registry_key="agents_registry", service_port=8001,
                                      daprGrpcPort=50001)

        await hobbit_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
```

Similar implementations exist for the Wizard (Gandalf) and Elf (Legolas) agents.

### Workflow Orchestrator Implementations

The workflow orchestrators manage the interaction between agents. Currently, Dapr Agents support three workflow types: RoundRobin, Random, and LLM-based. Here's an example for the Random workflow orchestrator (you can find examples for RoundRobin and LLM-based orchestrators in the project):

```python
# services/workflow-random/app.py
from dapr_agents import RandomOrchestrator
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        random_workflow_service = RandomOrchestrator(
            name="RandomOrchestrator",
            message_bus_name="messagepubsub",
            state_store_name="agenticworkflowstate",
            state_key="workflow_state",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
            service_port=8009,
            daprGrpcPort=50009,
            max_iterations=3
        )
        await random_workflow_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())
```

### Running the Multi-Agent System

The project includes three dapr multi-app run configuration files (`dapr-random.yaml`, `dapr-roundrobin.yaml` and `dapr-llm.yaml` ) for running all services and an additional Client application for interacting with the agents:

Example: `dapr-random.yaml`
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
  appDirPath: ./services/workflow-random/
  appPort: 8004
  command: ["python3", "app.py"]
  daprGRPCPort: 50004

- appId: ClientApp
  appDirPath: ./services/client/
  command: ["python3", "client.py"]
  daprGRPCPort: 50011
```

Start all services using the Dapr CLI:

<!-- STEP
name: Run text completion example
match_order: none
expected_stdout_lines:
  - "Workflow started successfully!"
  - "user:"
  - "How to get to Mordor? We all need to help!"
  - "assistant:"
  - "user:"
  - "assistant:"
  - "workflow completed with status 'ORCHESTRATION_STATUS_COMPLETED' workflowName 'RandomWorkflow'"
timeout_seconds: 20
output_match_mode: substring
background: false
sleep: 5
-->
```bash
dapr run -f dapr-random.yaml 
```
<!-- END_STEP -->

You will see the agents engaging in a conversation about getting to Mordor, with different agents contributing based on their character.

You can also run the RoundRobin and LLM-based orchestrators using `dapr-roundrobin.yaml` and `dapr-llm.yaml` respectively:

<!-- STEP
name: Run text completion example
match_order: none
expected_stdout_lines:
  - "Workflow started successfully!"
  - "user:"
  - "How to get to Mordor? We all need to help!"
  - "assistant:"
  - "user:"
  - "assistant:"
  - "workflow completed with status 'ORCHESTRATION_STATUS_COMPLETED' workflowName 'RoundRobinWorkflow'"
timeout_seconds: 20
output_match_mode: substring
background: false
sleep: 5
-->
```bash
dapr run -f dapr-roundrobin.yaml 
```
<!-- END_STEP -->

<!-- STEP
name: Run text completion example
match_order: none
expected_stdout_lines:
  - "Workflow started successfully!"
  - "user:"
  - "How to get to Mordor? We all need to help!"
  - "assistant:"
  - "user:"
  - "assistant:"
  - "workflow completed with status 'ORCHESTRATION_STATUS_COMPLETED' workflowName 'LLMWorkflow'"
timeout_seconds: 20
output_match_mode: substring
background: false
-->
```bash
dapr run -f dapr-llm.yaml 
```
<!-- END_STEP -->
**Expected output:** The agents will engage in a conversation about getting to Mordor, with different agents contributing based on their character.

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