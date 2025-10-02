# Multi-Agent Event-Driven Workflows with Distributed Tracing
This quickstart demonstrates how to create and orchestrate event-driven workflows with multiple autonomous agents using Dapr Agents, with comprehensive distributed tracing powered by Phoenix Arize. You'll learn how to set up agents as services, implement workflow orchestration, enable real-time agent collaboration through pub/sub messaging, and gain deep insights into agent interactions through distributed tracing.

## Prerequisites
- Python 3.10 (recommended)
- pip package manager
- OpenAI API key
- Dapr CLI and Docker installed
- PostgreSQL (for Phoenix Arize tracing backend)

## Environment Setup

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```



## Observability with Phoenix Arize

This section demonstrates how to add observability to your Dapr Agent workflows using Phoenix Arize for distributed tracing and monitoring. You'll learn how to set up Phoenix with PostgreSQL backend and instrument your workflow for comprehensive observability.

### Phoenix Server Setup

First, deploy Phoenix Arize server using Docker Compose with PostgreSQL backend for persistent storage.

#### Prerequisites

- Docker and Docker Compose installed on your system
- Verify Docker is running: `docker info`

#### Deploy Phoenix with PostgreSQL

1. Use the [docker-compose.yml](./docker-compose.yml) file provided to set up a Phoenix server locally with PostgreSQL backend.

2. Start the Phoenix server:

```bash
docker compose up --build
```

3. Verify Phoenix is running by navigating to [http://localhost:6006](http://localhost:6006)

#### Install Observability Dependencies

Install the updated requirements:

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file for your API keys:

```env
OPENAI_API_KEY=your_api_key_here
```

2. Configure the OpenAI component. You have two options:

   a. Directly update the `key` in [components/openai.yaml](components/openai.yaml):
   ```yaml
   metadata:
     - name: key
       value: "YOUR_OPENAI_API_KEY"
   ```

   b. Use environment variables (recommended):
   ```bash
   # Get the environment variables from the .env file:
   export $(grep -v '^#' ../../.env | xargs)

   # Create a temporary resources folder with resolved environment variables
   temp_resources_folder=$(../resolve_env_templates.py ./components)

   # Use the temporary folder when running your dapr commands
   dapr run -f dapr-random.yaml --resources-path $temp_resources_folder

   # Clean up the temporary folder when done
   rm -rf $temp_resources_folder
   ```

   Note: The temporary resources folder will be automatically deleted when the Dapr sidecar is stopped or when the computer is restarted.

3. Make sure Dapr is initialized on your system:

```bash
dapr init
```

4. The quickstart includes the necessary Dapr components in the `components` directory:

- `statestore.yaml`: Agent state configuration
- `pubsub.yaml`: Pub/Sub message bus configuration
- `workflowstate.yaml`: Workflow state configuration
- `openai.yaml`: OpenAI component configuration

## Project Structure

```
components/               # Dapr configuration files
├── statestore.yaml       # State store configuration
├── pubsub.yaml           # Pub/Sub configuration
└── workflowstate.yaml    # Workflow state configuration
services/                 # Directory for agent services
├── hobbit/               # First agent's service
│   └── app.py            # FastAPI app for hobbit
├── wizard/               # Second agent's service
│   └── app.py            # FastAPI app for wizard
├── elf/                  # Third agent's service
│   └── app.py            # FastAPI app for elf
└── workflow-random/      # Workflow orchestrator
    └── app.py            # Workflow service
└── workflow-roundrobin/  # Roundrobin orchestrator
    └── app.py            # Workflow service    
└── workflow-llm/         # LLM orchestrator
    └── app.py            # Workflow service        
dapr-random.yaml          # Multi-App Run Template using the random orchestrator
dapr-roundrobin.yaml      # Multi-App Run Template using the roundrobin orchestrator
dapr-llm.yaml             # Multi-App Run Template using the LLM orchestrator
```

## Examples

### Agent Service Implementation

Each agent is implemented as a separate service. Here's an example for the Hobbit agent:

```python
from dapr_agents import Agent, DurableAgent
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        hobbit_service = DurableAgent(
          name="Frodo",
          role="Hobbit",
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
          agents_registry_key="agents_registry",
          broadcast_topic_name="beacon_channel",
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

### Workflow Orchestrator Implementations

The workflow orchestrators manage the interaction between agents. Currently, Dapr Agents support three workflow types: RoundRobin, Random, and LLM-based. Here's an example for the Random workflow orchestrator (you can find examples for RoundRobin and LLM-based orchestrators in the project):

```python
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
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
            max_iterations=3
        ).as_service(port=8004)
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
- appID: HobbitApp
  appDirPath: ./services/hobbit/
  command: ["python3", "app.py"]

- appID: WizardApp
  appDirPath: ./services/wizard/
  command: ["python3", "app.py"]

- appID: ElfApp
  appDirPath: ./services/elf/
  command: ["python3", "app.py"]

- appID: WorkflowApp
  appDirPath: ./services/workflow-random/
  command: ["python3", "app.py"]
  appPort: 8004

- appID: ClientApp
  appDirPath: ./services/client/
  command: ["python3", "http_client.py"]
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
timeout_seconds: 120
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
timeout_seconds: 120
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
  - "workflow completed with status 'ORCHESTRATION_STATUS_COMPLETED' workflowName 'OrchestratorWorkflow'"
timeout_seconds: 200
output_match_mode: substring
background: false
-->
```bash
dapr run -f dapr-llm.yaml 
```
<!-- END_STEP -->
**Expected output:** The agents will engage in a conversation about getting to Mordor, with different agents contributing based on their character. Observe that in the logs, or checking the workflow state in [Redis Insights](https://dapr.github.io/dapr-agents/home/installation/#enable-redis-insights).

## Key Concepts
- **Agent Service**: Stateful service exposing an agent via API endpoints with independent lifecycle management
- **Pub/Sub Messaging**: Event-driven communication between agents for real-time collaboration
- **State Store**: Persistent storage for both agent registration and conversational memory
- **Actor Model**: Self-contained, sequential message processing via Dapr's Virtual Actor pattern
- **Workflow Orchestration**: Coordinating agent interactions in a durable and resilient manner

## Workflow Types
Dapr Agents supports multiple workflow orchestration patterns:

1. **RoundRobin**: Cycles through agents sequentially, ensuring equal task distribution
2. **Random**: Selects agents randomly for tasks, useful for load balancing and testing
3. **LLM-based**: Uses an LLM (default: OpenAI's models like gpt-4o) to intelligently select agents based on context and task requirements

## Monitoring and Observability
1. **Phoenix Arize**: Access comprehensive distributed tracing and agent interaction visualization at http://localhost:6006
   - View detailed agent-to-agent communication flows
   - Analyze conversation patterns and agent decision making
   - Monitor workflow performance and bottlenecks
   - Track LLM interactions and response times
2. **Console Logs**: Monitor real-time workflow execution and agent interactions
3. **Dapr Dashboard**: View components, configurations and service details at http://localhost:8080/
4. **Zipkin Tracing**: Access additional distributed tracing at http://localhost:9411/zipkin/
5. **Dapr Metrics**: Access agent performance metrics via (ex: HobbitApp) http://localhost:6001/metrics when configured

## Troubleshooting

1. **Service Startup**: If services fail to start, verify Dapr components configuration
2. **Communication Issues**: Check Redis connection and pub/sub setup
3. **Workflow Errors**: Check Zipkin traces for detailed request flows
4. **Port Conflicts**: If ports are already in use, check which port is already in use
5. **System Reset**: Clear Redis data through Redis Insights if needed

## Next Steps

After completing this quickstart, you can:

- Add more agents to the workflow
- Switch to another workflow orchestration pattern (RoundRobin, LLM-based)
- Extend agents with custom tools
- Deploy agents and Dapr to a Kubernetes cluster. For more information on read [Deploy Dapr on a Kubernetes cluster](https://docs.dapr.io/operations/hosting/kubernetes/kubernetes-deploy)
- Check out the [Cookbooks](../../cookbook/)