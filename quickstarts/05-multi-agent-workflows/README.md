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

   a. Directly update the `key` in [components/openai.yaml](components/openai.yaml), and remove the secretKeyRef:
   ```yaml
   metadata:
     - name: key
       value: "YOUR_OPENAI_API_KEY"
   ```

   b. Use environment variables (recommended):
   ```bash
   # Get the environment variables from the .env file exported to be use
   export $(grep -v '^#' ../../.env | xargs)
   ```

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

Each agent is implemented as a separate service. Here's an example for the Hobbit agent in [app.py](./services/hobbit/app.py).
Similar implementations exist for the Wizard (Gandalf) and Elf (Legolas) agents.

### Workflow Orchestrator Implementations

The workflow orchestrators manage the interaction between agents. Currently, Dapr Agents support three workflow types: RoundRobin, Random, and LLM-based. Here's an example for the Random workflow orchestrator (you can find examples for RoundRobin and LLM-based orchestrators in the project).

### Running the Multi-Agent System

The project includes three dapr multi-app run configuration files (`dapr-random.yaml`, `dapr-roundrobin.yaml` and `dapr-llm.yaml` ) for running all services and an additional Client application for interacting with the agents.

Start all services using the Dapr CLI:
```bash
dapr run -f dapr-random.yaml 
```


You will see the agents engaging in a conversation about getting to Mordor, with different agents contributing based on their character.

You can also run the RoundRobin and LLM-based orchestrators using `dapr-roundrobin.yaml` and `dapr-llm.yaml` respectively:
```bash
dapr run -f dapr-roundrobin.yaml 
```
```bash
dapr run -f dapr-llm.yaml 
```
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