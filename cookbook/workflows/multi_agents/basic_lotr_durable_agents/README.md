# Multi-Agent LOTR: Durable Agents

This guide shows you how to set up and run an event-driven agentic workflow using Dapr Agents. By leveraging [Dapr Pub/Sub](https://docs.dapr.io/developing-applications/building-blocks/pubsub/pubsub-overview/) and FastAPI, `Dapr Agents` enables agents to collaborate dynamically in decentralized systems.

## Prerequisites

Before you start, ensure you have the following:

* [Dapr Agents environment set up](https://github.com/dapr/dapr-agents), including Python 3.8 or higher and Dapr CLI.
* Docker installed and running.
* Basic understanding of microservices and event-driven architecture.

## Project Structure

The project is organized into multiple services, each representing an agent or a workflow. Here’s the layout:

```
├── components/              # Dapr configuration files
│   ├── statestore.yaml      # State store configuration
│   ├── pubsub.yaml          # Pub/Sub configuration
├── services/                # Directory for services
│   ├── hobbit/              # Hobbit Agent Service
│   │   └── app.py           # FastAPI app for Hobbit
│   ├── wizard/              # Wizard Agent Service
│   │   └── app.py           # FastAPI app for Wizard
│   ├── elf/                 # Elf Agent Service
│   │   └── app.py           # FastAPI app for Elf
│   ├── workflow-roundrobin/ # Workflow Service
│       └── app.py           # Orchestrator Workflow
├── dapr.yaml                # Multi-App Run Template
```

## Running the Services

0. Set Up Environment Variables: Create an `.env` file to securely store your API keys and other sensitive information. For example:

```
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

1. Multi-App Run: Use the dapr.yaml file to start all services simultaneously:

```bash
dapr run -f .
```

2. Verify console Logs: Each service outputs logs to confirm successful initialization.


3. Verify Redis entries: Access the Redis Insight interface at `http://localhost:5540/`

## Starting the Workflow

Send an HTTP POST request to the workflow service to start the workflow. Use curl or any API client:

```bash
curl -i -X POST http://localhost:8004/start-workflow \
    -H "Content-Type: application/json" \
    -d '{"task": "Lets solve the riddle to open the Doors of Durin and enter Moria."}'
```

```
HTTP/1.1 200 OK
date: Thu, 05 Dec 2024 07:46:19 GMT
server: uvicorn
content-length: 104
content-type: application/json

{"message":"Workflow initiated successfully.","workflow_instance_id":"422ab3c3f58f4221a36b36c05fefb99b"}
```

The workflow will trigger agents in a round-robin sequence to process the message.

## Monitoring Workflow Execution

1. Check console logs to trace activities in the workflow.

2. Verify Redis entries: Access the Redis Insight interface at `http://localhost:5540/`

3. As mentioned earlier, when we ran dapr init, Dapr initialized, a `Zipkin` container instance, used for observability and tracing. Open `http://localhost:9411/zipkin/` in your browser to view traces > Find a Trace > Run Query.

4. Select the trace entry with multiple spans labeled `<workflow name>: /taskhubsidecarservice/startinstance.`. When you open this entry, you’ll see details about how each task or activity in the workflow was executed. If any task failed, the error will also be visible here.

5. Check console logs to validate if workflow was executed successfuly.

### Reset Redis Database

1. Access the Redis Insight interface at `http://localhost:5540/`
2. In the search bar type `*` to select all items in the database.
3. Click on `Bulk Actions` > `Delete` > `Delete`