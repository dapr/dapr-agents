<!--
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# dapr-agents-ext-drasi

[Drasi](https://drasi.io/) extension for Dapr Agents, enabling resilient, scalable, and business event-driven AI agents through Drasi's change notification capabilities.

## Features

- Directly trigger agents from Drasi change events via Dapr pub/sub

## Getting Started

### Prerequisites

This extension is installed as a [PEP 771](https://peps.python.org/pep-0771/) extra on the core `dapr-agents` package; prerequisite steps can be found in the root [README](https://github.com/dapr/dapr-agents/blob/main/README.md).

Drasi also needs to be deployed with a pub/sub event producer. See the [Drasi documentation](https://drasi.io/) for supported deployment models and the currently available connectors for each deployment model.

### Installation

```bash
uv add dapr-agents[drasi]
```

### Example Usage

This minimal example demonstrates how to subscribe an agent to a Drasi query. Replace the placeholders with your desired values.

```python
from __future__ import annotations

import asyncio

from dapr_agents import AgentRunner, DurableAgent, DaprChatClient
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    AgentPubSubConfig,
)
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.utils.core import wait_for_shutdown

from dapr_agents.ext.drasi import drasi_trigger


async def main() -> None:
    # Create the agent
    agent = DurableAgent(
        name="<AGENT_NAME>",
        role="<AGENT_ROLE>",
        goal="<AGENT_GOAL>",
        instructions="<AGENT_INSTRUCTIONS>",
        llm=DaprChatClient(component_name="<YOUR_DAPR_CONVERSATION_COMPONENT_NAME>"),
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(store_name="<YOUR_DAPR_STATE_STORE_COMPONENT_NAME>"),
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="<YOUR_DAPR_STATE_STORE_COMPONENT_NAME>"),
        ),
        execution=AgentExecutionConfig(max_iterations=1),
    )

    # Register Drasi query subscriptions
    drasi_trigger(
        agent,
        query_id="<YOUR_DRASI_QUERY_ID>",
        pubsub="<YOUR_DAPR_PUBSUB_COMPONENT_NAME>",
        topic="<YOUR_TOPIC_NAME>",
        task_mapper=lambda event, ctx: TriggerAction(task="<AGENT_TASK_MESSAGE>"),
    )
    
    # Host the agent, wire Drasi query subscriptions and begin listening for Drasi events
    runner = AgentRunner()
    try:
        runner.subscribe(agent)
        await wait_for_shutdown()
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

```
