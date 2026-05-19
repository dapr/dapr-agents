#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""DurableAgent factory for the expert-agent example."""

from dapr_agents import DurableAgent, Hooks, OpenAIChatClient
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService

from hooks import enrich_with_tavily


INSTRUCTIONS = [
    "You are an expert assistant with access to live web context.",
    "Use the 'Fresh web context (Tavily)' system message that precedes the "
    "user question to ground your answer in current information.",
    "Be concise and cite which source(s) you used when relevant.",
]


def build_agent() -> DurableAgent:
    """Construct the DurableAgent with the Tavily before_llm_call hook attached."""
    return DurableAgent(
        name="ExpertAgent",
        role="Expert assistant with live web context",
        goal="Answer user questions using fresh web context.",
        instructions=INSTRUCTIONS,
        llm=OpenAIChatClient(model="gpt-4o-mini"),
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(store_name="conversationstore"),
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflowstatestore"),
        ),
        registry=AgentRegistryConfig(
            store=StateStoreService(store_name="agentregistrystore"),
            team_name="default",
        ),
        execution=AgentExecutionConfig(max_iterations=1),
        hooks=Hooks(before_llm_call=[enrich_with_tavily]),
    )
