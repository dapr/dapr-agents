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

import base64
import json
from typing import Optional

import chainlit as cl
from dapr.clients import DaprClient
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

from dapr_agents import DurableAgent, OpenAIChatClient, AgentRunner
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService

load_dotenv()

instructions = [
    "You are an assistant designed to understand and converse about user-uploaded documents. "
    "Your primary goal is to provide accurate, clear, and helpful answers based solely on the contents of the uploaded document. "
    "If something is unclear or you need more context, ask thoughtful clarifying questions. "
    "Avoid making assumptions beyond the document. Stay focused on what's written, and help the user explore or understand it as deeply as they'd like."
]

# Lazily initialized on first chat session
_agent: Optional[DurableAgent] = None
_runner: Optional[AgentRunner] = None


def _get_agent_and_runner() -> tuple[DurableAgent, AgentRunner]:
    """Create the DurableAgent and AgentRunner on first use (after Dapr sidecar is ready)."""
    global _agent, _runner
    if _agent is None:
        _agent = DurableAgent(
            name="KnowledgeBase",
            role="Content Expert",
            instructions=instructions,
            memory=AgentMemoryConfig(
                store=ConversationDaprStateMemory(store_name="conversationstore"),
            ),
            state=AgentStateConfig(
                store=StateStoreService(store_name="workflowstatestore"),
            ),
            registry=AgentRegistryConfig(
                store=StateStoreService(store_name="registrystatestore"),
                team_name="default",
            ),
            execution=AgentExecutionConfig(max_iterations=3),
            llm=OpenAIChatClient(model="gpt-4o-mini"),
        )
        _runner = AgentRunner()
        _runner.workflow(_agent)
    return _agent, _runner


def _extract_content(result: str | None) -> str:
    """Extract the content field from a serialized workflow result."""
    if not result:
        return ""
    try:
        parsed = json.loads(result)
        return parsed.get("content", result)
    except (json.JSONDecodeError, AttributeError):
        return result


@cl.on_chat_start
async def start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a document to begin!",
            accept=["application/pdf"],
            max_size_mb=10,
            max_files=1,
        ).send()

    text_file = files[0]
    elements = partition_pdf(filename=text_file.path)

    # Extract LLM-ready text and associated metadata
    document_text = "\n\n".join(
        [
            f"[{el.category}] {el.text.strip()}"
            for el in elements
            if el.text and el.text.strip()
        ]
    )

    # Save the file to the Dapr state store (Redis)
    with open(text_file.path, "rb") as f:
        file_bytes = f.read()
        upload_to_store(file_bytes, text_file.name)

    # Give the model the document to learn
    agent, runner = _get_agent_and_runner()
    result = await runner.run(
        agent, {"task": "This is a document element to learn: " + document_text}
    )

    await cl.Message(content=f"`{text_file.name}` uploaded.").send()
    await cl.Message(content=_extract_content(result)).send()


@cl.on_message
async def main(message: cl.Message):
    agent, runner = _get_agent_and_runner()
    result = await runner.run(agent, {"task": message.content})

    await cl.Message(
        content=_extract_content(result),
    ).send()


def upload_to_store(contents: bytes, filename: str) -> None:
    """Save the uploaded file to the Dapr state store (Redis by default)."""
    try:
        with DaprClient() as d:
            d.save_state(
                store_name="docstore",
                key=f"upload/{filename}",
                value=base64.b64encode(contents).decode("utf-8"),
            )
            print(f"Saved file to state store: {filename}")
    except Exception as e:
        print(f"Upload failed (non-fatal): {e}")
