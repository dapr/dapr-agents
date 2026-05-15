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

# Anthropic Claude LLM calls with Dapr Agents

This example demonstrates how to use Dapr Agents' `AnthropicChatClient` to call Claude models through the native Anthropic Python SDK.

## Prerequisites

- Python >= 3.11
- uv package manager
- Anthropic API key

## Environment Setup

```bash
uv venv
# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
uv sync --active
```

## Configuration

Create a `.env` file in this directory:

```env
ANTHROPIC_API_KEY=your_api_key_here
# Optional: pin a specific Claude model (default is claude-sonnet-4-6)
ANTHROPIC_MODEL=claude-sonnet-4-6
# Optional: route through a proxy or Bedrock-compatible endpoint
ANTHROPIC_BASE_URL=https://api.anthropic.com
```

## Examples

### 1. Basic text completion

```bash
uv run python text_completion.py
```

Demonstrates:
- A bare `AnthropicChatClient()` call with a string prompt.
- Passing a typed `UserMessage` list.
- Using the Anthropic-native `thinking={"type": "enabled", "budget_tokens": 1024}` parameter for extended reasoning.

### 2. Streaming

```bash
uv run python text_completion_stream.py
```

Reads `chunk.result.content` from the streamed `LLMChatResponseChunk` iterator and prints tokens as they arrive.

## Anthropic-native features

`AnthropicChatClient.generate(**kwargs)` forwards unrecognized kwargs straight to `client.messages.create`, so you can pass any of:

- `thinking={"type": "enabled", "budget_tokens": N}` — extended reasoning.
- `system="..."` — top-level system prompt (or include `role=system` messages and they will be collapsed automatically).
- `tool_choice`, `top_k`, `top_p`, `stop_sequences`, `metadata`, `service_tier`, `extra_headers` (e.g. for beta features).

Prompt caching works on both user/assistant messages and the system prompt: pass `cache_control` on individual content blocks. For the system prompt, supply `{"role": "system", "content": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]}` — the list-of-blocks form is forwarded verbatim to Anthropic's top-level `system` parameter. A plain string system message still works for the non-cached case.

`AgentTool` instances are rendered automatically into Anthropic's `{name, description, input_schema}` form via `ToolHelper.format_tool(t, tool_format="claude")`. Dict tools must already match the Claude shape — OpenAI-style `{"type": "function", "function": {...}}` dicts are not converted and will fail validation.

## Structured outputs

Pass a Pydantic model as `response_format` and the client returns a validated instance:

```python
from pydantic import BaseModel
from dapr_agents.llm.anthropic.chat import AnthropicChatClient

class Contact(BaseModel):
    name: str
    email: str
    plan_interest: str
    demo_requested: bool

client = AnthropicChatClient()
contact = client.generate(
    "Extract: John Smith (john@example.com) wants the Enterprise plan and a demo Tuesday at 2pm.",
    response_format=Contact,
)
```

Two `structured_mode` values are supported:

- `"json"` *(default)* — Anthropic's native `output_config.format` with `type: "json_schema"`. The schema is compiled to a grammar server-side and the model is constrained to emit valid JSON. Requires Sonnet 4.5+, Opus 4.1+, or Haiku 4.5+; older models (Sonnet 3.x / Opus 3.x / Haiku 3.x) will get a runtime API error — use `structured_mode="function_call"` for those.
- `"function_call"` — emulation via a forced `tool_choice`; useful for older Claude models or when you want the structured payload to appear as a `tool_use` block alongside other tool calls.

Combining `response_format` with `stream=True` returns a chunk iterator, not a validated model — the request is still constrained server-side, but the caller is responsible for buffering and parsing the streamed JSON.

Passing `response_format=list[Contact]` wraps the type into a generated `IterableContact` model. The call returns an `IterableContact` instance whose `.objects` field is the parsed list.

## Prompty

Load a Prompty file to pin model, parameters, and the prompt template in one place:

```python
from dapr_agents.llm.anthropic.chat import AnthropicChatClient

client = AnthropicChatClient.from_prompty("basic.prompty")
response = client.generate(input_data={"question": "What is your name?"})
```

The Prompty's `configuration.type` must be `anthropic`, and the model name must be set via either `parameters.model` or `configuration.name`. Parameters declared in the Prompty (e.g. `temperature`, `max_tokens`, `top_k`) become defaults; per-call `generate(**kwargs)` overrides them.
