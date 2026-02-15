# Durable Agent Hot-Reload Example

This example demonstrates how to enable a `DurableAgent` to subscribe to a Dapr Configuration Store for dynamic, zero-downtime updates to its persona and settings.

## Prerequisites

- [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/) installed and initialized.
- Redis running (default with `dapr init`).

## Structure

- `agent.py`: The agent implementation using `AgentConfigurationConfig`.
- `components/`: Dapr component definitions for the configuration store and workflow state.

## Running the Example

1. **Start the Agent with Dapr:**

   ```bash
   dapr run --app-id hot-reload-demo \
            --resources-path ./components \
            -- python agent.py
   ```

2. **Observe the Initial State:**
   The agent will start with its default role: `Base Assistant`.

3. **Trigger a Hot-Reload:**
   The Dapr Configuration API is **read-only**. To update configuration values, you must update the underlying store directly.

   **Using redis-cli:**

   ```bash
   redis-cli SET agent_role "Expert Data Scientist"
   ```

   The agent will detect the change in Redis via its Dapr subscription and hot-reload the persona.

4. **Verify the Change:**
   Check the logs of the running agent. You will see:
   `Agent config-aware-agent applying config update: agent_role=Expert Data Scientist`

## Supported Keys

The following keys are recognized by the `_apply_config_update` handler in `AgentBase`. Keys are normalized (lowercased, hyphens replaced with underscores) before matching.

### Profile
| Key | Aliases | Description |
|-----|---------|-------------|
| `role` | `agent_role` | Agent role string |
| `goal` | `agent_goal` | Agent goal string |
| `instructions` | `agent_instructions` | Single string or JSON list of instructions |
| `system_prompt` | `agent_system_prompt` | System prompt override |

### LLM
| Key | Description |
|-----|-------------|
| `llm_api_key`, `openai_api_key` | LLM API key (redacted in logs) |
| `llm_provider` | LLM provider name |
| `llm_model` | LLM model name |

### Component References
| Key | Description |
|-----|-------------|
| `state_store` | State store component name |
| `registry_store` | Registry store component name |
| `memory_store` | Memory store component name |

Values can also be sent as a JSON object. When the configuration value is a JSON dictionary, each key-value pair is applied individually.
