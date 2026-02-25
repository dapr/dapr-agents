# Durable Agent Hot-Reload Example

This example demonstrates how to enable a `DurableAgent` to subscribe to a Dapr Configuration Store for dynamic, zero-downtime updates to its persona and settings.

## Prerequisites

- [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/) installed and initialized.
- Redis running (default with `dapr init`), **or** PostgreSQL for the alternative backend.

## Structure

- `agent.py`: The agent implementation using `AgentConfigurationConfig`.
- `components/configstore.yaml`: Redis-backed configuration store (default).
- `components/configstore-postgres.yaml`: PostgreSQL-backed configuration store (alternative).
- `components/`: Other Dapr component definitions.

## Running the Example

### With Redis (default)

1. **Start the Agent with Dapr:**

   ```bash
   dapr run --app-id hot-reload-demo \
            --resources-path ./components \
            -- python agent.py
   ```

   > **Note:** Remove or rename `configstore-postgres.yaml` so only one `configstore` component is loaded.

2. **Trigger a Hot-Reload via redis-cli:**

   ```bash
   redis-cli SET agent_role "Expert Data Scientist"
   redis-cli SET agent_goal "Analyze complex datasets"
   redis-cli SET max_iterations "20"
   ```

### With PostgreSQL

#### 1. Start PostgreSQL

If you don't already have PostgreSQL running, you can start one with Docker:

```bash
docker run -d --name dapr-config-pg \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=dapr_config \
  -p 5432:5432 \
  postgres:16
```

#### 2. Create the configuration table and trigger

Dapr's PostgreSQL configuration store requires a table and a NOTIFY trigger so subscriptions receive real-time updates. Connect with `psql` and run:

```bash
psql -h localhost -U postgres -d dapr_config
```

```sql
-- Configuration table
CREATE TABLE IF NOT EXISTS configuration (
    KEY     VARCHAR NOT NULL,
    VALUE   VARCHAR NOT NULL,
    VERSION VARCHAR NOT NULL DEFAULT '1',
    METADATA JSON,
    PRIMARY KEY (KEY)
);

-- Trigger function that fires a pg_notify on every INSERT or UPDATE.
-- The channel name MUST match the pgNotifyChannel in the
-- AgentConfigurationConfig metadata dict (default: "config").
CREATE OR REPLACE FUNCTION notify_event()
RETURNS TRIGGER AS $$
DECLARE
    data JSON;
    notification JSON;
BEGIN
    IF (TG_OP = 'DELETE') THEN
        data = row_to_json(OLD);
    ELSE
        data = row_to_json(NEW);
    END IF;
    notification = json_build_object(
        'table', TG_TABLE_NAME,
        'action', TG_OP,
        'data', data
    );
    PERFORM pg_notify('config', notification::TEXT);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS config_trigger ON configuration;
CREATE TRIGGER config_trigger
    AFTER INSERT OR UPDATE ON configuration
    FOR EACH ROW EXECUTE FUNCTION notify_event();
```

#### 3. (Optional) Pre-populate initial values

Because the agent loads existing configuration on startup, you can seed values before launching:

```sql
INSERT INTO configuration (key, value, version) VALUES
    ('agent_role',         'Data Scientist',          '1'),
    ('agent_goal',         'Analyze complex datasets', '1'),
    ('agent_instructions', '["Use tables", "Cite sources"]', '1'),
    ('max_iterations',     '15',                       '1');
```

#### 4. Switch the Dapr component to PostgreSQL

Only one component named `configstore` can be loaded at a time. Swap the Redis component for the PostgreSQL one:

```bash
mv components/configstore.yaml components/configstore-redis.yaml
cp components/configstore-postgres.yaml components/configstore.yaml
```

Update the `connectionString` in `configstore.yaml` to match your environment.

#### 5. Start the Agent

```bash
dapr run --app-id hot-reload-demo \
         --resources-path ./components \
         -- python agent.py
```

#### 6. Trigger a Hot-Reload via psql

Insert a new key or update an existing one — the trigger fires `pg_notify` and Dapr pushes the change to the agent:

```sql
-- Insert a new key
INSERT INTO configuration (key, value, version)
VALUES ('agent_role', 'Expert Researcher', '1')
ON CONFLICT (key) DO UPDATE
  SET value   = EXCLUDED.value,
      version = (configuration.version::int + 1)::text;

-- Update multiple keys
UPDATE configuration SET value = 'Summarize research papers', version = (version::int + 1)::text
WHERE key = 'agent_goal';

UPDATE configuration SET value = '25', version = (version::int + 1)::text
WHERE key = 'max_iterations';
```

> **Important:** The `version` column must be incremented on each update — Dapr uses it to detect changes. The `ON CONFLICT ... DO UPDATE` pattern handles this automatically for upserts.

### Observing Changes

The agent logs its current persona every 10 seconds. After updating a key, you will see:

```
Agent config-aware-agent applying config update: agent_role=Expert Data Scientist
Current Persona: [Expert Data Scientist] - ...
```

## Initial Configuration Loading

When the agent starts, it calls `get_configuration()` to load any pre-existing values from the store **before** subscribing to changes. This means you can pre-populate the configuration store and the agent will pick up those values on startup.

## pgNotifyChannel (PostgreSQL only)

The Dapr PostgreSQL configuration store requires a `pgNotifyChannel` key in the **subscribe request metadata** — setting it in the component YAML alone is not sufficient ([source](https://github.com/dapr/components-contrib/blob/main/configuration/postgres/postgres.go)). The agent automatically injects `pgNotifyChannel: "config"` into the subscribe metadata so this works out of the box. Redis ignores the key, so the same code works for both backends.

If your PostgreSQL trigger uses a different channel name, override it via `AgentConfigurationConfig.metadata`:

```python
config = AgentConfigurationConfig(
    store_name="configstore",
    keys=["agent_role"],
    metadata={"pgNotifyChannel": "my_custom_channel"},
)
```

## Supported Keys

Keys are normalized (lowercased, hyphens replaced with underscores) before matching. Values from the Dapr Configuration API are always strings; the agent coerces them to the target type automatically.

### Profile

| Key | Aliases | Type | Description |
|-----|---------|------|-------------|
| `role` | `agent_role` | `str` | Agent role string |
| `goal` | `agent_goal` | `str` | Agent goal string |
| `instructions` | `agent_instructions` | `list` | Single string or JSON list of instructions |
| `system_prompt` | `agent_system_prompt` | `str` | System prompt override |
| `style_guidelines` | `agent_style_guidelines` | `list` | Single string or JSON list of style directives |

### Execution

| Key | Type | Description |
|-----|------|-------------|
| `max_iterations` | `int` | Maximum reasoning steps (must be >= 1) |
| `tool_choice` | `str` | LLM tool selection mode (`auto`, `none`, `required`) |

### LLM

| Key | Type | Description |
|-----|------|-------------|
| `llm_api_key`, `openai_api_key` | `str` | LLM API key (redacted in logs) |
| `llm_provider` | `str` | LLM provider name |
| `llm_model` | `str` | LLM model name |

### Component References

| Key | Type | Description |
|-----|------|-------------|
| `state_store` | `str` | State store component name |
| `registry_store` | `str` | Registry store component name |
| `memory_store` | `str` | Memory store component name |

### JSON Batch Updates

Values can be sent as a JSON object. When the configuration value is a JSON dictionary, each key-value pair is applied individually:

```bash
redis-cli SET agent_config '{"role": "Researcher", "goal": "Find answers", "max_iterations": "15"}'
```

> **Note:** The key (`agent_config` above) must be one of the keys the agent subscribes to (defined in `AgentConfigurationConfig.keys`). Unsubscribed keys will be ignored.

## Callback on Configuration Change

You can provide an `on_config_change` callback to react to configuration updates:

```python
def on_change(key: str, value):
    print(f"Config changed: {key} = {value}")

config = AgentConfigurationConfig(
    store_name="configstore",
    keys=["agent_role", "agent_goal"],
    on_config_change=on_change,
)
```
