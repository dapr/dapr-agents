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

# A conversational agent over a Postgres database using MCP

This example demonstrates how to build a fully functional, enterprise-ready agent that allows users to ask their database any question in natural text and get both the results and a highly structured analysis of complex questions. This example also shows the usage of MCP in Dapr Agents to connect to the database and provides a fully functional ChatGPT-like chat interface using Chainlit.

## Key Benefits

- **Conversational Knowledge Base**: Users can talk to their database in natural language, ask complex questions and perform advanced analysis over data
- **Conversational Memory**: The agent maintains context across interactions in the user's [database of choice](https://docs.dapr.io/reference/components-reference/supported-state-stores/)
- **UI Interface**: Use an out-of-the-box, LLM-ready chat interface using [Chainlit](https://github.com/Chainlit/chainlit)
- **Boilerplate-Free DB Layer**: MCP allows the Dapr Agent to connect to the database without requiring users to write Postgres-specific code

## Prerequisites

- uv package manager
- OpenAI API key (for the OpenAI example)
- [Dapr CLI installed](https://docs.dapr.io/getting-started/install-dapr-cli/)

## Environment Setup

```bash
uv venv
# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
uv sync --active
# Initialize Dapr
dapr init
```

## LLM Configuration

For this example, we'll be using the OpenAI client that is used by default. To target different LLMs, see [this example](../01-llm-call-dapr/README.md).

The conversation component in [resources/openai.yaml](resources/openai.yaml) reads the API key from the `OPENAI_API_KEY` environment variable via `envRef`:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: openai
spec:
  type: conversation.openai
  version: v1
  metadata:
  - name: key
    envRef: OPENAI_API_KEY
  - name: model
    value: gpt-4o-mini
```

Set the variable in your `.env` file (created in the next section) and Dapr will pick it up automatically.

Note: Many LLM providers are compatible with OpenAI's API (DeepSeek, Google AI, etc.) and can be used with this component by configuring the appropriate parameters. Dapr also has [native support](https://docs.dapr.io/reference/components-reference/supported-conversation/) for other providers like Google AI, Anthropic, Mistral, DeepSeek, etc.

## Postgres Configuration

### Connect to an existing database

Create an `.env` file in the root directory of this example and insert your database configuration:

```bash
DB_HOST=<HOST>
DB_PORT=<PORT>
DB_NAME=<DATABASE-NAME>
DB_USER=<USER>
DB_PASSWORD=<PASSWORD>
```

### Create a new sample database

The example includes sample SQL files (`schema.sql` and `users.sql`) that create a `users` table and seed it with example churn data. This data is used by the example queries in the [Ask Questions](#ask-questions) section below.

#### Option 1: Using Docker

Run the database container (Make sure you are in the example directory). The `docker-entrypoint-initdb.d/` directory is automatically mounted, so the schema and seed data are loaded on first start:

```bash
docker run --rm --name sampledb \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_USER=admin \
  -e POSTGRES_DB=userdb \
  -p 5432:5432 \
  -v $(pwd)/docker-entrypoint-initdb.d:/docker-entrypoint-initdb.d \
  -d postgres
```

#### Option 2: Using Brew

Install Postgres:

```bash
# Install and start PostgreSQL
brew install postgresql
brew services start postgresql

# Create user and database (safe to re-run)
psql postgres <<EOF
CREATE USER admin WITH PASSWORD 'mypassword';
CREATE DATABASE userdb OWNER admin;
GRANT ALL PRIVILEGES ON DATABASE userdb TO admin;
EOF
```

Next, create the users table and seed data:

```bash
psql -h localhost -U admin -d userdb -f schema.sql
psql -h localhost -U admin -d userdb -f users.sql
```

#### Create .env file

Finally, create an `.env` file in the root directory of this example and insert your database configuration:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=userdb
DB_USER=admin
DB_PASSWORD=mypassword
```

## MCP Configuration

To get the Dapr Agent to connect to our Postgres database, we'll use a Postgres MCP server.

First, source your `.env` file so the shell can use the database variables:

```bash
# On macOS/Linux:
export $(grep -v '^#' .env | xargs)
# On Windows (PowerShell):
Get-Content .env | Where-Object { $_ -and -not $_.StartsWith("#") } | ForEach-Object {
    $key, $value = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($key, $value)
}
```

Then start the MCP server:

*Note: If you're running Postgres in a Docker container, change `DB_HOST` to `localhost`.*

```bash
docker run --rm -ti -p 8000:8000 \
  -e DATABASE_URI=postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME \
  crystaldba/postgres-mcp --access-mode=unrestricted --transport=sse
```

## Examples

### Load data to Postgres and create a knowledge base chat interface

Run the agent (this will launch Chainlit on port 8001) after you set the api key:

```bash
uv run dapr run --app-id sql --resources-path resources -- chainlit run app.py -w --port 8001
```

Wait until the browser opens up. Once open, you're ready to talk to your Postgres database!
You can find the agent page at http://localhost:8001.

### Ask Questions

Now you can start talking to your data. If using the sample database, ask questions like `Show me all churned users from the past month` and `Can you identify the problematic area in our product that led to users churning?`.

#### Testing the agent's memory

If you exit the app and restart it, the agent will remember all the previous conversation. When you install Dapr using `dapr init`, Redis is installed by default and this is where the conversation memory is saved. To change it, edit the `./resources/conversationmemory.yaml` file.

## Summary

**How It Works:**
1. The MCP server is running and connects to our Postgres database
2. Dapr starts, loading the conversation history storage configs from the `resources` folder. The agent connects to the MCP server.
3. Chainlit loads and starts the agent UI in your browser.
4. Users can now talk to their database in natural language and have the agent analyze the data.
5. The conversation history is automatically managed by Dapr and saved in the state store configured in `./resources/conversationmemory.yaml`.

## Troubleshooting

1. **OpenAI API Key**: Ensure your key is set in `.env` or baked into `resources/openai.yaml`.
2. **Postgres MCP Server**: The `crystaldba/postgres-mcp` container must be running on port 8000 before launching Chainlit.
3. **Database Access**: The `.env` values for `DB_HOST`, `DB_USER`, etc., must match a reachable database. Run the provided SQL scripts if you use the sample data.
4. **Dependencies**: Run `uv sync --active` inside your virtual environment.
5. **Dapr Timeout**: For long-running conversations set `DAPR_API_TIMEOUT_SECONDS=300` so the Dapr gRPC client waits beyond the 60 s default.
