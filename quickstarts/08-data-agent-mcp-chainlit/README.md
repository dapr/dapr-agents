# A conversational agent over a Postgres database using MCP

This quickstart demonstrates how to build a fully functional, enterprise-ready agent that allows users to ask their database any question in natural text and get both the results and a highly structured analysis of complex questions. This quickstart also shows the usage of MCP in Dapr Agents to connect to the database and provides a fully functional ChatGPT-like chat interface using Chainlit.

## Key Benefits

- **Conversational Knowledge Base**: Users can talk to their database in natural language, ask complex questions and perform advanced analysis over data
- **Conversational Memory**: The agent maintains context across interactions in the user's [database of choice](https://docs.dapr.io/reference/components-reference/supported-state-stores/)
- **UI Interface**: Use an out-of-the-box, LLM-ready chat interface using [Chainlit](https://github.com/Chainlit/chainlit)
- **Boilerplate-Free DB Layer**: MCP allows the Dapr Agent to connect to the database without requiring users to write Postgres-specific code

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key (for the OpenAI example)
- [Dapr CLI installed](https://docs.dapr.io/getting-started/install-dapr-cli/)

## Environment Setup

```bash
# Create a virtual environment
python3.10 -m venv .venv

# Activate the virtual environment 
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize Dapr
dapr init
```

## LLM Configuration

For this example, we'll be using the OpenAI client that is used by default. To target different LLMs, see [this example](../02_llm_call_dapr/README.md).

The quickstart includes an OpenAI component configuration in the `components` directory. You have two options to configure your API key:

### Option 1: Using Environment Variables (Recommended)

1. Create a `.env` file in the project root and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

2. When running the examples with Dapr, use the helper script to resolve environment variables:
```bash
# Get the environment variables from the .env file:
export $(grep -v '^#' ../../.env | xargs)

# Create a temporary resources folder with resolved environment variables
temp_resources_folder=$(../resolve_env_templates.py ./components)

# Run your dapr command with the temporary resources
dapr run --app-id sql --resources-path $temp_resources_folder -- chainlit run app.py -w --port 8001

# Clean up when done
rm -rf $temp_resources_folder
```

Note: The temporary resources folder will be automatically deleted when the Dapr sidecar is stopped or when the computer is restarted.

### Option 2: Direct Component Configuration

You can directly update the `key` in [components/openai.yaml](components/openai.yaml):
```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: openai
spec:
  type: conversation.openai
  metadata:
    - name: key
      value: "YOUR_OPENAI_API_KEY"
```

Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

Note: Many LLM providers are compatible with OpenAI's API (DeepSeek, Google AI, etc.) and can be used with this component by configuring the appropriate parameters. Dapr also has [native support](https://docs.dapr.io/reference/components-reference/supported-conversation/) for other providers like Google AI, Anthropic, Mistral, DeepSeek, etc.

## Postgres Configuration

### Connect to an existing database

Create an `.env` file in the root directory of this quickstart and insert your database configuration:

```bash
DB_HOST=<HOST>
DB_PORT=<PORT>
DB_NAME=<DATABASE-NAME>
DB_USER=<USER>
DB_PASSWORD=<PASSWORD>
```

### Create a new sample database

First, install Postgres on your machine.

#### Option 1: Using Docker

Create the following directory and copy the sql files there:

```bash
mkdir docker-entrypoint-initdb.d
cp schema.sql users.sql ./docker-entrypoint-initdb.d
```

Run the database container (Make sure you are in the quickstart directory):

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

Finally, create an `.env` file in the root directory of this quickstart and insert your database configuration:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=userdb
DB_USER=admin
DB_PASSWORD=mypassword
```

## MCP Configuration

To get the Dapr Agent to connect to our Postgres database, we'll use a Postgres MCP server.
Change the settings below based on your Postgres configuration:

*Note: If you're running Postgres in a Docker container, change `<HOST>` to `localhost`.*

```bash
docker run --rm -ti -p 8000:8000 \
  -e DATABASE_URI=postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME \
  crystaldba/postgres-mcp --access-mode=unrestricted --transport=sse
```

## Examples

### Load data to Postgres and create a knowledge base chat interface

Run the agent:

```bash
dapr run --app-id sql --resources-path ./components -- chainlit run app.py -w --port 8001
```

Wait until the browser opens up. Once open, you're ready to talk to your Postgres database!
You can find the agent page at http://localhost:8001.

### Ask Questions

Now you can start talking to your data. If using the sample database, ask questions like `Show me all churned users from the past month` and `Can you identify the problematic area in our product that led to users churning?`.

#### Testing the agent's memory

If you exit the app and restart it, the agent will remember all the previous conversation. When you insall Dapr using `dapr init`, Redis is installed by default and this is where the conversation memory is saved. To change it, edit the `./components/conversationmemory.yaml` file.

## Summary

**How It Works:**
1. The MCP server is running and connects to our Postgres database
2. Dapr starts, loading the conversation history storage configs from the `components` folder. The agent connects to the MCP server.
3. Chainlit loads and starts the agent UI in your browser.
4. Users can now talk to their database in natural language and have the agent analyze the data.
5. The conversation history is automatically managed by Dapr and saved in the state store configured in `./components/conversationmemory.yaml`.
