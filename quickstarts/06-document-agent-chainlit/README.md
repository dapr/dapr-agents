# A conversational agent over unstructured documents with Chainlit

This quickstart demonstrates how to build a fully functional, enterprise-ready agent that can parse unstructured documents, learn them and converse with users over their contents while remembering all previous interactions. This example also shows how to integrate Dapr with Chainlit, giving users a fully functional chat interface to talk to their agent.

## Key Benefits

- **Converse With Unstructured Data**: Users can upload documents and have them parsed, contextualized and be made chattable
- **Conversational Memory**: The agent maintains context across interactions in the user's [database of choice](https://docs.dapr.io/reference/components-reference/supported-state-stores/)
- **UI Interface**: Use an out-of-the-box, LLM-ready chat interface using [Chainlit](https://github.com/Chainlit/chainlit)
- **Cloud Agnostic**: Uploads are handled automatically by Dapr and can be configured to target [different backends](https://docs.dapr.io/reference/components-reference/supported-bindings)

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

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

## File Upload Configuration (Optional)

Dapr will upload your files to a backend of your choice. The default YAML file in `./components/filestorage.yaml` targets an S3 bucket, but can be configured to be any of the available Dapr output binding components [here](https://docs.dapr.io/reference/components-reference/supported-bindings/).

If you leave the YAML file as-is, the example will run without uploading the file. An error might appear in the console when you upload the file - that's fine and you can ignore it if the storage provider is not configured.

## Examples

### Upload a PDF and chat to a document agent

Run the agent:

```bash
dapr run --app-id doc-agent --resources-path ./components -- chainlit run app.py -w
```

Wait until the browser opens up. Once open, you're ready to upload any document and start asking questions about it!
You can find the agent page at http://localhost:8000.

Upload a PDF of your choice, or use the example `red_foxes.pdf` file in this example.

#### Testing the agent's memory

If you exit the app and restart it, the agent will remember all the previously uploaded documents. The documents are stored in the binding component configured in `./components/filestorage.yaml`.

When you install Dapr using `dapr init`, Redis is installed by default and this is where the conversation memory is saved. To change it, edit the `./components/conversationmemory.yaml` file.

## Summary

**How It Works:**
1. Dapr starts, loading the file storage and conversation history storage configs from the `components` folder.
2. Chainlit loads and starts the agent UI in your browser.
3. When a file is uploaded, the contents are parsed and fed to the agent to be able to answer questions.
4. If the file storage component YAML is correctly configured, Dapr will upload the file to the storage provider.
5. The conversation history is automatically managed by Dapr and saved in the state store configured in `./components/conversationmemory.yaml`.
