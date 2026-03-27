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

# A conversational agent over unstructured documents with Chainlit

This example demonstrates how to build a fully functional, enterprise-ready agent that can parse unstructured documents, learn them and converse with users over their contents while remembering all previous interactions. This example also shows how to integrate Dapr with Chainlit, giving users a fully functional chat interface to talk to their agent.

## Key Benefits

- **Converse With Unstructured Data**: Users can upload documents and have them parsed, contextualized and be made chattable
- **Conversational Memory**: The agent maintains context across interactions in the user's [database of choice](https://docs.dapr.io/reference/components-reference/supported-state-stores/)
- **UI Interface**: Use an out-of-the-box, LLM-ready chat interface using [Chainlit](https://github.com/Chainlit/chainlit)
- **Cloud Agnostic**: Uploads are handled automatically by Dapr and can be configured to target [different backends](https://docs.dapr.io/reference/components-reference/supported-bindings)

## Prerequisites

- uv package manager
- OpenAI API key (for the OpenAI example)
- [Dapr CLI installed](https://docs.dapr.io/getting-started/install-dapr-cli/)
- Poppler (for PDF processing) - Install with:
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
  - Windows: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)
- Tesseract OCR (for text extraction from images/PDFs) - Install with:
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: Download from [tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

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

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

## File Upload Configuration (Optional)

Uploaded documents are saved to a Dapr state store (Redis by default) via `./resources/filestorage.yaml`. This can be changed to any [supported state store](https://docs.dapr.io/reference/components-reference/supported-state-stores/).

## Examples

### Upload a PDF and chat to a document agent

Run the agent:

```bash
uv run dapr run --app-id doc-agent --resources-path ./resources -- chainlit run app.py -w
```

Wait until the browser opens up. Once open, you're ready to upload any document and start asking questions about it!
You can find the agent page at http://localhost:8000.

Upload a PDF of your choice, or use the example `red_foxes.pdf` file in this example.

#### Testing the agent's memory

If you exit the app and restart it, the agent will remember all the previously uploaded documents. The documents are stored in the binding component configured in `./resources/filestorage.yaml`.

When you install Dapr using `dapr init`, Redis is installed by default and this is where the conversation memory is saved. To change it, edit the `./resources/conversationmemory.yaml` file.

## Summary

**How It Works:**
1. Dapr starts, loading the state store and workflow configs from the `resources` folder.
2. Chainlit loads and starts the agent UI in your browser.
3. When a file is uploaded, the contents are parsed and fed to the agent to be able to answer questions.
4. The uploaded file is saved to the Redis state store configured in `./resources/filestorage.yaml`.
5. The conversation history is automatically managed by Dapr and saved in the state store configured in `./resources/conversationmemory.yaml`.
