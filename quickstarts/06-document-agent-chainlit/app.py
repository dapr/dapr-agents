import chainlit as cl
from dapr.clients import DaprClient
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

from dapr_agents import Agent
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.types import AssistantMessage
from dapr_agents import OpenAIChatClient

load_dotenv()

instructions = [
    "You are an assistant designed to understand and converse about user-uploaded documents. "
    "Your primary goal is to provide accurate, clear, and helpful answers based solely on the contents of the uploaded document. "
    "If something is unclear or you need more context, ask thoughtful clarifying questions. "
    "Avoid making assumptions beyond the document. Stay focused on what's written, and help the user explore or understand it as deeply as they'd like."
]

agent = Agent(
    name="KnowledgeBase",
    role="Content Expert",
    instructions=instructions,
    memory=ConversationDaprStateMemory(
        store_name="conversationstore", session_id="my-unique-id"
    ),
    llm=OpenAIChatClient(model="gpt-3.5-turbo"),
)


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

    # Use extract LLM-ready text and associated metadata
    document_text = "\n\n".join(
        [
            f"[{el.category}] {el.text.strip()}"
            for el in elements
            if el.text and el.text.strip()
        ]
    )

    # upload the file
    with open(text_file.path, "rb") as f:
        file_bytes = f.read()
        upload(file_bytes, text_file.name, "upload")

    # give the model the document to learn
    response: AssistantMessage = await agent.run(
        "This is a document element to learn: " + document_text
    )

    await cl.Message(content=f"`{text_file.name}` uploaded.").send()

    await cl.Message(content=response.content).send()


@cl.on_message
async def main(message: cl.Message):
    # chat to the model about the document
    result: AssistantMessage = await agent.run(message.content)

    await cl.Message(
        content=result.content,
    ).send()


def upload(contents: bytes, filename: str, binding_name: str):
    # upload the file to any storage provider using the same code with Dapr
    try:
        with DaprClient() as d:
            d.invoke_binding(
                binding_name=binding_name,
                operation="create",
                data=contents,
                binding_metadata={"key": filename},
            )
            print(f"Uploaded file to storage: {filename}")
    except Exception as e:
        print(f"Upload failed: {e}")
