import logging

from dotenv import load_dotenv

from dapr_agents import Agent, OpenAIChatClient
from dapr_agents.document.embedder.sentence import SentenceTransformerEmbedder
from dapr_agents.storage.vectorstores import ChromaVectorStore
from dapr_agents.tool import tool
from dapr_agents.types.document import Document

load_dotenv()

logging.basicConfig(level=logging.INFO)

embedding_function = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
vector_store = ChromaVectorStore(
    name="demo_vectorstore",
    embedding_function=embedding_function,
    persistent=True,
    path="./chroma_db",
)


@tool
def search_documents(query: str) -> str:
    """Search for documents in the vector store"""
    logging.info(f"Searching for documents with query: {query}")
    results = vector_store.search_similar(query_texts=query, k=3)
    docs = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    if not docs:
        logging.info(f"No documents found for query: '{query}'")
        return f"No documents found for query: '{query}'"
    response = []
    for doc, meta in zip(docs, metadatas):
        response.append(f"Text: {doc}\nMetadata: {meta}")
    logging.info(f"Found {len(docs)} documents for query: '{query}'")
    return "\n---\n".join(response)


@tool
def add_document(content: str, metadata: str = "") -> str:
    """Add a document to the vector store"""
    import json

    try:
        meta = json.loads(metadata) if metadata else {}
    except Exception:
        meta = {"info": metadata}
    doc = Document(text=content, metadata=meta)
    logging.info(f"Adding document: {content[:50]}... with metadata: {meta}")
    ids = vector_store.add_documents(documents=[doc])
    logging.info(f"Added document with ID {ids[0]}")
    return f"Added document with ID {ids[0]}: {content[:50]}..."


@tool
def add_machine_learning_doc() -> str:
    """Add a synthetic machine learning basics document to the vector store."""
    content = (
        "Machine Learning Basics: Machine learning is a field of artificial intelligence "
        "that uses statistical techniques to give computer systems the ability to learn "
        "from data, without being explicitly programmed. Key concepts include supervised "
        "learning, unsupervised learning, and reinforcement learning."
    )
    metadata = {"topic": "machine learning", "category": "AI", "level": "beginner"}
    doc = Document(text=content, metadata=metadata)
    logging.info(
        f"Adding synthetic ML document: {content[:50]}... with metadata: {metadata}"
    )
    ids = vector_store.add_documents(documents=[doc])
    if ids and isinstance(ids, list) and len(ids) > 0:
        logging.info(f"Added ML document with ID {ids[0]}")
        return f"Added machine learning basics document with ID {ids[0]}"
    else:
        logging.info("Added ML document, but no ID was returned.")
        return "Added machine learning basics document (no ID returned)"


async def main():
    # Seed the vector store with initial documents using Document class
    documents = [
        Document(
            text="Gandalf: A wizard is never late, Frodo Baggins. Nor is he early; he arrives precisely when he means to.",
            metadata={"topic": "wisdom", "location": "The Shire"},
        ),
        Document(
            text="Frodo: I wish the Ring had never come to me. I wish none of this had happened.",
            metadata={"topic": "destiny", "location": "Moria"},
        ),
        Document(
            text="Sam: I can't carry it for you, but I can carry you!",
            metadata={"topic": "friendship", "location": "Mount Doom"},
        ),
    ]
    logging.info("Seeding vector store with initial documents...")
    ids = vector_store.add_documents(documents=documents)
    logging.info(f"Seeded {len(documents)} initial documents.")
    logging.info(f"Document IDs: {ids}")

    agent = Agent(
        name="VectorBot",
        role="Vector Database Assistant",
        goal="Help with document search and storage",
        instructions=[
            "Search documents in vector store",
            "Add documents to vector store",
            "Add a machine learning basics document",
            "Provide relevant information from stored documents",
        ],
        tools=[search_documents, add_document, add_machine_learning_doc],
        vector_store=vector_store,
        llm=OpenAIChatClient(model="gpt-3.5-turbo"),
    )
    try:
        logging.info("Starting Vector Database Agent...")
        await agent.run("Add a machine learning basics document")
        logging.info("Add Machine Learning Document Response:")
    except Exception as e:
        print(f"Error: {e}")

    try:
        logging.info("Searching for machine learning documents...")
        await agent.run("Search for documents about machine learning")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\nError occurred: {e}")
