from dapr_agents import Agent
from dapr_agents.tool import tool
from dapr_agents.storage.vectorstores import ChromaVectorStore
from dapr_agents.document.embedder.sentence import SentenceTransformerEmbedder


@tool
def search_documents(query: str) -> str:
    """Search for documents in the vector store"""
    return f"Found documents related to '{query}': Document 1, Document 2"


@tool
def add_document(content: str, metadata: str = "") -> str:
    """Add a document to the vector store"""
    return f"Added document: {content[:50]}..."


async def main():
    embedding_function = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    vector_store = ChromaVectorStore(
        name="demo_vectorstore",
        embedding_function=embedding_function,
        persistent=True,
        path="./chroma_db",
    )

    agent = Agent(
        name="VectorBot",
        role="Vector Database Assistant",
        goal="Help with document search and storage",
        instructions=[
            "Search documents in vector store",
            "Add documents to vector store",
            "Provide relevant information from stored documents",
        ],
        tools=[search_documents, add_document],
        vector_store=vector_store,
    )

    print("Starting Vector Database Agent...")
    print("Adding a sample document...")

    response = await agent.run("Add a document about machine learning basics")
    print("Add Document Response:")
    print(response)
    print()

    print("Searching for documents...")
    response = await agent.run("Search for documents about machine learning")
    print("Search Response:")
    print(response)


if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure you have the required dependencies installed:")
        print(" pip install sentence-transformers chromadb")
