from dapr_agents.document.embedder import OpenAIEmbedder
from dotenv import load_dotenv

load_dotenv()

# Initialize the embedder
embedder = OpenAIEmbedder(
    model="text-embedding-ada-002",  # Default embedding model
    chunk_size=1000,  # Batch size for processing
    max_tokens=8191,  # Maximum tokens per input
)

# Generate embedding with a single text
text = "Dapr Agents is an open-source framework for researchers and developers"

embedding = embedder.embed(text)

# Display the embedding
if len(embedding) > 0:
    print(f"Embedding (first 5 values): {embedding[:5]}...")

# Multiple input texts
texts = [
    "Dapr Agents is an open-source framework for researchers and developers",
    "It provides tools to create, orchestrate, and manage agents",
]

# Generate embeddings
embeddings = embedder.embed(texts)

if len(embeddings) == 0:
    print("No embeddings generated")
    exit()

# Display the embeddings
for i, emb in enumerate(embeddings):
    print(f"Text {i + 1} embedding (first 5 values): {emb[:5]}")
