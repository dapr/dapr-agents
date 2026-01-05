from dapr_agents.document.embedder import NVIDIAEmbedder
from dotenv import load_dotenv

load_dotenv()

# Initialize the embedder
embedder = NVIDIAEmbedder(
    model="nvidia/nv-embedqa-e5-v5",  # Default embedding model
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
