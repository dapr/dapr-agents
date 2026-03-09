#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
