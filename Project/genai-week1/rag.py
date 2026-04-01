from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Create a .env file in this folder with: OPENAI_API_KEY=your_key"
    )

api_key = api_key.strip()
if api_key.startswith("PASTE_") or "your_" in api_key.lower():
    raise RuntimeError(
        "OPENAI_API_KEY is a placeholder value. Update .env with your real key from https://platform.openai.com/api-keys"
    )

client = OpenAI(api_key=api_key)

text = """
Neural networks are a type of machine learning model inspired by the human brain.
They consist of layers of interconnected nodes called neurons.
These networks are widely used in image recognition, natural language processing, and many AI applications.

Transformers are a type of neural network architecture introduced in 2017.
They use attention mechanisms to understand relationships between words.
Transformers power modern language models like GPT.

Embeddings are numerical representations of text.
They allow machines to understand semantic meaning and similarity between words and sentences.
"""

def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

documents = chunk_text(text, chunk_size=50)

print("\nChunks:")
for i, doc in enumerate(documents):
    print(f"{i}: {doc}\n")

# Get embedding
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Create embeddings for documents
doc_embeddings = [get_embedding(doc) for doc in documents]

# Similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# User question
query = input("Ask a question: ")

query_embedding = get_embedding(query)

# Find best matching document
scores = [cosine_similarity(query_embedding, emb) for emb in doc_embeddings]
best_match_index = np.argmax(scores)

retrieved_doc = documents[best_match_index]

print("\nRetrieved context:", retrieved_doc)

# Now send to LLM
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an AI assistant that answers based on provided context."},
        {"role": "user", "content": f"Context: {retrieved_doc}\n\nQuestion: {query}"}
    ],
    max_tokens=150
)

answer = response.choices[0].message.content

print("\nAI Answer:", answer)