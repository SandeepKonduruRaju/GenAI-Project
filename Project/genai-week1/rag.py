from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

# Knowledge base (your data)
documents = [
    "Transformers are deep learning models used in NLP.",
    "Neural networks are inspired by the human brain.",
    "RAG combines retrieval and generation.",
    "Embeddings convert text into numerical vectors."
]

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