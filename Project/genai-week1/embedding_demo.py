from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

# Sample data (your knowledge base)
documents = [
    "Best smartphones for photography",
    "Top laptops for programming",
    "Healthy diet tips",
    "How to learn machine learning"
]

# Function to get embedding
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Convert documents to embeddings
doc_embeddings = [get_embedding(doc) for doc in documents]

# Similarity function (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# User query
query = input("Enter your search: ")

query_embedding = get_embedding(query)

# Find best match
scores = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]

best_match_index = np.argmax(scores)

print("\nBest match:", documents[best_match_index])