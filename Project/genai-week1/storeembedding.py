from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

# Load document
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunking
def chunk_text(text, chunk_size=50):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

documents = chunk_text(text)

# Get embedding
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Create embeddings
data = []

for doc in documents:
    embedding = get_embedding(doc)
    data.append({
        "text": doc,
        "embedding": embedding
    })

# Save to file
with open("embeddings.json", "w") as f:
    json.dump(data, f)

print("Embeddings stored successfully!")