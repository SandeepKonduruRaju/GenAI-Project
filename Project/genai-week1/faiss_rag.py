from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError, BadRequestError
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

try:
    import faiss
except ImportError as e:
    raise RuntimeError(
        "faiss is not installed. Install it with: pip install faiss-cpu"
    ) from e


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Add it to .env in this folder as OPENAI_API_KEY=..."
    )

client = OpenAI(api_key=api_key)


def chunk_text(input_text: str, chunk_size: int = 50) -> list[str]:
    words = input_text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def get_embedding(input_text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=input_text,
    )
    return response.data[0].embedding


# 1) Load source document
with open(Path(__file__).with_name("data.txt"), "r", encoding="utf-8") as f:
    text = f.read()

documents = chunk_text(text, chunk_size=50)
if not documents:
    raise RuntimeError("No text chunks were created from data.txt.")

print(f"Creating embeddings for {len(documents)} chunks...")
doc_embeddings = []
try:
    for i, doc in enumerate(documents, start=1):
        print(f"  Processing chunk {i}/{len(documents)}...")
        doc_embeddings.append(get_embedding(doc))
except APIConnectionError:
    raise RuntimeError("Cannot connect to OpenAI API. Check your network and try again.")
except AuthenticationError:
    raise RuntimeError("Invalid OPENAI_API_KEY. Update your .env file.")
except RateLimitError:
    raise RuntimeError("Rate limit/quota reached. Check usage and billing.")
except BadRequestError as e:
    raise RuntimeError(f"Embedding request failed: {e}")


# 2) Build FAISS index (L2 distance over normalized vectors = cosine similarity search)
embedding_array = np.array(doc_embeddings, dtype=np.float32)
faiss.normalize_L2(embedding_array)
index = faiss.IndexFlatIP(embedding_array.shape[1])
index.add(embedding_array)

print("Ready. Chat with your document (type 'exit' to quit)\n")


# 3) Chat loop (this is where user input is requested)
while True:
    query = input("You: ").strip()

    if query.lower() == "exit":
        print("Session ended.")
        break
    if not query:
        continue

    try:
        query_embedding = np.array([get_embedding(query)], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        top_k = min(3, len(documents))
        _, indices = index.search(query_embedding, top_k)
        top_indices = indices[0].tolist()
        retrieved_docs = [documents[i] for i in top_indices if i >= 0]
        context = "\n".join(retrieved_docs)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question using ONLY "
                        "the provided context. If not found in context, say: "
                        "'I don't have information about that in the document.'"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            max_tokens=200,
            temperature=0,
        )

        answer = response.choices[0].message.content or "No response text returned."
        print(f"\nAI: {answer}\n")
    except APIConnectionError:
        print("\nError: API connection failed. Check network/VPN/firewall and try again.\n")
    except AuthenticationError:
        print("\nError: Invalid API key. Update OPENAI_API_KEY in .env.\n")
    except RateLimitError:
        print("\nError: Rate limit or quota issue. Check OpenAI usage/billing.\n")
    except Exception as e:
        print(f"\nError: {e}\n")
