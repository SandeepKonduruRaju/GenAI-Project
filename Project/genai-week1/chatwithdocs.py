from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError, BadRequestError
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path
import json

# Load embeddings
with open("embeddings.json", "r") as f:
    data = json.load(f)

documents = [item["text"] for item in data]
doc_embeddings = [item["embedding"] for item in data]

# Load .env from this script's folder so running from other directories still works.
load_dotenv(dotenv_path=Path(__file__).with_name('.env'))

api_key = os.getenv('OPENAI_API_KEY', '').strip()
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Add it to .env in this folder as OPENAI_API_KEY=..."
    )

client = OpenAI(api_key=api_key)


# -------- STEP 1: LOAD DOCUMENT --------
with open(Path(__file__).with_name('data.txt'), 'r', encoding='utf-8') as f:
    text = f.read()


# -------- STEP 2: CHUNKING --------
def chunk_text(input_text, chunk_size=100):
    words = input_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i : i + chunk_size]))
    return chunks


documents = chunk_text(text, chunk_size=50)


# -------- STEP 3: EMBEDDINGS --------
def get_embedding(input_text):
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=input_text,
    )
    return response.data[0].embedding


print(f'Creating embeddings for {len(documents)} documents (this may take a moment)...')
doc_embeddings = []
try:
    for i, doc in enumerate(documents):
        print(f'  Processing document {i + 1}/{len(documents)}...')
        doc_embeddings.append(get_embedding(doc))
except APIConnectionError:
    raise RuntimeError(
        'Cannot connect to OpenAI API. Check internet/VPN/firewall settings. '
        'If you are on a restricted network, try another network.'
    )
except AuthenticationError:
    raise RuntimeError('Invalid OPENAI_API_KEY. Update .env with a valid key.')
except RateLimitError:
    raise RuntimeError('Rate limit/quota reached. Check usage and billing on your OpenAI account.')
except BadRequestError as e:
    raise RuntimeError(f'Embedding request failed: {e}')

print('Embeddings created! Ready for chat.\n')


# -------- STEP 4: SIMILARITY --------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------- STEP 5: CHAT LOOP --------
print("Chat with your document (type 'exit' to quit)\n")

while True:
    query = input('You: ').strip()

    if query.lower() == 'exit':
        break
    if not query:
        continue

    try:
        query_embedding = get_embedding(query)

        # Top-K retrieval
        scores = [cosine_similarity(query_embedding, emb) for emb in doc_embeddings]
        top_k = min(3, len(documents))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        retrieved_docs = [documents[i] for i in top_indices]
        context = '\n'.join(retrieved_docs)

        # -------- STEP 6: LLM --------
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    'role': 'system',
                    'content': (
                        "You are a helpful assistant. Answer the user's question using ONLY "
                        'the information in the provided context. If the answer is not in context, say: '
                        "'I don't have information about that in the document.'"
                    ),
                },
                {
                    'role': 'user',
                    'content': f'Context:\n{context}\n\nQuestion: {query}',
                },
            ],
            max_tokens=200,
            temperature=0,
        )

        answer = response.choices[0].message.content or 'No response text returned.'
        print(f'\nAI: {answer}\n')
    except APIConnectionError:
        print('\nError: API connection failed. Check network/VPN/firewall and try again.\n')
    except AuthenticationError:
        print('\nError: Invalid API key. Update OPENAI_API_KEY in .env.\n')
    except RateLimitError:
        print('\nError: Rate limit or quota issue. Check OpenAI usage/billing.\n')
    except Exception as e:
        print(f'\nError: {e}\n')
