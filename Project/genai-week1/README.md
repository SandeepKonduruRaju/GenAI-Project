# GenAI Week 1

Simple OpenAI Python demos for:
- Chat with short-term memory (`chatbot.py`)
- Embedding similarity search (`embedding_demo.py`)
- Basic RAG flow: retrieve + generate (`rag.py`)

## 1) Prerequisites
- Python 3.10+
- OpenAI account + API key

Install dependencies:

```bash
pip install openai python-dotenv numpy
```

## 2) Environment setup
Create a `.env` file in this folder:

```env
OPENAI_API_KEY=your_api_key_here
```

> Keep `.env` private. Do not commit API keys.

## 3) Run scripts
From this folder (`Project/genai-week1`):

### Chatbot demo
```bash
python chatbot.py
```

### Embedding similarity demo
```bash
python embedding_demo.py
```
Example query: `best phone camera`

### RAG demo
```bash
python rag.py
```
Example query: `What is RAG?`

## 4) What each script does
- `chatbot.py`: Sends conversation history to `gpt-4o-mini` with a small memory window.
- `embedding_demo.py`: Converts documents and query to embeddings, computes cosine similarity, returns best match.
- `rag.py`: Retrieves most relevant context using embeddings, then asks the LLM to answer based on that context.

## 5) Common issues
- **"API key not found"**: Check `.env` exists and `OPENAI_API_KEY` is correct.
- **"model not found / access denied"**: Your account may not have access to that model yet.
- **GitHub push blocked for secrets**: Rotate leaked keys and keep `.env` ignored.
