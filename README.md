---
title: Enterprise AI Document Intelligence Engine
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Enterprise AI Document Intelligence Engine

A full-stack Retrieval-Augmented Generation (RAG) application that lets you upload documents and ask questions about them in natural language. Built with FastAPI, Streamlit, Pinecone, and Google Gemini — deployed as a single Docker container on Hugging Face Spaces.

🔗 **Live Demo:** [huggingface.co/spaces/bhuvaneswari2620/enterprise-document-intelligence-engine](https://huggingface.co/spaces/bhuvaneswari2620/enterprise-document-intelligence-engine)

---

## What it does

- Upload PDF, TXT, or DOCX files
- Documents are chunked, embedded, and stored in a Pinecone vector database
- Ask questions in natural language and get streamed, cited answers
- Each browser session is isolated in its own Pinecone namespace
- Session data can be cleared on demand
- Includes a custom LLM-as-judge evaluation harness for measuring faithfulness, answer relevance, and context precision

---

## Architecture
User → Streamlit Frontend (app.py, port 7860, public)

│

▼ HTTP (localhost)

FastAPI Backend (src/api/main.py, port 8000, internal)

│

┌──────────────────────────────┐

│  Upload Pipeline             │

│  load → clean → chunk        │

│  → embed → Pinecone upsert   │

└──────────────────────────────┘

│

┌──────────────────────────────┐

│  Query Pipeline              │

│  embed query → Pinecone      │

│  retrieval → Gemini LLM      │

│  → streamed SSE response     │

└──────────────────────────────┘
Both processes run inside a single Docker container, started and supervised by `start.sh`. Hugging Face Spaces exposes only one public port (7860), so the Streamlit frontend listens there while the FastAPI backend listens internally on 8000.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Embeddings | `BAAI/bge-small-en-v1.5` (SentenceTransformers) |
| Vector Store | Pinecone (Serverless, cosine similarity) |
| LLM | Google Gemini 2.0 Flash |
| Evaluation | Custom LLM-as-judge harness (faithfulness, relevance, context precision) |
| Deployment | Hugging Face Spaces (Docker SDK) |
| Containerization | Docker (multi-stage build) |

---

## Project Structure
├── app.py                          # Streamlit frontend

├── start.sh                        # Supervisor: launches backend + frontend in one container

├── src/

│   ├── api/main.py                 # FastAPI app, upload & query endpoints

│   ├── ingestion/

│   │   ├── load_documents.py       # PDF / TXT / DOCX loader

│   │   ├── chunker.py              # Pure-Python overlap chunker

│   │   └── text_cleaner.py        # Text normalization

│   ├── embeddings/

│   │   ├── embedding_generator.py  # Embedding provider router (hash / sentence-transformer)

│   │   ├── hash_embedding.py       # Lightweight fallback embedding provider

│   │   └── sparse_encoder.py       # Optional sparse encoding for hybrid search

│   ├── vector_store/

│   │   └── pinecone_store.py       # Pinecone upsert, namespace management

│   ├── retrieval/

│   │   ├── retriever.py            # Dense retrieval, optional reranker

│   │   └── evaluator.py            # LLM-as-judge evaluation harness

│   ├── generation/

│   │   ├── prompt_builder.py       # RAG prompt construction

│   │   ├── llm_client.py           # Gemini / Grok / mock provider router

│   │   └── llm_generator.py        # Gemini SDK wrapper with retry/backoff

│   └── storage/

│       └── session_file_store.py   # Per-session file management

├── tests/

│   ├── test_chunker.py             # Unit tests (no external deps)

│   ├── test_text_cleaner.py        # Unit tests (no external deps)

│   └── test_rag_evaluation.py      # Gated live evaluation (RUN_EVAL=true)

├── Dockerfile

├── .dockerignore

└── requirements.txt

---

## Key Engineering Decisions

**Single-container, dual-process deployment.** Hugging Face Spaces (Docker SDK) exposes exactly one port and runs one process. `start.sh` launches the FastAPI backend in the background, waits for its `/health` check to pass, then starts Streamlit in the foreground bound to the public port. A POSIX-compliant polling loop supervises both PIDs and brings the container down cleanly if either process dies — `bash`-only constructs like `wait -n` were deliberately avoided since the base image's `/bin/sh` is `dash`.

**Multi-stage Docker build.** Dependencies are installed in a `builder` stage and copied into a slim `runner` stage, keeping the final image free of build tooling. The container runs as a non-root user, per Hugging Face Spaces requirements.

**Embedding model preloaded at startup.** `sentence-transformers` is loaded once via FastAPI's `lifespan` context before any request arrives, avoiding cold-start latency or memory spikes mid-request.

**Pluggable embedding and LLM providers.** `EMBEDDING_PROVIDER` switches between a real semantic embedding model and a lightweight hash-based fallback for constrained environments. `LLM_PROVIDER` supports Gemini, Grok, and a mock mode, with automatic fallback on quota/rate-limit errors.

**Session-scoped namespaces.** Each browser session gets a unique Pinecone namespace, so uploads from different users never mix in search results.

**Custom evaluation harness.** `ProductionRAGEvaluator` uses Gemini itself as a judge to score generated answers on faithfulness, answer relevance, and context precision — gated behind `RUN_EVAL=true` since it requires live API access and a populated index.

---

## Running Locally

**Prerequisites:** Python 3.11, Docker (optional)

```bash
git clone https://github.com/your-username/Enterprise-AI-Document-Intelligence-Engine.git
cd Enterprise-AI-Document-Intelligence-Engine
cp .env.example .env
# Fill in PINECONE_API_KEY and GEMINI_API_KEY in .env
```

**With Docker:**
```bash
docker build -t doc-intel .
docker run -p 7860:7860 --env-file .env doc-intel
# App available at http://localhost:7860
```

**Without Docker:**
```bash
pip install -r requirements.txt
chmod +x start.sh
./start.sh
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PINECONE_API_KEY` | Pinecone API key | required |
| `GEMINI_API_KEY` | Google Gemini API key | required |
| `PINECONE_INDEX_NAME` | Pinecone index name | `ai-document-intelligence` |
| `PINECONE_REGION` | Pinecone serverless region | `us-east-1` |
| `EMBEDDING_PROVIDER` | `sentence_transformer` or `hash` | `sentence_transformer` |
| `EMBEDDING_MODEL_NAME` | SentenceTransformers model | `BAAI/bge-small-en-v1.5` |
| `LLM_PROVIDER` | `gemini`, `grok`, or `mock` | `gemini` |
| `GEMINI_MODEL` | Gemini model string | `gemini-2.0-flash` |
| `BACKEND_PORT` | Internal FastAPI port | `8000` |
| `FRONTEND_PORT` | Public Streamlit port | `7860` |
| `RUN_EVAL` | Enable live LLM-as-judge evaluation script | `false` |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/upload` | Upload and index a document (async, returns 202) |
| `POST` | `/api/v1/query` | Query documents (SSE streaming) |
| `POST` | `/api/v1/clear-session` | Clear session namespace and files |

Full interactive docs available internally at `/docs` (Swagger UI) on the backend port.

---

## Testing

```bash
# Unit tests (no external dependencies, safe for CI)
python3 tests/test_chunker.py
python3 tests/test_text_cleaner.py

# Live evaluation (requires GEMINI_API_KEY and populated Pinecone index)
RUN_EVAL=true python3 tests/test_rag_evaluation.py
```

---

## License

MIT