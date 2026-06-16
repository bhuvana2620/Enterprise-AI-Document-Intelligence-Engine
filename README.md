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

A full stack Retrieval Augmented Generation application that lets users upload documents and ask questions about them in natural language. The system uses FastAPI, Streamlit, Pinecone, semantic embeddings, and Google Gemini to provide document grounded answers through a deployed Hugging Face Spaces application.

**Live Demo:** [https://huggingface.co/spaces/bhuvaneswari2620/enterprise-document-intelligence-engine](https://huggingface.co/spaces/bhuvaneswari2620/enterprise-document-intelligence-engine)

---

## Overview

Enterprise AI Document Intelligence Engine is designed as a production style RAG system for document understanding. Users can upload PDF, TXT, or DOCX files, the backend extracts and chunks the content, generates embeddings, stores vectors in Pinecone, and retrieves the most relevant context when a user asks a question.

The application is deployed as a single Docker based Hugging Face Space, with Streamlit serving the user interface and FastAPI running as the internal backend service.

---

## Key Features

- Upload PDF, TXT, and DOCX documents
- Extract, clean, and chunk document text
- Generate semantic embeddings using SentenceTransformers
- Store and retrieve document chunks using Pinecone
- Ask natural language questions over uploaded documents
- Generate answers using Google Gemini
- Stream answers back to the frontend
- Use isolated Pinecone namespaces for each browser session
- Clear active session data using the End Session action
- Support local and containerized execution
- Include unit tests for core text processing components
- Provide a custom evaluation harness for RAG quality testing

---

## Live Deployment Status

The application is deployed and running on Hugging Face Spaces.

**Demo URL:** [https://huggingface.co/spaces/bhuvaneswari2620/enterprise-document-intelligence-engine](https://huggingface.co/spaces/bhuvaneswari2620/enterprise-document-intelligence-engine)

The current deployed version supports document upload, semantic chunking, Pinecone indexing, session scoped retrieval, Gemini based answer generation, and manual cleanup through the End Session button.

---

## Architecture

```text
User
  |
  v
Streamlit Frontend
app.py, public port 7860
  |
  | HTTP requests
  v
FastAPI Backend
src/api/main.py, internal port 8000
  |
  +--> Upload Pipeline
  |       load document
  |       clean text
  |       chunk text
  |       generate embeddings
  |       upsert vectors to Pinecone
  |
  +--> Query Pipeline
          embed user query
          retrieve relevant chunks from Pinecone
          build grounded prompt
          generate answer with Gemini
          stream response to frontend
```

Both the frontend and backend run inside one Docker container. Hugging Face Spaces exposes only one public port, so Streamlit listens on port `7860`, while FastAPI runs internally on port `8000`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| Vector Database | Pinecone Serverless |
| Embeddings | SentenceTransformers, BAAI/bge-small-en-v1.5 |
| LLM | Google Gemini |
| Document Processing | PyMuPDF, PyPDF2, python-docx |
| API Style | REST and Server Sent Events |
| Deployment | Hugging Face Spaces with Docker |
| Testing | Pytest and custom evaluation scripts |
| Containerization | Docker |

---

## Project Structure

```text
.
├── app.py
├── start.sh
├── Dockerfile
├── requirements.txt
├── README.md
├── .env.example
├── .streamlit/
│   └── config.toml
├── src/
│   ├── api/
│   │   └── main.py
│   ├── ingestion/
│   │   ├── load_documents.py
│   │   ├── chunker.py
│   │   └── text_cleaner.py
│   ├── embeddings/
│   │   ├── embedding_generator.py
│   │   ├── hash_embedding.py
│   │   └── sparse_encoder.py
│   ├── vector_store/
│   │   └── pinecone_store.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── evaluator.py
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   ├── llm_client.py
│   │   └── llm_generator.py
│   └── storage/
│       └── session_file_store.py
└── tests/
    ├── test_chunker.py
    ├── test_text_cleaner.py
    └── test_rag_evaluation.py
```

---

## Engineering Decisions

### Single Container Deployment

Hugging Face Docker Spaces expose one public port. To support both frontend and backend inside one container, the project uses `start.sh` to launch FastAPI on the internal backend port and Streamlit on the public frontend port.

### Session Scoped Namespaces

Each browser session uses a separate Pinecone namespace. This prevents documents uploaded in one session from mixing with another user session.

### Manual Session Cleanup

The current version supports explicit cleanup through the End Session button. When clicked, the app clears the active Pinecone namespace for that session.

Browser refresh or tab close does not reliably send a cleanup request to the backend, so automatic cleanup for abandoned sessions is listed as a future enhancement.

### Embedding Provider Flexibility

The embedding layer supports semantic embeddings and lightweight fallback behavior. This makes the project easier to run in constrained environments while still supporting production quality embeddings.

### Evaluation Hooks

The project includes a gated evaluation script for measuring answer quality. Live evaluation is disabled by default and can be enabled through environment variables.

---

## Environment Variables

Create a `.env` file locally using `.env.example`.

```env
PINECONE_API_KEY=
GEMINI_API_KEY=

PINECONE_INDEX_NAME=ai-document-intelligence
PINECONE_REGION=us-east-1

EMBEDDING_PROVIDER=sentence_transformer
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

ENABLE_HYBRID_SEARCH=false
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash

BACKEND_PORT=8000
FRONTEND_PORT=7860

RUN_EVAL=false
```

For Hugging Face Spaces, store API keys as Secrets and non sensitive configuration as Variables.

Recommended Hugging Face Secrets:

```text
PINECONE_API_KEY
GEMINI_API_KEY
```

Recommended Hugging Face Variables:

```text
PINECONE_INDEX_NAME=ai-document-intelligence
PINECONE_REGION=us-east-1
ENABLE_HYBRID_SEARCH=false
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash
BACKEND_PORT=8000
FRONTEND_PORT=7860
RUN_EVAL=false
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/bhuvana2620/Enterprise-AI-Document-Intelligence-Engine.git
cd Enterprise-AI-Document-Intelligence-Engine
```

### 2. Create environment file

```bash
cp .env.example .env
```

Update `.env` with your Pinecone and Gemini API keys.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
chmod +x start.sh
./start.sh
```

The app will be available at:

```text
http://localhost:7860
```

---

## Running with Docker

```bash
docker build -t enterprise-doc-intelligence .
docker run -p 7860:7860 --env-file .env enterprise-doc-intelligence
```

Then open:

```text
http://localhost:7860
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend health check |
| POST | `/api/v1/upload` | Upload and index a document |
| POST | `/api/v1/query` | Ask a question over uploaded documents |
| POST | `/api/v1/clear-session` | Clear current session data from Pinecone |

---

## Testing

Run unit tests:

```bash
python3 tests/test_chunker.py
python3 tests/test_text_cleaner.py
```

Run live RAG evaluation only when API keys and Pinecone data are available:

```bash
RUN_EVAL=true python3 tests/test_rag_evaluation.py
```

---

## Current Limitation

The End Session button clears the current session namespace from Pinecone. However, if a user refreshes the page, closes the browser tab, or exits without clicking End Session, the backend does not reliably receive a cleanup signal.

This is expected behavior in browser based applications. Browser close and refresh events are not dependable cleanup triggers, especially in hosted Streamlit deployments. A TTL based cleanup process is planned as a future enhancement.

---

## Future Enhancements

- **Automatic session cleanup:** Add TTL based cleanup for abandoned Pinecone namespaces when users refresh, close the browser tab, or leave without clicking End Session.
- **Upload job tracking:** Return a `job_id` for document uploads and show indexing states such as pending, indexing, completed, and failed.
- **Page level citations:** Store page number, chunk index, retrieval score, and source metadata so answers can cite exact document locations.
- **Document management UI:** Add controls to view uploaded documents, show chunk counts, delete individual documents, and clear all session data.
- **OCR fallback:** Add optional OCR support for scanned PDFs using an environment flag to keep the default deployment lightweight.
- **Optional reranking:** Enable CrossEncoder reranking behind a feature flag to improve retrieval precision when memory allows.
- **Hybrid retrieval:** Combine dense semantic retrieval with sparse keyword based retrieval for better matching on exact terms and short factual queries.
- **Query rewriting:** Generate multiple retrieval friendly query variants to improve recall before answer generation.
- **RAG evaluation dataset:** Add a golden dataset to measure faithfulness, answer relevance, context precision, citation accuracy, and refusal correctness.
- **Observability metrics:** Track extraction, chunking, embedding, Pinecone upsert, retrieval, LLM generation, and total query latency.
- **Production security controls:** Add file validation, upload limits, query limits, rate limiting, namespace TTL cleanup, and authentication for non demo deployments.

---

## Portfolio Value

This project demonstrates practical AI engineering skills across:

- Full stack AI application development
- Retrieval Augmented Generation
- Vector database integration
- Semantic search
- Document ingestion
- Backend API design
- Streamlit frontend development
- Docker based deployment
- Hugging Face Spaces deployment
- Session isolation
- RAG evaluation planning
- Production oriented system design

---

## License

MIT
