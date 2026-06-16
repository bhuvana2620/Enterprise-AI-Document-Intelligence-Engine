# src/api/main.py

import os
import json
import time
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/ai-document-intelligence/uploads"))


def should_delete_uploaded_file_after_indexing() -> bool:
    return os.getenv("DELETE_UPLOADED_FILE_AFTER_INDEXING", "true").strip().lower() == "true"


def clear_local_upload_folder() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for path in UPLOAD_DIR.iterdir():
        try:
            if path.name == ".gitkeep":
                continue
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except Exception as e:
            print(f"[WARN] Could not delete upload artifact '{path}': {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from src.embeddings.embedding_generator import preload_embedding_model
        preload_embedding_model()
    except Exception as e:
        print(f"[WARN] Embedding model preload failed: {e}", flush=True)
    yield


app = FastAPI(
    title="AI Document Intelligence API",
    description="FastAPI backend for streaming RAG-based document intelligence.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    category_filter: Optional[str] = None
    namespace: Optional[str] = None


class ClearSessionRequest(BaseModel):
    namespace: str = Field(..., min_length=1)


def sse_packet(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_text(answer: str, delay: float = 0.02):
    for token in answer.split(" "):
        yield sse_packet({"type": "content", "text": token + " "})
        time.sleep(delay)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ai-document-intelligence-api", "version": "1.0.0"}


@app.get("/")
def root():
    return {"message": "AI Document Intelligence API is running.", "docs": "/docs", "health": "/health"}


@app.post("/api/v1/query")
def query_documents(request: QueryRequest):
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def stream_response():
        try:
            from src.retrieval.retriever import retrieve_chunks
            from src.generation.prompt_builder import build_prompt
            from src.generation.llm_client import generate_answer

            retrieved_chunks = retrieve_chunks(
                query=user_query,
                category_filter=request.category_filter,
                top_k=request.top_k,
                namespace=request.namespace
            )

            sources = [{
                "source": c.get("source", "Unknown Source"),
                "category": c.get("category", "Uncategorized"),
                "score": float(c.get("score", c.get("raw_score", 0.0)))
            } for c in retrieved_chunks]

            yield sse_packet({"type": "metadata", "sources": sources})

            if not retrieved_chunks:
                yield sse_packet({"type": "content", "text": "I could not find the answer in the provided documents."})
                return

            prompt = build_prompt(user_query, retrieved_chunks)
            answer = generate_answer(prompt)

            for packet in stream_text(answer):
                yield packet

        except Exception as e:
            yield sse_packet({"type": "error", "detail": str(e)})

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


def _run_indexing_in_background(
    temp_file_path: str,
    stored_filename: str,
    category: str,
    target_namespace: str,
    session_file_id: str,
) -> None:
    try:
        from src.vector_store.pinecone_store import index_document
        index_document(
            file_path=temp_file_path,
            category=category,
            namespace=target_namespace,
            source_name=stored_filename,
        )
        print(f"[BG] Indexed '{stored_filename}' into namespace '{target_namespace}'.", flush=True)
    except Exception as e:
        print(f"[BG] Indexing failed for '{stored_filename}': {e}", flush=True)
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"[WARN] Could not remove temp file: {e}", flush=True)

        if should_delete_uploaded_file_after_indexing():
            try:
                from src.storage.session_file_store import delete_session_file
                delete_session_file(session_file_id)
            except Exception as e:
                print(f"[WARN] Could not delete session file '{session_file_id}': {e}", flush=True)

        clear_local_upload_folder()


@app.post("/api/v1/upload", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(default="uploaded_document"),
    namespace: Optional[str] = Form(default=None),
):
    allowed_extensions = {".pdf", ".txt", ".docx"}
    original_filename = Path(file.filename or "uploaded_document").name
    file_ext = Path(original_filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Supported types: PDF, TXT, DOCX.",
        )

    target_namespace = namespace or os.getenv("PINECONE_NAMESPACE", "default")

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        from src.storage.session_file_store import save_session_file, materialize_session_file

        session_file_id = save_session_file(
            namespace=target_namespace,
            filename=original_filename,
            content_type=file.content_type,
            file_bytes=file_bytes,
        )
        temp_file_path, stored_filename = materialize_session_file(session_file_id)

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Required storage module import failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")

    background_tasks.add_task(
        _run_indexing_in_background,
        temp_file_path=temp_file_path,
        stored_filename=stored_filename,
        category=category,
        target_namespace=target_namespace,
        session_file_id=session_file_id,
    )

    return {
        "status": "accepted",
        "message": "File received. Indexing is running in the background.",
        "filename": original_filename,
        "category": category,
        "namespace": target_namespace,
        "file_id": session_file_id,
    }


@app.post("/api/v1/clear-session")
def clear_session(request: ClearSessionRequest):
    try:
        from src.vector_store.pinecone_store import clear_namespace
        from src.storage.session_file_store import clear_session_files

        result = clear_namespace(request.namespace)
        deleted_file_count = clear_session_files(request.namespace)
        clear_local_upload_folder()

        return {
            "status": "success",
            "message": f"Cleared namespace '{request.namespace}', session files, and local upload artifacts.",
            "deleted_session_files": deleted_file_count,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear session namespace: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )