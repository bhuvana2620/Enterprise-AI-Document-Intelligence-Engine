# app.py

import os
import sys
import json
import uuid
import hashlib
import time
import random
from pathlib import Path
from io import BytesIO

import requests
import streamlit as st


# ---------------------------------------------------
# Dynamic Workspace Path Injection
# ---------------------------------------------------
root_dir = str(Path(__file__).resolve().parent)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


# ---------------------------------------------------
# Backend API Configuration
# ---------------------------------------------------
def resolve_api_base_url() -> str:
    """
    Resolve backend API URL across environments.

    Local:
      API_BASE_URL=http://127.0.0.1:8000

    Docker Compose:
      API_BASE_URL=http://backend:8000

    Render:
      API_BASE_HOSTPORT is injected from backend private service hostport.
    """
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    api_base_hostport = os.getenv("API_BASE_HOSTPORT", "").strip()

    if api_base_url:
        return api_base_url.rstrip("/")

    if api_base_hostport:
        if api_base_hostport.startswith("http://") or api_base_hostport.startswith("https://"):
            return api_base_hostport.rstrip("/")

        return f"http://{api_base_hostport}".rstrip("/")

    return "http://127.0.0.1:8000"


API_BASE_URL = resolve_api_base_url()
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/upload"
CLEAR_SESSION_ENDPOINT = f"{API_BASE_URL}/api/v1/clear-session"
UPLOAD_SUCCESS_CODES = {200, 201, 202}

# Upload retry configuration.
# These are temporary failures that should be retried instead of shown immediately as errors.
RETRYABLE_UPLOAD_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_UPLOAD_RETRIES = int(os.getenv("MAX_UPLOAD_RETRIES", "4"))
UPLOAD_BASE_RETRY_DELAY_SECONDS = float(os.getenv("UPLOAD_BASE_RETRY_DELAY_SECONDS", "4"))

# ---------------------------------------------------
# Browser Session State
# ---------------------------------------------------
if "session_namespace" not in st.session_state:
    st.session_state.session_namespace = f"current-session-{uuid.uuid4().hex}"

if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "indexed_file_hashes" not in st.session_state:
    st.session_state.indexed_file_hashes = set()

if "active_category" not in st.session_state:
    st.session_state.active_category = None

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = f"uploader-{uuid.uuid4().hex[:8]}"

if "last_upload_summary" not in st.session_state:
    st.session_state.last_upload_summary = None


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def normalize_category(value: str) -> str:
    """
    Normalize category text so user-friendly inputs like:
    - work authorization
    - work_authorization
    - work-authorization

    all become:
    - work_authorization
    """
    if not value:
        return ""

    normalized = value.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized


def infer_category_from_filename(filename: str) -> str:
    """
    Gives a sensible default category based on filename.
    User can override it before indexing.
    """
    normalized = filename.lower()

    if "resume" in normalized or "cv" in normalized:
        return "resume"

    if (
        "ead" in normalized
        or "i-766" in normalized
        or "766" in normalized
        or "opt" in normalized
        or "work authorization" in normalized
    ):
        return "work_authorization"

    if "policy" in normalized:
        return "policy"

    if "form" in normalized:
        return "form"

    return "uploaded_document"


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Computes a stable hash for uploaded file content.
    Used to prevent re-indexing the exact same file in the same browser session.
    """
    return hashlib.sha256(file_bytes).hexdigest()



def get_upload_retry_wait_seconds(response, attempt: int) -> float:
    """
    Determine wait time before retrying an upload.

    Uses Retry-After header if provided.
    Otherwise uses exponential backoff with jitter.
    """
    retry_after = response.headers.get("Retry-After")

    if retry_after:
        try:
            return min(60.0, float(retry_after))
        except ValueError:
            pass

    return min(
        45.0,
        UPLOAD_BASE_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
    ) + random.uniform(0, 2.0)


def upload_file_with_retry(
    uploaded_file,
    category: str,
    namespace: str,
    file_bytes: bytes
):
    """
    Upload a file with automatic retry for temporary rate limits and gateway errors.

    This prevents the UI from immediately showing raw errors like:
    - 429 Too Many Requests
    - 502 Bad Gateway
    - 503 Service Unavailable
    - 504 Gateway Timeout
    """
    content_type = uploaded_file.type or "application/octet-stream"
    last_response = None

    for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
        file_obj = BytesIO(file_bytes)

        files = {
            "file": (
                uploaded_file.name,
                file_obj,
                content_type
            )
        }

        data = {
            "category": category,
            "namespace": namespace
        }

        try:
            response = requests.post(
                UPLOAD_ENDPOINT,
                files=files,
                data=data,
                timeout=600
            )
        finally:
            file_obj.close()

        if response.status_code not in RETRYABLE_UPLOAD_STATUS_CODES:
            return response

        last_response = response

        if attempt < MAX_UPLOAD_RETRIES:
            wait_seconds = get_upload_retry_wait_seconds(response, attempt)

            st.warning(
                f"`{uploaded_file.name}` upload was temporarily rate-limited "
                f"or the backend was busy. Retrying in {wait_seconds:.1f} seconds "
                f"({attempt + 1}/{MAX_UPLOAD_RETRIES})..."
            )

            time.sleep(wait_seconds)

    return last_response


def reset_browser_session():
    """
    Reset Streamlit-side browser session state and create a fresh namespace.
    """
    st.session_state.session_namespace = f"current-session-{uuid.uuid4().hex}"
    st.session_state.upload_counter = 0
    st.session_state.uploaded_documents = []
    st.session_state.indexed_file_hashes = set()
    st.session_state.active_category = None
    st.session_state.file_uploader_key = f"uploader-{uuid.uuid4().hex[:8]}"
    st.session_state.last_upload_summary = None


# ---------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Enterprise Document Intelligence Engine",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise AI Document Intelligence Engine")
st.markdown(
    "##### Production-grade RAG pipeline with Pinecone retrieval, CrossEncoder reranking, citations, and streaming generation."
)
st.markdown("---")


# ---------------------------------------------------
# Sidebar: Pipeline Configurations
# ---------------------------------------------------
st.sidebar.header("Pipeline Configurations")

top_k = st.sidebar.slider(
    "Top-K Retrieved Contexts",
    min_value=1,
    max_value=5,
    value=3
)

category_filter = st.sidebar.text_input(
    "Filter by Category During Query (Optional)",
    placeholder="e.g., resume, work authorization, work_authorization, policy"
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend API: `{API_BASE_URL}`")

st.sidebar.markdown("### Private Session Data")
st.sidebar.write(f"**Current Namespace:** `{st.session_state.session_namespace}`")

st.sidebar.caption(
    "Uploaded document vectors stay available only during this browser session. "
    "Use the button below when you are done to delete this session's vectors from Pinecone."
)

confirm_clear_session = st.sidebar.checkbox(
    "I understand this will permanently delete this session's vectors from Pinecone."
)

if st.sidebar.button(
    "End Session & Delete Pinecone Data",
    disabled=not confirm_clear_session
):
    current_namespace = st.session_state.session_namespace

    try:
        response = requests.post(
            CLEAR_SESSION_ENDPOINT,
            json={"namespace": current_namespace},
            timeout=120
        )

        if response.status_code == 200:
            st.sidebar.success(
                f"Deleted session namespace `{current_namespace}` from Pinecone."
            )

            reset_browser_session()
            st.rerun()

        else:
            st.sidebar.error("Failed to delete session namespace from Pinecone.")
            st.sidebar.code(response.text)

    except Exception as e:
        st.sidebar.error(f"Clear session error: {str(e)}")

if st.session_state.uploaded_documents:
    st.sidebar.success(f"{len(st.session_state.uploaded_documents)} document(s) indexed")

    with st.sidebar.expander("Uploaded Documents", expanded=True):
        for doc in st.session_state.uploaded_documents:
            st.write(
                f"**{doc['number']}. {doc['filename']}**  \n"
                f"Category: `{doc['category']}`  \n"
                f"Chunks: `{doc.get('upserted_count', 'N/A')}`  \n"
                f"Hash: `{doc.get('file_hash', 'N/A')}`"
            )
else:
    st.sidebar.warning("No documents indexed yet")


# ---------------------------------------------------
# Document Upload Section
# ---------------------------------------------------
st.subheader("📤 Upload & Index Documents")

with st.expander("Upload documents into this browser session", expanded=True):
    st.caption(
        "You can upload one or more PDF/TXT/DOCX files at once. "
        "All selected files will be indexed into the same browser-session namespace. "
        "Duplicate files in the same session will be skipped automatically."
    )

    uploaded_files = st.file_uploader(
        "Choose one or more PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key=st.session_state.file_uploader_key
    )

    st.info(
        f"Selected files will be indexed into browser-session namespace: "
        f"`{st.session_state.session_namespace}`"
    )

    # Show last upload result after rerun.
    if st.session_state.last_upload_summary:
        summary = st.session_state.last_upload_summary

        if summary.get("successful_uploads"):
            st.success(
                f"Accepted {len(summary['successful_uploads'])} document(s) for indexing."
            )

        if summary.get("skipped_uploads"):
            st.warning(
                f"Skipped {len(summary['skipped_uploads'])} duplicate document(s)."
            )

        if summary.get("failed_uploads"):
            st.error(
                f"{len(summary['failed_uploads'])} document(s) failed to upload."
            )

        with st.expander("Last Upload Details", expanded=False):
            st.json(summary)

    file_category_map = {}

    if uploaded_files:
        st.markdown("#### Document Categories")

        for idx, file in enumerate(uploaded_files, start=1):
            default_category = infer_category_from_filename(file.name)

            category = st.text_input(
                f"Category for {idx}. {file.name}",
                value=default_category,
                key=f"category_{st.session_state.file_uploader_key}_{idx}_{file.name}",
                placeholder="e.g., resume, work_authorization, policy"
            )

            file_category_map[idx] = normalize_category(category) or "uploaded_document"

    if st.button("Index Uploaded Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document before indexing.")
            st.stop()

        target_namespace = st.session_state.session_namespace

        successful_uploads = []
        skipped_uploads = []
        failed_uploads = []

        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        total_files = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            category = file_category_map.get(idx, "uploaded_document")
            file_bytes = uploaded_file.getvalue()
            file_hash = compute_file_hash(file_bytes)

            # Skip exact same file content if already indexed in this browser session.
            if file_hash in st.session_state.indexed_file_hashes:
                skipped_uploads.append({
                    "filename": uploaded_file.name,
                    "category": category,
                    "file_hash": file_hash[:12],
                    "reason": "Already indexed in this browser session"
                })

                progress_bar.progress(idx / total_files)
                continue

            status_placeholder.info(
                f"Indexing {idx}/{total_files}: `{uploaded_file.name}` "
                f"as category `{category}`..."
            )

            try:
                response = upload_file_with_retry(
                    uploaded_file=uploaded_file,
                    category=category,
                    namespace=target_namespace,
                    file_bytes=file_bytes
                )

                if response.status_code in UPLOAD_SUCCESS_CODES:
                    result = response.json()

                    st.session_state.upload_counter += 1
                    st.session_state.indexed_file_hashes.add(file_hash)

                    uploaded_doc_record = {
                        "number": st.session_state.upload_counter,
                        "filename": uploaded_file.name,
                        "category": category,
                        "namespace": target_namespace,
                        "file_hash": file_hash[:12],
                        "upserted_count": result.get("result", {}).get("upserted_count")
                    }

                    st.session_state.uploaded_documents.append(uploaded_doc_record)
                    successful_uploads.append(uploaded_doc_record)

                else:
                    error_text = response.text

                    if response.status_code == 429:
                        error_text = (
                            "Upload was temporarily rate-limited even after automatic retries. "
                            "Please wait a moment and try again."
                        )

                    elif response.status_code in {502, 503, 504}:
                        error_text = (
                            "The backend was temporarily unavailable during upload even after retries. "
                            "Please wait a moment and try again."
                        )

                    failed_uploads.append({
                        "filename": uploaded_file.name,
                        "category": category,
                        "status_code": response.status_code,
                        "error": error_text
                    })

            except Exception as e:
                failed_uploads.append({
                    "filename": uploaded_file.name,
                    "category": category,
                    "status_code": "exception",
                    "error": str(e)
                })

            progress_bar.progress(idx / total_files)
            time.sleep(1)

        status_placeholder.empty()

        st.session_state.last_upload_summary = {
            "namespace": target_namespace,
            "successful_uploads": successful_uploads,
            "skipped_uploads": skipped_uploads,
            "failed_uploads": failed_uploads
        }

        # Clear selected files from uploader after successful/duplicate-only processing.
        # If there are failures, keep the widget populated so user can retry.
        if successful_uploads or skipped_uploads:
            st.session_state.file_uploader_key = f"uploader-{uuid.uuid4().hex[:8]}"
            st.rerun()

        # If all files failed, show errors without clearing selected files.
        if failed_uploads:
            st.error(f"{len(failed_uploads)} document(s) failed to upload.")
            st.json(failed_uploads)


st.markdown("---")


# ---------------------------------------------------
# Query Section
# ---------------------------------------------------
st.subheader("🔎 Ask Questions from This Browser Session")

if st.session_state.uploaded_documents:
    st.caption(
        f"Current query namespace: `{st.session_state.session_namespace}`. "
        f"Searching across {len(st.session_state.uploaded_documents)} uploaded document(s)."
    )
else:
    st.caption("Upload and index at least one document before asking questions.")

user_query = st.text_input(
    "Enter your document intelligence query:",
    placeholder="What is the work authorization valid-until date?"
)


if st.button("Execute Pipeline Search", type="primary"):
    if not st.session_state.uploaded_documents:
        st.warning("Please upload and index at least one document before asking questions.")
        st.stop()

    if not user_query.strip():
        st.warning("Please enter a valid query before executing.")
        st.stop()

    citation_container = st.container()
    answer_header = st.empty()
    answer_text = st.empty()

    normalized_filter = normalize_category(category_filter)

    payload = {
        "query": user_query.strip(),
        "top_k": top_k,
        "category_filter": normalized_filter if normalized_filter else None,
        "namespace": st.session_state.session_namespace
    }

    try:
        with requests.post(
            QUERY_ENDPOINT,
            json=payload,
            stream=True,
            timeout=180
        ) as response:

            if response.status_code != 200:
                st.error(
                    f"Backend API Error: Received Status Code {response.status_code}"
                )
                st.code(response.text)

            else:
                compiled_answer = ""
                sources_displayed = False

                for line in response.iter_lines():
                    if not line:
                        continue

                    decoded_line = line.decode("utf-8")

                    if not decoded_line.startswith("data: "):
                        continue

                    raw_json = decoded_line.replace("data: ", "").strip()

                    try:
                        packet = json.loads(raw_json)

                    except json.JSONDecodeError:
                        continue

                    packet_type = packet.get("type")

                    # --------------------------------------------
                    # Packet Type A: Citation Metadata
                    # --------------------------------------------
                    if packet_type == "metadata":
                        sources = packet.get("sources", [])

                        with citation_container:
                            if sources:
                                st.markdown("### 📄 Retrieved Document Citations")

                                cols = st.columns(len(sources))

                                for idx, src in enumerate(sources):
                                    with cols[idx]:
                                        source_name = src.get("source", "Unknown Source")
                                        category = src.get("category", "Uncategorized")
                                        score = src.get("score", 0.0)

                                        st.info(
                                            f"**Source [{idx + 1}]:** {source_name}\n\n"
                                            f"**Category:** {category}\n\n"
                                            f"**Session Namespace:** `{st.session_state.session_namespace}`\n\n"
                                            f"**Rerank Confidence:** {score:.4f}"
                                        )
                            else:
                                st.caption(
                                    "⚠️ No matching document chunks were retrieved from this browser session."
                                )

                        sources_displayed = True

                    # --------------------------------------------
                    # Packet Type B: Streaming Content
                    # --------------------------------------------
                    elif packet_type == "content":
                        if sources_displayed and not compiled_answer:
                            answer_header.markdown(
                                "### ⚡ Synthesized Document Intelligence"
                            )

                        compiled_answer += packet.get("text", "")
                        answer_text.markdown(compiled_answer + "▌")

                    # --------------------------------------------
                    # Packet Type C: Backend Error
                    # --------------------------------------------
                    elif packet_type == "error":
                        st.error(
                            f"Stream generation error: {packet.get('detail')}"
                        )

                if compiled_answer:
                    answer_text.markdown(compiled_answer)

    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to backend query endpoint: `{QUERY_ENDPOINT}`. "
            "Please verify FastAPI is running."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The backend request timed out. The model or retriever may be taking too long."
        )

    except Exception as e:
        st.error(f"Unexpected frontend error: {str(e)}")