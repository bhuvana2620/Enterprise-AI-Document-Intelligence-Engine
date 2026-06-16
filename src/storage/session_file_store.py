# src/storage/session_file_store.py

import os
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = "/tmp/ai-document-intelligence/session_files.sqlite3"


def get_db_path() -> Path:
    db_path = Path(os.getenv("SESSION_FILE_DB_PATH", DEFAULT_DB_PATH))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection():
    conn = sqlite3.connect(get_db_path())
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_files (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            filename TEXT NOT NULL,
            content_type TEXT,
            file_ext TEXT,
            file_bytes BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_session_file(
    namespace: str,
    filename: str,
    content_type: Optional[str],
    file_bytes: bytes
) -> str:
    file_id = uuid.uuid4().hex
    file_ext = Path(filename).suffix.lower()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO session_files (
                id, namespace, filename, content_type, file_ext, file_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                namespace,
                filename,
                content_type,
                file_ext,
                sqlite3.Binary(file_bytes),
            ),
        )
        conn.commit()

    return file_id


def materialize_session_file(file_id: str) -> tuple[str, str]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT filename, file_ext, file_bytes
            FROM session_files
            WHERE id = ?
            """,
            (file_id,),
        ).fetchone()

    if not row:
        raise ValueError(f"Session file not found: {file_id}")

    filename, file_ext, file_bytes = row

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_ext,
        prefix="doc_intel_upload_"
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    return temp_file_path, filename


def delete_session_file(file_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM session_files WHERE id = ?",
            (file_id,),
        )
        conn.commit()


def clear_session_files(namespace: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM session_files WHERE namespace = ?",
            (namespace,),
        )
        conn.commit()
        return cursor.rowcount


def clear_all_session_files() -> int:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM session_files")
        conn.commit()
        return cursor.rowcount