# scripts/clear_all_pinecone_vectors.py

import os
import time
from pathlib import Path
from pinecone import Pinecone


def load_dotenv_manually(path=".env"):
    env_path = Path(path)

    if not env_path.exists():
        print("No .env file found.")
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


def get_namespaces(stats):
    if isinstance(stats, dict):
        return stats.get("namespaces", {}) or {}

    if hasattr(stats, "to_dict"):
        return stats.to_dict().get("namespaces", {}) or {}

    return getattr(stats, "namespaces", {}) or {}


def get_vector_count(namespace_info):
    if isinstance(namespace_info, dict):
        return namespace_info.get("vector_count", 0)

    return getattr(namespace_info, "vector_count", 0)


load_dotenv_manually()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-document-intelligence")

if not PINECONE_API_KEY:
    raise SystemExit("Missing PINECONE_API_KEY. Check your .env file.")

print(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

stats = index.describe_index_stats()
namespaces = get_namespaces(stats)

if not namespaces:
    print("No namespaces found. Pinecone index is already empty.")
    raise SystemExit(0)

print("\nNamespaces before cleanup:")
for namespace, info in namespaces.items():
    display_name = namespace if namespace else "<default>"
    print(f"- {display_name}: {get_vector_count(info)} vectors")

for namespace, info in namespaces.items():
    vector_count = get_vector_count(info)

    if vector_count <= 0:
        continue

    display_name = namespace if namespace else "<default>"
    print(f"\nDeleting all vectors from namespace: {display_name}")

    index.delete(
        delete_all=True,
        namespace=namespace
    )

print("\nWaiting for Pinecone to apply deletes...")
time.sleep(10)

after_stats = index.describe_index_stats()
after_namespaces = get_namespaces(after_stats)

print("\nNamespaces after cleanup:")
if not after_namespaces:
    print("No namespaces found.")
else:
    for namespace, info in after_namespaces.items():
        display_name = namespace if namespace else "<default>"
        print(f"- {display_name}: {get_vector_count(info)} vectors")

print("\nPinecone cleanup complete.")
