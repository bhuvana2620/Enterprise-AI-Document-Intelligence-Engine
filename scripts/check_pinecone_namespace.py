import os
from pathlib import Path
from pinecone import Pinecone


def load_dotenv_manually(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv_manually()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "ai-document-intelligence")

if not api_key:
    raise SystemExit("Missing PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

stats = index.describe_index_stats()

if hasattr(stats, "to_dict"):
    stats = stats.to_dict()

print("Index:", index_name)
print("Namespaces:")

namespaces = stats.get("namespaces", {})

if not namespaces:
    print("No namespaces found.")
else:
    for namespace, info in namespaces.items():
        print(f"- {namespace}: {info.get('vector_count', 0)} vectors")
