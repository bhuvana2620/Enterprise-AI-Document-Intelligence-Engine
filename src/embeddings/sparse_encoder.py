import re
import hashlib
from collections import Counter


class NativeSparseEncoder:
    """
    Native token-frequency sparse encoder.

    Converts text into Pinecone-compatible sparse vector format:
    {
        "indices": [token_index_1, token_index_2, ...],
        "values": [token_weight_1, token_weight_2, ...]
    }

    This is used for the sparse/keyword side of hybrid search.
    """

    def __init__(self):
        # Tokenizes words with at least 2 characters.
        # Example: "AI-powered search!" -> ["ai", "powered", "search"]
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def tokenize(self, text: str) -> list[str]:
        """
        Convert input text into normalized lowercase tokens.
        """
        if not text:
            return []

        return self.token_pattern.findall(text.lower())

    def stable_hash(self, token: str) -> int:
        """
        Convert a token into a stable integer index.

        Important:
        Do NOT use Python's built-in hash() here because it may change
        between interpreter sessions. Sparse vector indices must remain
        stable between indexing time and query time.
        """
        return int(
            hashlib.md5(token.encode("utf-8")).hexdigest(),
            16
        ) % (2**31 - 1)

    def encode_text(self, text: str) -> dict:
        """
        Convert a block of text into Pinecone-ready sparse vector format.

        Output format:
        {
            "indices": [stable_token_hash_1, stable_token_hash_2],
            "values": [term_frequency_1, term_frequency_2]
        }
        """

        tokens = self.tokenize(text)

        if not tokens:
            return {
                "indices": [],
                "values": []
            }

        counts = Counter(tokens)
        total_tokens = len(tokens)

        sparse_items = []

        for token, count in counts.items():
            token_index = self.stable_hash(token)

            # Term frequency:
            # If a word appears more often in this chunk, give it more weight.
            tf_score = count / total_tokens

            sparse_items.append((token_index, float(tf_score)))

        # Pinecone expects sparse indices to be sorted.
        sparse_items.sort(key=lambda item: item[0])

        return {
            "indices": [item[0] for item in sparse_items],
            "values": [item[1] for item in sparse_items]
        }


if __name__ == "__main__":
    encoder = NativeSparseEncoder()

    sample_text = "The university library incident happened inside the library."

    sparse_vector = encoder.encode_text(sample_text)

    print("===== TOKENS =====")
    print(encoder.tokenize(sample_text))

    print("\n===== SPARSE VECTOR =====")
    print(sparse_vector)

    print("\nTotal sparse dimensions used:", len(sparse_vector["indices"]))