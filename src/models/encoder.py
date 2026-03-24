"""Voyage AI encoder - cross-lingual sentence embeddings + consistency scoring.

Replaces local LaBSE. Free tier: 200M tokens (one-time).
API docs: https://docs.voyageai.com/reference/embeddings-api
"""

import numpy as np
from itertools import combinations
import requests
import config


def encode(texts: list[str]) -> np.ndarray:
    """Encode texts into embeddings via Voyage AI API.

    Args:
        texts: List of texts (can be in different languages).

    Returns:
        Numpy array of shape (n, embedding_dim).
    """
    resp = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.VOYAGE_API_KEY}",
        },
        json={
            "model": config.VOYAGE_EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    embeddings = [item["embedding"] for item in data["data"]]
    return np.array(embeddings, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def consistency_score(texts: list[str]) -> tuple[float, list[dict]]:
    """Compute cross-lingual consistency score.

    Calculates mean cosine similarity across all C(n,2) pairs.

    Args:
        texts: List of responses in different languages.

    Returns:
        (mean_score, pair_details) where pair_details is a list of
        {"pair": (i, j), "similarity": float}.
    """
    embeddings = encode(texts)
    n = len(texts)

    pair_details = []
    for i, j in combinations(range(n), 2):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        pair_details.append({"pair": (i, j), "similarity": sim})

    mean_score = float(np.mean([p["similarity"] for p in pair_details]))
    return mean_score, pair_details
