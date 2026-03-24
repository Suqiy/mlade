"""Wikipedia Retriever - two-stage retrieval.

Stage A (Coarse): Wikipedia API finds relevant articles.
Stage B (Fine):   Jina Embeddings V3 finds most relevant passages per claim.
"""

import os
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import config


_cache_dir = None


def _get_cache_dir() -> str:
    global _cache_dir
    if _cache_dir is None:
        _cache_dir = os.path.join(config.WIKI_INDEX_DIR, "api_cache")
        os.makedirs(_cache_dir, exist_ok=True)
    return _cache_dir


# ============================================================
# Wikipedia API helpers
# ============================================================

def _wiki_api(params: dict) -> dict:
    """Call Wikipedia API."""
    params["format"] = "json"
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "MLADE/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _search_titles(query: str, limit: int = 5) -> list[str]:
    """Search Wikipedia for article titles matching the query."""
    data = _wiki_api({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
    })
    return [result["title"] for result in data.get("query", {}).get("search", [])]


def _get_article_text(title: str) -> str | None:
    """Fetch the plain text of a Wikipedia article (up to 10000 chars)."""
    data = _wiki_api({
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": "",
        "explaintext": "",
        "exlimit": 1,
        "exchars": 10000,
    })
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id != "-1":
            return page.get("extract", "")
    return None


def _split_into_passages(text: str, title: str, window: int = 3) -> list[dict]:
    """Split article into sentence-level passages with sliding window."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return []

    passages = []
    for i in range(len(sentences)):
        chunk = " ".join(sentences[i:i + window])
        if len(chunk) > 30:
            passages.append({"title": title, "text": chunk})

    return passages


# ============================================================
# Jina Embeddings V3 API
# ============================================================

def _jina_embed(texts: list[str], task: str = "retrieval.passage") -> np.ndarray:
    """Call Jina Embeddings V3 API.

    Args:
        texts: List of texts to embed.
        task: "retrieval.query" for queries/claims, "retrieval.passage" for passages.

    Returns:
        Numpy array of shape (n, 1024).
    """
    import requests as req_lib

    resp = req_lib.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.JINA_API_KEY}",
        },
        json={
            "model": config.JINA_EMBEDDING_MODEL,
            "task": task,
            "input": texts,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    embeddings = [item["embedding"] for item in data["data"]]
    return np.array(embeddings, dtype=np.float32)


# ============================================================
# Cache helpers
# ============================================================

def _load_cache(query: str) -> list[dict] | None:
    cache_file = os.path.join(_get_cache_dir(), urllib.parse.quote(query, safe="") + ".json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(query: str, results: list[dict]):
    cache_file = os.path.join(_get_cache_dir(), urllib.parse.quote(query, safe="") + ".json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


# ============================================================
# Public API
# ============================================================

def search(query: str, top_k: int | None = None) -> list[dict]:
    """Coarse retrieval: fetch relevant Wikipedia articles, split into passages."""
    if top_k is None:
        top_k = config.WIKI_TOP_K

    cached = _load_cache(query)
    if cached is not None:
        print(f"  Retrieved {len(cached)} passages from cache")
        return cached

    start_time = time.time()

    titles = _search_titles(query, limit=top_k)
    if not titles:
        print("  No Wikipedia articles found.")
        return []

    all_passages = []
    for title in titles:
        text = _get_article_text(title)
        if text:
            passages = _split_into_passages(text, title)
            all_passages.extend(passages)

    _save_cache(query, all_passages)

    elapsed = time.time() - start_time
    print(f"  Retrieved {len(all_passages)} passages from {len(titles)} articles in {elapsed:.3f}s")

    return all_passages


def find_relevant_passages(claim: str, passages: list[dict], top_k: int = 3) -> list[dict]:
    """Fine retrieval: use Jina V3 to find the most relevant passages for a claim.

    Uses separate task types for query vs passage (asymmetric retrieval).
    """
    if not passages:
        return []

    # Encode claim as query, passages as documents (different adapters)
    claim_emb = _jina_embed([claim], task="retrieval.query")
    passage_texts = [p["text"] for p in passages]
    passage_embs = _jina_embed(passage_texts, task="retrieval.passage")

    # Cosine similarity (Jina embeddings are already normalized)
    similarities = np.dot(passage_embs, claim_emb.T).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "title": passages[idx]["title"],
            "text": passages[idx]["text"],
            "score": float(similarities[idx]),
        })

    return results
