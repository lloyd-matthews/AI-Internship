"""Pinecone vector store. Ingest and query both embed with text-embedding-3-small."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Same folder as main.py, so a script run from another cwd still finds .env.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Default output of text-embedding-3-small. Do not pass a shorter `dimensions`
# value: ingest and query must write and search the same vector size.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

_UPSERT_BATCH = 100

_index = None
_index_key: tuple[str, str, str] | None = None
_openai: OpenAI | None = None


class PineconeConfigError(RuntimeError):
    """Required Pinecone settings are missing from the environment."""


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str
    index_name: str
    index_host: str
    namespace: str


def load_settings() -> PineconeSettings:
    """Read Pinecone settings from the environment. Never falls back to a hardcoded key."""

    missing = [
        name
        for name in ("PINECONE_API_KEY", "PINECONE_INDEX_NAME")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        joined = ", ".join(missing)
        raise PineconeConfigError(
            f"Missing environment variable(s): {joined}. "
            "Set them in .env locally and in the Render service environment."
        )
    return PineconeSettings(
        api_key=os.environ["PINECONE_API_KEY"].strip(),
        index_name=os.environ["PINECONE_INDEX_NAME"].strip(),
        index_host=os.environ.get("PINECONE_INDEX_HOST", "").strip(),
        namespace=os.environ.get("PINECONE_NAMESPACE", "").strip(),
    )


def _openai_client() -> OpenAI:
    global _openai
    if _openai is None:
        # Reads OPENAI_API_KEY from the environment.
        _openai = OpenAI()
    return _openai


def get_index():
    """Open the configured index. A host skips Pinecone's control-plane lookup."""

    global _index, _index_key
    settings = load_settings()
    # Cache on name + host only. The API key stays out of this tuple so a
    # debug print of module state cannot show it.
    cache_key = (settings.index_name, settings.index_host, settings.namespace)
    if _index is not None and _index_key == cache_key:
        return _index

    client = Pinecone(api_key=settings.api_key)
    if settings.index_host:
        index = client.index(name=settings.index_name, host=settings.index_host)
    else:
        index = client.index(name=settings.index_name)
    _index = index
    _index_key = cache_key
    return index


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed with text-embedding-3-small. Shared by upsert and query."""

    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Cannot embed an empty string")

    response = _openai_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def upsert_texts(
    items: list[tuple[str, str]],
    *,
    metadata: dict | None = None,
    namespace: str | None = None,
) -> int:
    """Embed (id, text) pairs and upsert them. Returns how many vectors were sent."""

    if not items:
        return 0

    settings = load_settings()
    ns = settings.namespace if namespace is None else namespace
    index = get_index()
    extra = metadata or {}
    written = 0

    for start in range(0, len(items), _UPSERT_BATCH):
        batch = items[start : start + _UPSERT_BATCH]
        vectors = embed_texts([text for _, text in batch])
        payload = [
            {
                "id": item_id,
                "values": values,
                "metadata": {**extra, "text": text},
            }
            for (item_id, text), values in zip(batch, vectors, strict=True)
        ]
        index.upsert(
            vectors=payload,
            namespace=ns,
            batch_size=_UPSERT_BATCH,
            show_progress=False,
        )
        written += len(payload)
    return written


def query_text(
    query: str,
    *,
    top_k: int = 5,
    namespace: str | None = None,
) -> list[dict]:
    """Embed the query with text-embedding-3-small and return the nearest matches."""

    settings = load_settings()
    ns = settings.namespace if namespace is None else namespace
    vector = embed_texts([query])[0]
    response = get_index().query(
        vector=vector,
        top_k=top_k,
        namespace=ns,
        include_metadata=True,
    )
    hits: list[dict] = []
    for match in response.matches:
        meta = dict(match.metadata or {})
        hits.append(
            {
                "id": match.id,
                "score": match.score,
                "text": meta.get("text"),
                "metadata": meta,
            }
        )
    return hits


def _namespace_vector_count(stats, namespace: str) -> int | None:
    """Pull one namespace's count. Pinecone labels the default namespace `__default__`."""

    namespaces = getattr(stats, "namespaces", None) or {}
    if not hasattr(namespaces, "get"):
        return None
    key = namespace or "__default__"
    summary = namespaces.get(key)
    if summary is None and namespace == "":
        summary = namespaces.get("")
    if summary is None:
        return None
    if isinstance(summary, dict):
        count = summary.get("vector_count")
    else:
        count = getattr(summary, "vector_count", None)
    return int(count) if count is not None else None


def check_pinecone() -> dict:
    """Confirm the configured index answers. The payload contains no secrets.

    Call this directly, or GET /health/pinecone. A successful describe_index_stats
    means the data plane is reachable. dimension_matches_model is false when the
    index was not created at 1536, which is the size text-embedding-3-small writes.
    """

    settings = load_settings()
    index = get_index()
    stats = index.describe_index_stats()
    dimension = stats.dimension
    return {
        "reachable": True,
        "index_name": settings.index_name,
        "host": settings.index_host or index.host,
        "namespace": settings.namespace,
        "embedding_model": EMBEDDING_MODEL,
        "index_dimension": dimension,
        "expected_dimension": EMBEDDING_DIMENSIONS,
        "dimension_matches_model": dimension == EMBEDDING_DIMENSIONS,
        "total_vector_count": stats.total_vector_count,
        "namespace_vector_count": _namespace_vector_count(stats, settings.namespace),
        "metric": getattr(stats, "metric", None),
    }


if __name__ == "__main__":
    import json
    import sys

    try:
        print(json.dumps(check_pinecone(), indent=2))
    except PineconeConfigError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
