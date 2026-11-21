"""Pinecone store for the current document index.

The index embeds the chunk_text field with text-embedding-3-small. Ingest and
query both go through the documents API; this index does not accept vector upserts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone

# Same folder as main.py, so a script run from another cwd still finds .env.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Default output of text-embedding-3-small. Do not pass a shorter `dimensions`
# value: ingest and query must write and search the same vector size.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Character window for POST /ingest. Override with CHUNK_SIZE / CHUNK_OVERLAP
# or with chunk_size / chunk_overlap on a single request.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

_UPSERT_BATCH = 20
# Schema field the index embeds with text-embedding-3-small.
_TEXT_FIELD = "chunk_text"
_METADATA_VALUE = str | int | float | bool | list[str]

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


def _optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def resolve_chunking(chunk_size: int | None, chunk_overlap: int | None) -> tuple[int, int]:
    """Request values win, then CHUNK_SIZE / CHUNK_OVERLAP, then 800 / 100."""

    size = chunk_size if chunk_size is not None else _optional_int_env("CHUNK_SIZE")
    overlap = chunk_overlap if chunk_overlap is not None else _optional_int_env("CHUNK_OVERLAP")
    size = DEFAULT_CHUNK_SIZE if size is None else size
    overlap = DEFAULT_CHUNK_OVERLAP if overlap is None else overlap
    if size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater")
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return size, overlap


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split on paragraphs, lines, then words. Drop chunks that are only whitespace."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _safe_metadata(metadata: dict | None) -> dict[str, _METADATA_VALUE]:
    """Keep only values Pinecone will store. Nested objects are rejected."""

    if not metadata:
        return {}
    cleaned: dict[str, _METADATA_VALUE] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            cleaned[str(key)] = value
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            cleaned[str(key)] = value
            continue
        raise ValueError(
            f"metadata.{key} must be a string, number, boolean, or list of strings"
        )
    return cleaned


def _documents_namespace(namespace: str | None) -> str:
    """Document calls require a namespace. Blank means Pinecone's default."""

    settings = load_settings()
    chosen = settings.namespace if namespace is None else namespace
    return chosen.strip() or "__default__"


def _document_payload(item_id: str, text: str, meta: dict) -> dict:
    """One record. Pinecone embeds chunk_text; the other keys are stored metadata."""

    source = str(meta.get("source") or "")
    document_id = str(meta.get("document_id") or item_id)
    payload = {
        "_id": item_id,
        _TEXT_FIELD: text,
        "title": source or document_id,
        "document_id": document_id,
        "source": source,
    }
    for key, value in meta.items():
        if key in payload:
            continue
        payload[key] = value
    return payload


def _upsert_records(
    records: list[tuple[str, str, dict]],
    namespace: str | None,
) -> int:
    """Write chunks to the document index. Pinecone embeds chunk_text on write."""

    if not records:
        return 0

    ns = _documents_namespace(namespace)
    index = get_index()
    documents = [
        _document_payload(item_id, text, meta) for item_id, text, meta in records
    ]
    written = 0
    for start in range(0, len(documents), _UPSERT_BATCH):
        batch = documents[start : start + _UPSERT_BATCH]
        index.documents.upsert(namespace=ns, documents=batch)
        written += len(batch)
    return written


def _delete_document_chunks(doc_id: str, namespace: str) -> None:
    """Remove earlier chunks for this document. IDs are '{document_id}:{chunk_index}'."""

    index = get_index()
    try:
        matches = index.documents.list(namespace=namespace, prefix=f"{doc_id}:")
        ids = [entry.id for entry in matches]
    except Exception as exc:
        message = str(exc).lower()
        if "namespace" in message or "not found" in message:
            return
        raise
    if ids:
        index.documents.delete(namespace=namespace, ids=ids)


def upsert_texts(
    items: list[tuple[str, str]],
    *,
    metadata: dict | None = None,
    namespace: str | None = None,
) -> int:
    """Embed (id, text) pairs and upsert them. Returns how many vectors were sent."""

    extra = _safe_metadata(metadata)
    records = [(item_id, text, dict(extra)) for item_id, text in items]
    return _upsert_records(records, namespace)


def _require_embedding_dimension() -> None:
    """Refuse to write vectors the index cannot store."""

    stats = get_index().describe_index_stats()
    dimension = stats.dimension
    if dimension != EMBEDDING_DIMENSIONS:
        raise PineconeConfigError(
            f"Index dimension is {dimension}; {EMBEDDING_MODEL} writes "
            f"{EMBEDDING_DIMENSIONS}-dimensional vectors. Recreate the index "
            f"at dimension {EMBEDDING_DIMENSIONS} before ingesting."
        )


def ingest_document(
    text: str,
    document_id: str,
    *,
    source: str = "",
    metadata: dict | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Chunk text and upsert it. The index embeds chunk_text with text-embedding-3-small."""

    cleaned = text.strip()
    doc_id = document_id.strip()
    if not cleaned:
        raise ValueError("text is empty. Send a non-empty document body.")
    if not doc_id:
        raise ValueError("document_id is empty. Send a stable id for this document.")
    if not doc_id.isascii() or any(char.isspace() for char in doc_id):
        raise ValueError("document_id must be ASCII without spaces.")

    size, overlap = resolve_chunking(chunk_size, chunk_overlap)
    chunks = chunk_text(cleaned, chunk_size=size, chunk_overlap=overlap)
    if not chunks:
        raise ValueError("text produced no chunks.")

    extra = _safe_metadata(metadata)
    _require_embedding_dimension()
    source_value = source.strip()
    records = []
    for chunk_index, chunk in enumerate(chunks):
        records.append(
            (
                f"{doc_id}:{chunk_index}",
                chunk,
                {
                    **extra,
                    "document_id": doc_id,
                    "chunk_index": chunk_index,
                    "source": source_value,
                },
            )
        )

    namespace = _documents_namespace(None)
    _delete_document_chunks(doc_id, namespace)
    return _upsert_records(records, namespace)


def query_text(
    query: str,
    *,
    top_k: int = 5,
    namespace: str | None = None,
) -> list[dict]:
    """Search chunk_text. The index embeds the query with text-embedding-3-small."""

    from pinecone.models.documents.score_by import EmbedQuery

    ns = _documents_namespace(namespace)
    response = get_index().documents.search(
        namespace=ns,
        score_by=[EmbedQuery(field=_TEXT_FIELD, query=query)],
        top_k=top_k,
        include_fields=[_TEXT_FIELD, "document_id", "chunk_index", "source", "title"],
    )
    hits: list[dict] = []
    for match in response.matches:
        fields = match.to_dict()
        fields.pop("_id", None)
        fields.pop("_score", None)
        text = fields.pop(_TEXT_FIELD, None)
        hits.append(
            {
                "id": match.id,
                "score": match.score,
                "text": text,
                "metadata": fields,
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
