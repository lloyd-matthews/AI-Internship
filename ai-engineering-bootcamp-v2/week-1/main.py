"""Week 1 live demo — five stages in one file, built up live in class."""

import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from vector_store import PineconeConfigError, check_pinecone, ingest_document, query_text

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Used when a request omits model.
DEFAULT_MODEL = "gpt-6-luna"

# How many chunks POST /ask retrieves before generation.
RETRIEVE_TOP_K = 5

# Sent as the user message to the existing structured-output call.
GROUNDING_PROMPT = """Answer the question using only the context below.

Rules:
- Use only facts that appear in the context. Do not add outside knowledge.
- Cite the document_id of every chunk you use, in the answer, as [document_id].
- If the context is not enough to answer, refuse. Say that the context is insufficient, set confidence to 0, and set sources_needed to false. Do not guess.
- When the answer uses the context, set sources_needed to true.

Context:
{context}

Question:
{question}
"""

# Stage 5 — per-1K-token input/output USD.
# Luna rates are standard short-context list prices from the OpenAI pricing page
# (per 1M tokens: gpt-5.6-luna $0.20/$1.20, gpt-6-luna $0.10/$0.50).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    # gpt-4o is deprecated.
    # o3-mini is an obsolete model.
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-5.6-luna": (0.0002, 0.0012),
    "gpt-6-luna": (0.0001, 0.0005),
}


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class IngestRequest(BaseModel):
    """One document to chunk, embed, and store."""

    text: str | None = None
    document_id: str | None = None
    source: str | None = None  # Optional filename or other origin label.
    metadata: dict | None = None
    chunk_size: int | None = None  # Characters. Defaults to CHUNK_SIZE or 800.
    chunk_overlap: int | None = None  # Characters. Defaults to CHUNK_OVERLAP or 100.


class RetrieveMatch(BaseModel):
    """One retrieved chunk. score is cosine similarity from the index."""

    id: str
    score: float
    document_id: str | None = None
    chunk_index: int | None = None
    source: str | None = None
    text: str | None = None


class RetrieveResponse(BaseModel):
    """Top chunks for a question. No chat completion is called."""

    query: str
    matches: list[RetrieveMatch]


class IngestResponse(BaseModel):
    """What POST /ingest returns after a successful upsert."""

    document_id: str
    chunks_indexed: int
    status: str


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.
    top_k: int = Field(default=RETRIEVE_TOP_K, ge=1, le=20)


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    retrieved_chunk_ids: list[str]


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    # A model missing from the table, including deprecated gpt-4o, uses the default model's rates.
    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks so the model can cite document_id."""

    if not hits:
        return "(no context retrieved)"

    blocks: list[str] = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        document_id = meta.get("document_id") or "unknown"
        chunk_id = hit.get("id") or ""
        text = hit.get("text") or ""
        blocks.append(f"[chunk_id={chunk_id} document_id={document_id}]\n{text}")
    return "\n\n".join(blocks)


def build_grounding_prompt(question: str, hits: list[dict]) -> str:
    """Fill the grounding template. This string is the user message to the model."""

    return GROUNDING_PROMPT.format(context=build_context(hits), question=question)


def call_model_structured(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Retrieve top chunks, then answer with the existing structured-output path."""

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty.")

    try:
        hits = query_text(question, top_k=body.top_k)
    except PineconeConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieve failed: {exc}") from exc

    retrieved_chunk_ids = [hit["id"] for hit in hits if hit.get("id")]
    grounded_question = build_grounding_prompt(question, hits)
    model = body.model or DEFAULT_MODEL
    last_error: str | None = None

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            start = time.perf_counter()

            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    grounded_question, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    grounded_question, model
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )


# Linux:
# curl -s -X POST http://127.0.0.1:8000/ingest \
#   -H "Content-Type: application/json" \
#   -d '{"text":"Quarterly revenue grew.","document_id":"doc-1","source":"notes.txt"}'
#
# Windows PowerShell (curl.exe, so PowerShell does not call Invoke-WebRequest):
# curl.exe --% -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d "{\"text\":\"Quarterly revenue grew.\",\"document_id\":\"doc-1\",\"source\":\"notes.txt\"}"
@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk text, embed each chunk, and upsert it into Pinecone."""

    source = body.source
    metadata = dict(body.metadata or {})
    if source is None and metadata.get("source") is not None:
        source = str(metadata.pop("source"))

    try:
        chunks_indexed = ingest_document(
            body.text or "",
            body.document_id or "",
            source=source or "",
            metadata=metadata,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PineconeConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}") from exc

    return IngestResponse(
        document_id=(body.document_id or "").strip(),
        chunks_indexed=chunks_indexed,
        status="indexed",
    )


# Linux:
# curl -s "http://127.0.0.1:8000/debug/retrieve?q=How%20many%20days%20of%20annual%20leave"
#
# Windows PowerShell:
# curl.exe -s "http://127.0.0.1:8000/debug/retrieve?q=How%20many%20days%20of%20annual%20leave"
@app.get("/debug/retrieve")
def debug_retrieve(q: str = "") -> RetrieveResponse:
    """Embed q with the index model and return the top 5 chunks. Does not call the LLM."""

    question = q.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="q is empty. Pass a question, for example /debug/retrieve?q=annual+leave",
        )

    try:
        hits = query_text(question, top_k=5)
    except PineconeConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieve failed: {exc}") from exc

    matches: list[RetrieveMatch] = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        chunk_index = meta.get("chunk_index")
        if isinstance(chunk_index, float):
            chunk_index = int(chunk_index)
        matches.append(
            RetrieveMatch(
                id=hit["id"],
                score=float(hit["score"] or 0.0),
                document_id=meta.get("document_id"),
                chunk_index=chunk_index if isinstance(chunk_index, int) else None,
                source=meta.get("source"),
                text=hit.get("text"),
            )
        )
    return RetrieveResponse(query=question, matches=matches)


@app.get("/health/pinecone")
def pinecone_health() -> dict:
    """Debug probe: describe the configured Pinecone index. Response has no secrets."""

    try:
        return check_pinecone()
    except PineconeConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Pinecone unreachable: {exc}") from exc
