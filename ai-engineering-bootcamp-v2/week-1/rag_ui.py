"""Minimal Streamlit client for the live RAG API.

The FastAPI service owns ingest, retrieval, and generation.
This page only sends POST /ingest and POST /ask.

Run from ai-engineering-bootcamp-v2/week-1:

  $env:API_BASE_URL = "https://ai-internship-24cp.onrender.com"
  .venv/Scripts/python.exe -m streamlit run rag_ui.py
"""

import os

import httpx
import streamlit as st

DEFAULT_API_BASE_URL = "https://ai-internship-24cp.onrender.com"
TIMEOUT_SECONDS = 120.0

st.set_page_config(page_title="Northwind RAG", layout="centered")
st.title("Northwind RAG")
st.caption("Streamlit calls the live API. It does not embed, retrieve, or generate on its own.")

with st.sidebar:
    st.header("API")
    api_base_url = st.text_input(
        "API base URL",
        value=os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL),
        help="Set API_BASE_URL, or type the Render URL here. No API keys belong in this page.",
    ).strip().rstrip("/")
    st.caption("Uses POST /ingest and POST /ask on that host.")


def post_json(path: str, payload: dict) -> tuple[int, dict | str]:
    """Send one JSON request. The API response is returned unchanged."""
    if not api_base_url:
        return 0, "Enter the API base URL in the sidebar."
    try:
        response = httpx.post(f"{api_base_url}{path}", json=payload, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        return 0, f"Could not reach the API: {exc}"
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


def document_ids(chunk_ids: list[str]) -> list[str]:
    """chunk ids look like doc1_handbook:7. Keep order, drop duplicates."""
    seen: list[str] = []
    for chunk_id in chunk_ids:
        document_id = chunk_id.rsplit(":", 1)[0]
        if document_id and document_id not in seen:
            seen.append(document_id)
    return seen


def show_ask(status: int, body: dict | str) -> None:
    if not isinstance(body, dict) or "answer" not in body:
        st.error(f"API returned HTTP {status}")
        st.json(body)
        return

    answer = body["answer"] if isinstance(body["answer"], dict) else {"answer": str(body["answer"])}
    text = answer.get("answer", "")
    confidence = answer.get("confidence")
    sources_needed = answer.get("sources_needed")
    refused = sources_needed is False or confidence == 0

    if refused:
        st.error("Refusal. The API says the retrieved context is not enough to answer.")
    else:
        st.success("Answered from the retrieved documents.")

    st.markdown(text)

    chunk_ids = body.get("retrieved_chunk_ids") or []
    cited = document_ids(chunk_ids)
    if refused:
        st.write("No document was cited.")
    elif cited:
        st.write("Cited document_id: " + ", ".join(cited))
    if chunk_ids:
        st.write("Chunk IDs")
        st.code("\n".join(chunk_ids))

    details = {
        "confidence": confidence,
        "sources_needed": sources_needed,
        "model": body.get("model"),
        "tokens_used": body.get("tokens_used"),
        "cost_usd": body.get("cost_usd"),
    }
    st.caption(
        " · ".join(f"{key}={value}" for key, value in details.items() if value is not None)
    )
    with st.expander("Full JSON response"):
        st.json(body)


ingest_tab, ask_tab = st.tabs(["Ingest", "Ask"])

with ingest_tab:
    st.subheader("Add a document")
    with st.form("ingest"):
        ingest_text = st.text_area("Text", height=180, placeholder="Paste the document text")
        document_id = st.text_input("document_id", placeholder="handbook")
        ingest_submitted = st.form_submit_button("Call /ingest", type="primary")
    if ingest_submitted:
        status, body = post_json(
            "/ingest",
            {"text": ingest_text, "document_id": document_id},
        )
        if isinstance(body, dict) and status == 200:
            st.success(
                f"Indexed {body.get('chunks_indexed')} chunk(s) "
                f"for document_id {body.get('document_id')}."
            )
        else:
            st.error(f"API returned HTTP {status}")
        st.json(body)

with ask_tab:
    st.subheader("Ask a question")
    with st.form("ask"):
        question = st.text_input("Question", placeholder="How many days each year is annual leave?")
        ask_submitted = st.form_submit_button("Call /ask", type="primary")
    if ask_submitted:
        status, body = post_json("/ask", {"question": question})
        show_ask(status, body)
