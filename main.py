from pathlib import Path
from typing import Any, List

import streamlit as st

from backend.core import retrieve_docs, stream_answer

APP_DIR = Path(__file__).resolve().parent
LOGO = str(APP_DIR / "static" / "Trimmed Padded Langchain.png")
SUGGESTIONS = [
    "What is LangChain?",
    "How do I create an agent?",
    "What is a retriever?",
    "What are deep agents?",
]


def _inject_css() -> None:
    css_path = APP_DIR / "static" / "app.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def _format_sources(context_docs: List[Any]) -> List[str]:
    seen: List[str] = []
    for doc in context_docs or []:
        source = str((getattr(doc, "metadata", None) or {}).get("source") or "").strip()
        if source and source not in seen:
            seen.append(source)
    return seen


def _render_sources(sources: List[str]) -> None:
    if not sources:
        return
    links = " · ".join(
        f'<a href="{src}" target="_blank" rel="noopener noreferrer">{src}</a>'
        for src in sources
    )
    st.markdown(
        f'<div class="sources"><strong>Sources</strong><br/>{links}</div>',
        unsafe_allow_html=True,
    )


def _run_turn(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=LOGO):
        try:
            with st.spinner("Searching documentation…"):
                context_docs = retrieve_docs(prompt)
            answer = st.write_stream(stream_answer(prompt, context_docs))
            answer = str(answer or "").strip() or "I could not generate an answer."
            sources = _format_sources(context_docs)
            _render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as exc:
            st.error("Something went wrong while generating a response.")
            st.exception(exc)


st.set_page_config(
    page_title="LangChain Docs",
    page_icon=LOGO,
    layout="centered",
    initial_sidebar_state="expanded",
)
_inject_css()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.image(LOGO, width=44)
    st.markdown("### LangChain Docs")
    st.caption("Documentation assistant")
    st.write("")
    st.caption(
        "Ask questions about the official documentation. Answers are retrieved from indexed LangChain pages."
    )
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Model · Groq gpt-oss-20b")
    st.caption("Embeddings · BGE large 1024-d")

if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero">
          <h1>How can I help you today?</h1>
          <p>Ask anything about LangChain documentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(suggestion, use_container_width=True, key=f"suggest-{i}"):
                st.session_state["queued_prompt"] = suggestion
                st.rerun()
else:
    for msg in st.session_state.messages:
        avatar = LOGO if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            _render_sources(msg.get("sources") or [])

prompt = st.chat_input("Message LangChain Docs")
queued = st.session_state.pop("queued_prompt", None)
if queued:
    prompt = queued
if prompt:
    _run_turn(prompt)
