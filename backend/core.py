import os
from typing import Any, Dict, Iterator, List

import torch
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _embedding_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": _embedding_device()},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
)
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME", "documentation-helper"),
    embedding=embedding,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq",
    max_tokens=512,
    max_retries=3,
    reasoning_effort="low",
)

MAX_CHARS_PER_DOC = 2500
BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)
SYSTEM_PROMPT = (
    "You are a helpful assistant for LangChain documentation. "
    "Use the retrieved context to answer. Combine related snippets into a clear explanation. "
    "If the context is only partly relevant, still answer what you can from it and cite sources. "
    "Say you don't know only if the context is unrelated to the question."
)


def _format_context(docs: List[Any]) -> str:
    return "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Content: {(doc.page_content or '')[:MAX_CHARS_PER_DOC]}"
        )
        for doc in docs
    )


def retrieve_docs(query: str) -> List[Any]:
    return retriever.invoke(f"{BGE_QUERY_PREFIX}{query}")


def _messages(query: str, docs: List[Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{_format_context(docs)}\n\nQuestion: {query}",
        },
    ]


def _chunk_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
        return "".join(parts)
    return ""


def stream_answer(query: str, docs: List[Any]) -> Iterator[str]:
    for chunk in model.stream(_messages(query, docs)):
        text = _chunk_text(getattr(chunk, "content", ""))
        if text:
            yield text


def run_llm(query: str) -> Dict[str, Any]:
    """Retrieve docs, then answer in a single Groq call."""
    retrieved_docs = retrieve_docs(query)
    answer = "".join(stream_answer(query, retrieved_docs))
    return {
        "answer": answer,
        "context": retrieved_docs,
    }


if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")
    print(result)
