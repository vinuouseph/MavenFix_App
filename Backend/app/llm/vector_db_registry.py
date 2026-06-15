"""
vector_db_registry.py
─────────────────────
Factory for VectorDB clients and text embedders.
Switch provider via VECTOR_DB_PROVIDER in .env:
  - "qdrant"  (default) → QdrantClient
  - "milvus"            → MilvusClient  (pymilvus)

Also exposes get_embedder() which returns a callable
  embed(texts: list[str]) -> list[list[float]]
using the same LLM API key.
"""

from __future__ import annotations

import logging
from typing import Callable

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── VectorDB client ───────────────────────────────────────────────────────────

def get_vector_db():
    """Return a connected VectorDB client based on VECTOR_DB_PROVIDER."""
    provider = settings.VECTOR_DB_PROVIDER.lower()

    if provider == "milvus":
        try:
            from pymilvus import MilvusClient
            return MilvusClient(uri=settings.MILVUS_URI)
        except ImportError:
            raise ImportError(
                "pymilvus is not installed. Run: pip install pymilvus"
            )

    # Default: Qdrant
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(
            host=settings.VECTOR_DB_HOST,
            port=settings.VECTOR_DB_PORT,
            prefer_grpc=True,
        )
    except ImportError:
        raise ImportError(
            "qdrant-client is not installed. Run: pip install qdrant-client"
        )


# ── Text embedder ─────────────────────────────────────────────────────────────

def get_embedder() -> Callable[[list[str]], list[list[float]]]:
    """
    Return a callable  embed(texts: list[str]) -> list[list[float]]
    Provider is controlled by EMBEDDING_PROVIDER in .env:
      - "openai"  (default) → langchain_openai.OpenAIEmbeddings
      - "ollama"            → langchain_ollama.OllamaEmbeddings
    Model and base URL are read from EMBEDDING_MODEL / EMBEDDING_BASE_URL.
    """
    provider  = settings.EMBEDDING_PROVIDER.lower()
    model     = settings.EMBEDDING_MODEL
    base_url  = settings.EMBEDDING_BASE_URL or None

    http_client_verify: bool = str(settings.http_client_verify).strip().lower() != "true"

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            embedder = OllamaEmbeddings(
                model=model,
                base_url=base_url or "http://localhost:11434",
            )
            return embedder.embed_documents
        except ImportError:
            raise ImportError(
                "langchain-ollama is not installed. Run: pip install langchain-ollama"
            )

    # Default: OpenAI
    try:
        import httpx
        from langchain_openai import OpenAIEmbeddings

        http_kwargs = {}
        if http_client_verify:
            http_kwargs = {
                "http_client": httpx.Client(verify=False),
                "http_async_client": httpx.AsyncClient(verify=False),
            }

        embedder = OpenAIEmbeddings(
            api_key=settings.llm_api_key,
            base_url=base_url,
            model=model,
            **http_kwargs,
        )
        return embedder.embed_documents
    except ImportError:
        raise ImportError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        )