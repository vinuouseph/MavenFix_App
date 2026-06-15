"""
fix_knowledge_store.py
───────────────────────
Persists successful LLM fixes (confidence ≥ 90/100) into a VectorDB so that
future identical/similar errors can be resolved faster.

Public API
──────────
  store_fix_if_confident(java_version, error, fix, confidence)
      Embeds the error text and upserts the document if confidence >= 90.

  search_similar_fix(error, threshold=0.90) -> str | None
      Embeds the query, searches for the nearest neighbour, and returns the
      stored fix text if similarity >= threshold; otherwise None.

Provider selection is driven by VECTOR_DB_PROVIDER in .env.
Collection / index name: VECTOR_DB_COLLECTION (default: "fix_knowledge").
"""

from __future__ import annotations

import logging
import re
import uuid

from app.core.config import settings
from app.llm.vector_db_registry import get_vector_db, get_embedder

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.VECTOR_DB_COLLECTION
VECTOR_DIM      = settings.EMBEDDING_DIMENSION  # e.g., 1536 or 3072 depending on model
CONFIDENCE_MIN  = 90             # only store fixes with score >= this
SIMILARITY_MIN  = 0.75           # cosine similarity threshold for retrieval


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers — Qdrant
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_qdrant_collection(client) -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    from qdrant_client.models import Distance, VectorParams
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(f"[fix_knowledge] Created Qdrant collection '{COLLECTION_NAME}'")


def _qdrant_store(client, vector: list[float], payload: dict) -> None:
    from qdrant_client.models import PointStruct
    _ensure_qdrant_collection(client)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload,
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    logger.info(f"[fix_knowledge] Stored fix in Qdrant (java={payload.get('java_version')})")


def _qdrant_search(client, vector: list[float]) -> tuple[str | None, float]:
    """Returns (fix_text, score) or (None, 0.0)."""
    _ensure_qdrant_collection(client)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=1,
        with_payload=True,
    )
    if not response.points:
        return None, 0.0
    top = response.points[0]
    return top.payload.get("fix"), top.score


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers — Milvus
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_milvus_collection(client) -> None:
    """Create the Milvus collection if it doesn't exist yet."""
    if client.has_collection(COLLECTION_NAME):
        return
    from pymilvus import DataType
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id",           DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("vector",       DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("java_version", DataType.VARCHAR, max_length=32)
    schema.add_field("error",        DataType.VARCHAR, max_length=2048)
    schema.add_field("fix",          DataType.VARCHAR, max_length=8192)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="COSINE", index_type="FLAT")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"[fix_knowledge] Created Milvus collection '{COLLECTION_NAME}'")


def _milvus_store(client, vector: list[float], payload: dict) -> None:
    _ensure_milvus_collection(client)
    data = {
        "id":           str(uuid.uuid4()),
        "vector":       vector,
        "java_version": payload.get("java_version", ""),
        "error":        payload.get("error", "")[:2048],
        "fix":          payload.get("fix", "")[:8192],
    }
    client.insert(collection_name=COLLECTION_NAME, data=[data])
    logger.info(f"[fix_knowledge] Stored fix in Milvus (java={data['java_version']})")


def _milvus_search(client, vector: list[float]) -> tuple[str | None, float]:
    _ensure_milvus_collection(client)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[vector],
        anns_field="vector",
        limit=1,
        output_fields=["fix"],
    )
    if not results or not results[0]:
        return None, 0.0
    top = results[0][0]
    # Milvus cosine similarity is in [0, 1] when metric_type=COSINE
    score = top.get("distance", 0.0)
    fix   = top.get("entity", {}).get("fix")
    return fix, score


# ═════════════════════════════════════════════════════════════════════════════
# Error text normalisation (strip file paths + line numbers for better matching)
# ═════════════════════════════════════════════════════════════════════════════

_PATH_RE   = re.compile(r"[\w/\\]+\.java")
_LINENO_RE = re.compile(r":\d+[,:]")  # e.g.  :42, or :42:

def _normalize_error(raw: str) -> str:
    """Strip file paths and line numbers so embeddings focus on the error semantics."""
    text = _PATH_RE.sub("<FILE>", raw)
    text = _LINENO_RE.sub("", text)
    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def store_fix_if_confident(
    java_version: str,
    error: str,
    fix: str,
    confidence: int,
) -> None:
    """
    Embed the error text and store the error+fix document in the VectorDB
    if confidence >= CONFIDENCE_MIN (90).

    Args:
        java_version: e.g. "17", "11", "21"
        error:        The raw compiler error message / description
        fix:          A summary of what was changed to fix the error
        confidence:   LLM-assigned confidence score 0–100
    """
    if confidence < CONFIDENCE_MIN:
        logger.debug(
            f"[fix_knowledge] Skipping store: confidence {confidence} < {CONFIDENCE_MIN}"
        )
        return

    if not error.strip() or not fix.strip():
        return

    try:
        normalized = _normalize_error(error)
        embedder  = get_embedder()
        vector    = embedder([normalized])[0]
        client    = get_vector_db()
        payload   = {"java_version": java_version, "error": error, "fix": fix}

        provider = settings.VECTOR_DB_PROVIDER.lower()
        if provider == "milvus":
            _milvus_store(client, vector, payload)
        else:
            _qdrant_store(client, vector, payload)

    except Exception as exc:
        # Never crash the main pipeline because of a VectorDB write failure
        logger.warning(f"[fix_knowledge] store_fix_if_confident failed: {exc}", exc_info=True)


def search_similar_fix(
    error: str,
    threshold: float = SIMILARITY_MIN,
) -> str | None:
    """
    Embed the error text, search the VectorDB for the most similar stored fix.

    Returns:
        The stored fix string if similarity >= threshold, else None.
    """
    if not error.strip():
        return None

    try:
        normalized = _normalize_error(error)
        logger.info(f"[fix_knowledge] Searching VDB for: {normalized[:100]}")
        embedder  = get_embedder()
        vector    = embedder([normalized])[0]
        client    = get_vector_db()

        provider = settings.VECTOR_DB_PROVIDER.lower()
        if provider == "milvus":
            fix, score = _milvus_search(client, vector)
        else:
            fix, score = _qdrant_search(client, vector)

        if fix and score >= threshold:
            logger.info(
                f"[fix_knowledge] Found similar fix (score={score:.3f} >= {threshold})"
            )
            return fix

        logger.info(
            f"[fix_knowledge] No fix above threshold "
            f"(best score={score:.3f}, threshold={threshold})"
        )
        return None

    except Exception as exc:
        logger.warning(f"[fix_knowledge] search_similar_fix failed: {exc}", exc_info=True)
        return None
