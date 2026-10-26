"""Shared helpers: logging, collection-name sanitization, dates, and Milvus utilities."""

import logging
import re
from typing import Any, List, Optional

import pandas as pd
from pymilvus import MilvusClient


def timestamp_to_iso(ts: Any) -> Optional[str]:
    """Convert a pandas/duckdb timestamp into an ISO 8601 UTC string."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)) or ts is pd.NaT:
        return None
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def to_str_list(raw: Any) -> List[str]:
    """Normalize a list-like field (numpy array, list, None) into a list of stripped strings."""
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        return [str(raw).strip()] if str(raw).strip() else []
    return [str(v).strip() for v in raw if v is not None and str(v).strip()]


def format_author_name(name: str) -> str:
    """Convert a 'Last, First' author name (as stored by OAI-PMH sources) to 'First Last'.

    Handles multi-comma forms like 'Iovanna, Juan, L.' -> 'Juan L. Iovanna'.
    Names without a comma are returned unchanged.
    """
    parts = [p.strip() for p in name.split(",") if p.strip()]
    if len(parts) < 2:
        return name.strip()
    return " ".join([*parts[1:], parts[0]])


def format_authors_text(raw_authors: Any) -> str:
    """Format authors to 'First Author et al.' if multiple authors, or single author."""
    authors = [format_author_name(a) for a in to_str_list(raw_authors)]
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    return f"{authors[0]} et al."


def format_authors_field(raw_authors: Any) -> str:
    """Convert an authors list of 'Last, First' strings into a comma-separated 'First Last' string."""
    # Keep at max 20 authors to avoid bloating the database with giant list of names in papers from collaborations
    max_authors = 20
    return ", ".join(format_author_name(a) for a in to_str_list(raw_authors)[:max_authors])


def setup_logging(name: str, silence_http_libs: bool = False) -> logging.Logger:
    """Configure root logging and return a named logger.

    Set silence_http_libs=True in long-running ingestion scripts to keep
    HTTP INFO logs from breaking the progress bar.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if silence_http_libs:
        # Disable verbose HTTP logs from all underlying libraries
        for lib in ("httpx", "httpcore", "openai", "urllib3"):
            logging.getLogger(lib).setLevel(logging.WARNING)
            logging.getLogger(lib).propagate = False
    return logging.getLogger(name)


def sanitize_collection_name(collection_name: str) -> str:
    """Sanitize a Milvus collection name to alphanumeric characters and underscores."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", collection_name)


def normalize_iso_date(date_str: str, is_end: bool = False) -> str:
    """Normalize date strings (e.g. 'YYYY-MM-DD') into ISO 8601 UTC bounds."""
    date_str = date_str.strip()
    if len(date_str) == 10 and date_str.count("-") == 2:
        return f"{date_str}T23:59:59Z" if is_end else f"{date_str}T00:00:00Z"
    return date_str


def escape_milvus_like(value: str) -> str:
    """Escape a user value for safe use inside a Milvus LIKE expression."""
    return re.escape(value.strip()).replace(chr(92), "")


def ensure_vector_index(
    milvus_client: MilvusClient,
    collection_name: str,
    field_name: str = "vector",
    metric_type: str = "COSINE",
    index_type: str = "HNSW",
    logger: logging.Logger | None = None,
) -> None:
    """Create the vector index if the collection does not have one yet.

    Needed when the collection already exists (e.g. created by an earlier
    interrupted run) but was never indexed — Milvus refuses to load an
    unindexed collection.
    """
    logger = logger or logging.getLogger(__name__)
    indexed_fields = set()
    for index_name in milvus_client.list_indexes(collection_name=collection_name):
        info = milvus_client.describe_index(
            collection_name=collection_name, index_name=index_name
        )
        indexed_field = info.get("field_name")
        if indexed_field:
            indexed_fields.add(indexed_field)

    if field_name not in indexed_fields:
        logger.info(
            f"Creating {index_type} index on '{field_name}' for collection '{collection_name}'..."
        )
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name=field_name,
            metric_type=metric_type,
            index_type=index_type,
        )
        milvus_client.create_index(
            collection_name=collection_name, index_params=index_params
        )


def get_existing_ids(
    milvus_client: MilvusClient,
    collection_name: str,
    pk_field: str,
    batch_size: int = 1000,
    logger: logging.Logger | None = None,
) -> set:
    """Fetch all existing primary keys currently in a Milvus collection."""
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Scanning Milvus collection '{collection_name}' for existing IDs...")
    existing_ids = set()
    try:
        iterator = milvus_client.query_iterator(
            collection_name=collection_name,
            filter="",
            output_fields=[pk_field],
            batch_size=batch_size,
        )
        while True:
            batch = iterator.next()
            if not batch:
                break
            for entry in batch:
                pk = entry.get(pk_field) or entry.get("id")
                if pk is not None:
                    existing_ids.add(str(pk))
        iterator.close()
    except Exception as e:
        logger.warning(f"Could not fetch existing IDs (collection might be empty): {e}")
    logger.info(f"Found {len(existing_ids)} existing IDs in Milvus.")
    return existing_ids
