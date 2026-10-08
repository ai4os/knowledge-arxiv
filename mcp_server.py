"""
MCP server to expose arXiv metadata collection from Milvus.
"""

import argparse
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
import httpx
from mcp.server.mcpserver import MCPServer
from openai import OpenAI
from pymilvus import MilvusClient

load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mcp_server")

# Configuration constants
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "arxiv_metadata_v2")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "AI4EOSC/default-embeddings")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://vllm.cloud.ai4eosc.eu/v1")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", OPENAI_BASE_URL)
OPENAI_API_KEY = os.getenv("LITELLM_KEY")

# Initialize OpenAI client with retries
_http_client = httpx.Client(transport=httpx.HTTPTransport(retries=3))
openai_client = OpenAI(
    base_url=EMBEDDINGS_BASE_URL,
    api_key=OPENAI_API_KEY,
    http_client=_http_client,
)


def get_milvus_client() -> MilvusClient:
    """Initialize Milvus client from environment variables."""
    milvus_uri = os.getenv("MILVUS_URI", "./milvus.db")
    milvus_pwd = os.getenv("MILVUS_PWD")
    milvus_user = os.getenv("MILVUS_USER", "root")

    if milvus_uri.startswith("http://") or milvus_uri.startswith("https://") or ":" in milvus_uri:
        return MilvusClient(uri=milvus_uri, user=milvus_user, password=milvus_pwd)
    return MilvusClient(uri=milvus_uri)


milvus_client = get_milvus_client()

# Ensure collection name is sanitized and load if available
sanitized_collection_name = re.sub(r"[^a-zA-Z0-9_]", "_", COLLECTION_NAME)
try:
    if milvus_client.has_collection(collection_name=sanitized_collection_name):
        milvus_client.load_collection(collection_name=sanitized_collection_name)
        logger.info(f"Loaded collection '{sanitized_collection_name}' into memory.")
except Exception as e:
    logger.warning(f"Could not load collection '{sanitized_collection_name}': {e}")

# Initialize MCP server
mcp = MCPServer("arxiv-metadata-mcp")


def get_embedding(text: str) -> list[float]:
    """Generate vector embedding for the search query."""
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDINGS_MODEL,
    )
    return response.data[0].embedding


def normalize_iso_date(date_str: str, is_end: bool = False) -> str:
    """Normalize date strings (e.g. 'YYYY-MM-DD') into ISO 8601 UTC bounds."""
    date_str = date_str.strip()
    if len(date_str) == 10 and date_str.count("-") == 2:
        return f"{date_str}T23:59:59Z" if is_end else f"{date_str}T00:00:00Z"
    return date_str


def build_filter_expression(
    submitted: list[str | None] | None = None,
    updated: list[str | None] | None = None,
    categories: list[str] | None = None,
    has_doi: bool | None = None,
    authors: list[str] | None = None,
) -> str:
    """Construct a Milvus boolean filter expression for scalar fields."""
    clauses: list[str] = []

    # Handle submitted date range [start, end]
    if submitted:
        start = submitted[0] if len(submitted) >= 1 else None
        end = submitted[1] if len(submitted) >= 2 else None
        if start and str(start).strip():
            clauses.append(f'submitted >= "{normalize_iso_date(str(start), is_end=False)}"')
        if end and str(end).strip():
            clauses.append(f'submitted <= "{normalize_iso_date(str(end), is_end=True)}"')

    # Handle updated date range [start, end]
    if updated:
        start = updated[0] if len(updated) >= 1 else None
        end = updated[1] if len(updated) >= 2 else None
        if start and str(start).strip():
            clauses.append(f'updated >= "{normalize_iso_date(str(start), is_end=False)}"')
        if end and str(end).strip():
            clauses.append(f'updated <= "{normalize_iso_date(str(end), is_end=True)}"')

    # Handle has_doi boolean filter
    if has_doi is not None:
        clauses.append(f"has_doi == {'true' if has_doi else 'false'}")

    # Handle categories list filter (match any of the specified categories)
    if categories:
        if isinstance(categories, str):
            categories = [categories]
        cat_subclauses = [
            f'arxiv_categories like "%{re.escape(cat.strip()).replace(chr(92), "")}%"'
            for cat in categories
            if cat and cat.strip()
        ]
        if cat_subclauses:
            clauses.append(f"({' or '.join(cat_subclauses)})")

    # Handle authors list filter (match any of the specified authors)
    if authors:
        if isinstance(authors, str):
            authors = [authors]
        author_subclauses = [
            f'authors like "%{re.escape(author.strip()).replace(chr(92), "")}%"'
            for author in authors
            if author and author.strip()
        ]
        if author_subclauses:
            clauses.append(f"({' or '.join(author_subclauses)})")

    return " and ".join(clauses)


@mcp.tool()
def search(
    query: str,
    limit: int = 5,
    submitted: list[str | None] | None = None,
    updated: list[str | None] | None = None,
    categories: list[str] | None = None,
    has_doi: bool | None = None,
    authors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Search the arXiv metadata collection using semantic vector similarity and optional metadata filters.

    Args:
        query: Search query text to find relevant papers.
        limit: Maximum number of papers to retrieve (default: 5).
        submitted: Date range [start_date, end_date] for submission date (e.g. ['2023-01-01', '2023-12-31']). Either element can be None/empty for open-ended ranges.
        updated: Date range [start_date, end_date] for last update date (e.g. ['2024-01-01', '2024-12-31']). Either element can be None/empty for open-ended ranges.
        categories: List of categories to filter entries (e.g. ['cs.CL', 'cs.AI']). Matches entries containing any of these.
        has_doi: Boolean flag to filter entries that have (True) or do not have (False) a DOI.
        authors: List of author names to filter entries (e.g. ['Hinton', 'Bengio']). Matches entries containing any of these.

    Returns:
        List of matching papers with similarity scores, title/abstract text, authors, categories, dates, and URLs.
    """
    target_collection = sanitized_collection_name
    if not milvus_client.has_collection(collection_name=target_collection):
        raise RuntimeError(f"Collection '{target_collection}' does not exist in Milvus.")

    # Build Milvus filter expression
    filter_expr = build_filter_expression(
        submitted=submitted,
        updated=updated,
        categories=categories,
        has_doi=has_doi,
        authors=authors,
    )

    embedding = get_embedding(query)

    search_params: dict[str, Any] = {
        "collection_name": target_collection,
        "data": [embedding],
        "limit": limit,
        "output_fields": [
            "arxiv_id",
            "text",
            "filename",
            "url",
            "arxiv_categories",
            "submitted",
            "updated",
            "has_doi",
            "authors",
        ],
    }

    if filter_expr:
        search_params["filter"] = filter_expr

    try:
        search_res = milvus_client.search(**search_params)
    except Exception as e:
        logger.error(f"Search failed on '{target_collection}': {e}", exc_info=True)
        raise RuntimeError(f"Milvus search error on '{target_collection}': {e}") from e

    results: list[dict[str, Any]] = []
    if search_res and len(search_res) > 0:
        for hit in search_res[0]:
            entity = (
                hit.get("entity", {})
                if isinstance(hit, dict)
                else getattr(hit, "entity", {})
            )
            score = (
                hit.get("distance", hit.get("score", 0.0))
                if isinstance(hit, dict)
                else getattr(hit, "distance", getattr(hit, "score", 0.0))
            )
            doc_id = (
                hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None)
            )

            results.append(
                {
                    "arxiv_id": entity.get("arxiv_id") or str(doc_id),
                    "score": score,
                    "text": entity.get("text", ""),
                    "authors": entity.get("authors", ""),
                    "categories": entity.get("arxiv_categories", ""),
                    "submitted": entity.get("submitted", ""),
                    "updated": entity.get("updated", ""),
                    "has_doi": entity.get("has_doi", False),
                    "url": entity.get("url", ""),
                    "filename": entity.get("filename", ""),
                }
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arXiv Metadata Milvus MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for HTTP/SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP/SSE transport (default: 8000)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
