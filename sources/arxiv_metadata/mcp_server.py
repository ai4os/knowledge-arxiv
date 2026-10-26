"""
MCP server to expose arXiv metadata collection from Milvus.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.mcpserver import MCPServer

from common.clients import get_embedding, get_milvus_client, get_openai_client
from common.utils import (
    escape_milvus_like,
    normalize_iso_date,
    sanitize_collection_name,
    setup_logging,
)

logger = setup_logging("mcp_server")

# Configuration constants
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "arxiv_metadata_v2")

# Initialize clients
openai_client = get_openai_client()
milvus_client = get_milvus_client()

# Ensure collection name is sanitized and load if available
sanitized_collection_name = sanitize_collection_name(COLLECTION_NAME)
try:
    if milvus_client.has_collection(collection_name=sanitized_collection_name):
        milvus_client.load_collection(collection_name=sanitized_collection_name)
        logger.info(f"Loaded collection '{sanitized_collection_name}' into memory.")
except Exception as e:
    logger.warning(f"Could not load collection '{sanitized_collection_name}': {e}")

# Initialize MCP server
mcp = MCPServer("arxiv-metadata-mcp")


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
            f'arxiv_categories like "%{escape_milvus_like(cat)}%"'
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
            f'authors like "%{escape_milvus_like(author)}%"'
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

    embedding = get_embedding(openai_client, query)

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
