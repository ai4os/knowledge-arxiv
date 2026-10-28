"""
MCP server to expose HAL metadata collection from Milvus.
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
COLLECTION_NAME = os.getenv("HAL_COLLECTION_NAME", "hal_metadata_v1")

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
mcp = MCPServer("hal-mcp")


def build_filter_expression(
    year: list[int | None] | None = None,
    updated: list[str | None] | None = None,
    domains: list[str] | None = None,
    authors: list[str] | None = None,
    type: str | None = None,
    language: str | None = None,
    has_doi: bool | None = None,
) -> str:
    """Construct a Milvus boolean filter expression for scalar fields."""
    clauses: list[str] = []

    # Handle year range [start, end]
    if year:
        start = year[0] if len(year) >= 1 else None
        end = year[1] if len(year) >= 2 else None
        if start is not None:
            clauses.append(f"year >= {int(start)}")
        if end is not None:
            clauses.append(f"year <= {int(end)}")

    # Handle updated (OAI-PMH datestamp) date range [start, end]
    if updated:
        start = updated[0] if len(updated) >= 1 else None
        end = updated[1] if len(updated) >= 2 else None
        if start and str(start).strip():
            clauses.append(f'updated >= "{normalize_iso_date(str(start), is_end=False)}"')
        if end and str(end).strip():
            clauses.append(f'updated <= "{normalize_iso_date(str(end), is_end=True)}"')

    # Handle exact type filter (short info:eu-repo type, e.g. 'article')
    if type and type.strip():
        clauses.append(f'type == "{escape_milvus_like(type)}"')

    # Handle exact language filter (e.g. 'en', 'fr')
    if language and language.strip():
        clauses.append(f'language == "{escape_milvus_like(language)}"')

    # Handle has_doi boolean filter
    if has_doi is not None:
        clauses.append(f"has_doi == {'true' if has_doi else 'false'}")

    # Handle domains list filter (match any of the specified domains)
    if domains:
        if isinstance(domains, str):
            domains = [domains]
        domain_subclauses = [
            f'domains like "%{escape_milvus_like(dom)}%"'
            for dom in domains
            if dom and dom.strip()
        ]
        if domain_subclauses:
            clauses.append(f"({' or '.join(domain_subclauses)})")

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
    year: list[int | None] | None = None,
    updated: list[str | None] | None = None,
    domains: list[str] | None = None,
    authors: list[str] | None = None,
    type: str | None = None,
    language: str | None = None,
    has_doi: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Search the HAL metadata collection using semantic vector similarity and optional metadata filters.

    Args:
        query: Search query text to find relevant items.
        limit: Maximum number of items to retrieve (default: 5).
        year: Year range [start_year, end_year] for publication year (e.g. [2000, 2020]). Either element can be None for open-ended ranges.
        updated: Date range [start_date, end_date] for last record modification date (e.g. ['2024-01-01', '2024-12-31']). Either element can be None/empty for open-ended ranges.
        domains: List of HAL domains (discipline labels) to filter items (e.g. ['Life Sciences', 'Computer Science']). Matches items containing any of these.
        authors: List of author names to filter items (e.g. ['Malicet', 'Verrecchia']). Matches items containing any of these.
        type: Document type to filter by (type: 'article', 'conferenceObject', 'bookPart', 'other', 'doctoralThesis', 'book', 'preprint', 'report', 'patent', 'lecture').
        language: Language code to filter by (e.g. 'en', 'fr').
        has_doi: Boolean flag to filter items that have (True) or do not have (False) a DOI.

    Returns:
        List of matching items with similarity scores, title/abstract text, authors, domains, type, language, dates, and URLs.
    """
    target_collection = sanitized_collection_name
    if not milvus_client.has_collection(collection_name=target_collection):
        raise RuntimeError(f"Collection '{target_collection}' does not exist in Milvus.")

    # Build Milvus filter expression
    filter_expr = build_filter_expression(
        year=year,
        updated=updated,
        domains=domains,
        authors=authors,
        type=type,
        language=language,
        has_doi=has_doi,
    )

    embedding = get_embedding(openai_client, query)

    search_params: dict[str, Any] = {
        "collection_name": target_collection,
        "data": [embedding],
        "limit": limit,
        "output_fields": [
            "hal_id",
            "text",
            "url",
            "domains",
            "authors",
            "type",
            "language",
            "date_issued",
            "year",
            "updated",
            "has_doi",
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
                    "hal_id": entity.get("hal_id") or str(doc_id),
                    "score": score,
                    "text": entity.get("text", ""),
                    "authors": entity.get("authors", ""),
                    "domains": entity.get("domains", ""),
                    "type": entity.get("type", ""),
                    "language": entity.get("language", ""),
                    "date_issued": entity.get("date_issued", ""),
                    "year": entity.get("year"),
                    "updated": entity.get("updated", ""),
                    "has_doi": entity.get("has_doi", False),
                    "url": entity.get("url", ""),
                }
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAL Metadata Milvus MCP Server")
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
