"""
Local integration test for arXiv Metadata MCP server over stdio.
Tests both positive and negative cases for semantic search and metadata filtering.

We use the following entry to test the integration:
```
  updated: 2023-12-05T14:27:59Z
  arxiv_id: 2210.07367
  text: Title: Gravitational focusing effects on streaming dark matter as a new detection concept [...]
  arxiv_categories: astro-ph.IM astro-ph.CO astro-ph.EP gr-qc hep-ph
  has_doi: True
  authors: Abaz Kryemadhi, Marios Maroudas, Andreas Mastronikolis, Konstantin Zioutas
  filename: 2210.07367.parquet
  url: https://arxiv.org/abs/2210.07367
  submitted: 2022-10-13T21:21:16Z
```
"""

import asyncio
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Reference paper details
TARGET_ARXIV_ID = "2210.07367"
QUERY_TEXT = "Gravitational focusing effects on streaming dark matter as a new detection concept"


def extract_results(call_tool_response: Any) -> list[dict[str, Any]]:
    """Helper to extract list of dictionaries from MCP tool call content."""
    # Check if tool execution was marked as an error
    if getattr(call_tool_response, "isError", False):
        error_msgs = [getattr(c, "text", str(c)) for c in getattr(call_tool_response, "content", [])]
        print(f"❌ Tool returned error: {error_msgs}")
        return []

    # FastMCP / mcpserver may return structured data in res.data or in res.content
    if hasattr(call_tool_response, "data") and isinstance(call_tool_response.data, list):
        return call_tool_response.data

    content_list = getattr(call_tool_response, "content", [])
    for content in content_list:
        text = getattr(content, "text", None)
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    if "error" in parsed:
                        print(f"⚠️ Server returned error: {parsed['error']}")
                        return []
                    return [parsed]
            except json.JSONDecodeError:
                pass

    # If no content found or unable to parse JSON
    print(f"🔍 Raw response content: {content_list}")
    return []


def assert_paper_found(results: list[dict[str, Any]], expected_id: str, test_name: str) -> None:
    """Assert that the target paper is present in the returned results."""
    ids = [str(r.get("arxiv_id", "")) for r in results]
    assert expected_id in ids, (
        f"❌ [{test_name}] Expected paper '{expected_id}' to be found in results, but got: {ids}"
    )
    print(f"✅ [{test_name}] PASSED (paper '{expected_id}' found)")


def assert_paper_not_found(results: list[dict[str, Any]], expected_id: str, test_name: str) -> None:
    """Assert that the target paper is NOT present in the returned results."""
    ids = [str(r.get("arxiv_id", "")) for r in results]
    assert expected_id not in ids, (
        f"❌ [{test_name}] Expected paper '{expected_id}' NOT to be found, but it appeared in results: {ids}"
    )
    print(f"✅ [{test_name}] PASSED (paper '{expected_id}' successfully excluded)")


async def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_path = os.path.join(workspace_dir, "mcp_server.py")

    # 1. Configure the local MCP server process running via stdio
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path],
        env=dict(os.environ),
    )

    print(f"🚀 Connecting to local MCP server via stdio ({sys.executable} {server_path})...\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize MCP session
            await session.initialize()

            # Verify available tools
            tools_list = await session.list_tools()
            tool_names = [t.name for t in tools_list.tools]
            print(f"🛠️  Discovered tools: {tool_names}\n")
            assert "search" in tool_names, "Expected 'search' tool to be exposed by the server"

            # -------------------------------------------------------------
            # Baseline Positive Test: Semantic search without filters
            # -------------------------------------------------------------
            res = await session.call_tool(
                "search",
                arguments={"query": QUERY_TEXT, "limit": 10},
            )
            results = extract_results(res)
            print(f"📊 Baseline returned {len(results)} paper(s): {[r.get('arxiv_id') for r in results]}")
            assert_paper_found(results, TARGET_ARXIV_ID, "Baseline Semantic Search")

            # -------------------------------------------------------------
            # Submitted Date Range Tests (Actual submitted: 2022-10-13)
            # -------------------------------------------------------------
            # Positive: Proper submitted date range covering 2022-10-13
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "submitted": ["2022-01-01", "2022-12-31"],
                },
            )
            assert_paper_found(extract_results(res), TARGET_ARXIV_ID, "Submitted Date: Positive")

            # Negative: Improper submitted date range (after submission)
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "submitted": ["2023-01-01", "2023-12-31"],
                },
            )
            assert_paper_not_found(extract_results(res), TARGET_ARXIV_ID, "Submitted Date: Negative")

            # -------------------------------------------------------------
            # Updated Date Range Tests (Actual updated: 2023-12-05)
            # -------------------------------------------------------------
            # Positive: Proper updated date range covering 2023-12-05
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "updated": ["2023-01-01", "2023-12-31"],
                },
            )
            assert_paper_found(extract_results(res), TARGET_ARXIV_ID, "Updated Date: Positive")

            # Negative: Improper updated date range (before update)
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "updated": ["2020-01-01", "2021-12-31"],
                },
            )
            assert_paper_not_found(extract_results(res), TARGET_ARXIV_ID, "Updated Date: Negative")

            # -------------------------------------------------------------
            # Categories Filter Tests (Actual categories: astro-ph.CO ...)
            # -------------------------------------------------------------
            # Positive: Category contains matching category
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "categories": ["astro-ph.CO"],
                },
            )
            assert_paper_found(extract_results(res), TARGET_ARXIV_ID, "Categories: Positive")

            # Negative: Non-matching categories
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "categories": ["math.PR", "q-bio.NC"],
                },
            )
            assert_paper_not_found(extract_results(res), TARGET_ARXIV_ID, "Categories: Negative")

            # -------------------------------------------------------------
            # Authors Filter Tests (Actual authors include: Konstantin Zioutas)
            # -------------------------------------------------------------
            # Positive: Matching author
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "authors": ["Zioutas"],
                },
            )
            assert_paper_found(extract_results(res), TARGET_ARXIV_ID, "Authors: Positive")

            # Negative: Non-matching author
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "authors": ["NonExistentAuthorXYZ"],
                },
            )
            assert_paper_not_found(extract_results(res), TARGET_ARXIV_ID, "Authors: Negative")

            # -------------------------------------------------------------
            # Has DOI Filter Tests (Actual: has_doi == True)
            # -------------------------------------------------------------
            # Positive: has_doi = True
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "has_doi": True,
                },
            )
            assert_paper_found(extract_results(res), TARGET_ARXIV_ID, "Has DOI: Positive (True)")

            # Negative: has_doi = False
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "has_doi": False,
                },
            )
            assert_paper_not_found(extract_results(res), TARGET_ARXIV_ID, "Has DOI: Negative (False)")

            print("\n🎉 All positive and negative tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
