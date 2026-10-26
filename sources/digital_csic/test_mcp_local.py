"""
Local integration test for DIGITAL.CSIC Metadata MCP server over stdio.
Tests both positive and negative cases for semantic search and metadata filtering.

We use the following entry to test the integration:
```
  csic_id: 10261/1423
  title: Is mammalian chromosomal evolution driven by regions of genome fragility?
  authors: Aurora Ruiz-Herrera, José Castresana, Terence J. Robinson
  subjects: Comparative genomics, Genome evolution, Mammalian, Eutherian species
  type: artículo
  language: eng
  date_issued: 2006-12-08
  year: 2006
  updated: 2021-11-12T09:38:55Z
  has_doi: True
  url: http://hdl.handle.net/10261/1423
```
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from common.mcp_testing import extract_results

load_dotenv()

# Reference item details
TARGET_CSIC_ID = "10261/1423"
QUERY_TEXT = "Is mammalian chromosomal evolution driven by regions of genome fragility?"


def assert_item_found(results: list[dict[str, Any]], expected_id: str, test_name: str) -> None:
    """Assert that the target item is present in the returned results."""
    ids = [str(r.get("csic_id", "")) for r in results]
    assert expected_id in ids, (
        f"❌ [{test_name}] Expected item '{expected_id}' to be found in results, but got: {ids}"
    )
    print(f"✅ [{test_name}] PASSED (item '{expected_id}' found)")


def assert_item_not_found(results: list[dict[str, Any]], expected_id: str, test_name: str) -> None:
    """Assert that the target item is NOT present in the returned results."""
    ids = [str(r.get("csic_id", "")) for r in results]
    assert expected_id not in ids, (
        f"❌ [{test_name}] Expected item '{expected_id}' NOT to be found, but it appeared in results: {ids}"
    )
    print(f"✅ [{test_name}] PASSED (item '{expected_id}' successfully excluded)")


async def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
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
            print(f"📊 Baseline returned {len(results)} item(s): {[r.get('csic_id') for r in results]}")
            assert_item_found(results, TARGET_CSIC_ID, "Baseline Semantic Search")

            # -------------------------------------------------------------
            # Year Range Tests (Actual year: 2006)
            # -------------------------------------------------------------
            # Positive: Proper year range covering 2006
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "year": [2000, 2010],
                },
            )
            assert_item_found(extract_results(res), TARGET_CSIC_ID, "Year Range: Positive")

            # Negative: Improper year range (after publication)
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "year": [2010, 2020],
                },
            )
            assert_item_not_found(extract_results(res), TARGET_CSIC_ID, "Year Range: Negative")

            # -------------------------------------------------------------
            # Updated Date Range Tests (Actual updated: 2021-11-12)
            # -------------------------------------------------------------
            # Positive: Proper updated date range covering 2021-11-12
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "updated": ["2021-01-01", "2021-12-31"],
                },
            )
            assert_item_found(extract_results(res), TARGET_CSIC_ID, "Updated Date: Positive")

            # Negative: Improper updated date range (before update)
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "updated": ["2010-01-01", "2015-12-31"],
                },
            )
            assert_item_not_found(extract_results(res), TARGET_CSIC_ID, "Updated Date: Negative")

            # -------------------------------------------------------------
            # Subjects Filter Tests (Actual subjects: Comparative genomics, ...)
            # -------------------------------------------------------------
            # Positive: Subjects contains matching subject
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "subjects": ["Genome evolution"],
                },
            )
            assert_item_found(extract_results(res), TARGET_CSIC_ID, "Subjects: Positive")

            # Negative: Non-matching subjects
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "subjects": ["Quantum chromodynamics", "Dark matter"],
                },
            )
            assert_item_not_found(extract_results(res), TARGET_CSIC_ID, "Subjects: Negative")

            # -------------------------------------------------------------
            # Authors Filter Tests (Actual authors include: Ruiz-Herrera)
            # -------------------------------------------------------------
            # Positive: Matching author
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "authors": ["Ruiz-Herrera"],
                },
            )
            assert_item_found(extract_results(res), TARGET_CSIC_ID, "Authors: Positive")

            # Negative: Non-matching author
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "authors": ["NonExistentAuthorXYZ"],
                },
            )
            assert_item_not_found(extract_results(res), TARGET_CSIC_ID, "Authors: Negative")

            # -------------------------------------------------------------
            # Type Filter Tests (Actual type: artículo)
            # -------------------------------------------------------------
            # Positive: Matching type
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "type": "artículo",
                },
            )
            assert_item_found(extract_results(res), TARGET_CSIC_ID, "Type: Positive")

            # Negative: Non-matching type
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "type": "documento de trabajo",
                },
            )
            assert_item_not_found(extract_results(res), TARGET_CSIC_ID, "Type: Negative")

            # -------------------------------------------------------------
            # Language Filter Tests (Actual language: eng)
            # -------------------------------------------------------------
            # Positive: Matching language
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "language": "eng",
                },
            )
            assert_item_found(extract_results(res), TARGET_CSIC_ID, "Language: Positive")

            # Negative: Non-matching language
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "language": "spa",
                },
            )
            assert_item_not_found(extract_results(res), TARGET_CSIC_ID, "Language: Negative")

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
            assert_item_found(extract_results(res), TARGET_CSIC_ID, "Has DOI: Positive (True)")

            # Negative: has_doi = False
            res = await session.call_tool(
                "search",
                arguments={
                    "query": QUERY_TEXT,
                    "has_doi": False,
                },
            )
            assert_item_not_found(extract_results(res), TARGET_CSIC_ID, "Has DOI: Negative (False)")

            print("\n🎉 All positive and negative tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
