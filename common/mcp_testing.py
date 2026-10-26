"""Shared helpers for local MCP server integration tests."""

import json
from typing import Any


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
