"""
Registers the MCP servers (one per knowledge source) in LiteLLM.

Usage:
    python common/litellm_mcp_register.py                      # register all servers
    python common/litellm_mcp_register.py --server arxiv_metadata
    python common/litellm_mcp_register.py --server csic_metadata
"""

import argparse
import os
import json

from dotenv import load_dotenv
import requests


load_dotenv()

# MCP server configurations, one per knowledge source
SERVERS = {
    "arxiv_metadata": {
        "name": "arxiv_metadata",
        "description": "MCP server to perform RAG on the full arXiv metadata.",
        "url": "https://mcp-arxiv-metadata.ifca-deployments.cloud.ai4eosc.eu/mcp",
    },
    "csic_metadata": {
        "name": "csic_metadata",
        "description": "MCP server to perform RAG on the DIGITAL.CSIC metadata.",
        "url": "https://mcp-csic-metadata.ifca-deployments.cloud.ai4eosc.eu/mcp",
    },
    "hal_metadata": {
        "name": "hal_metadata",
        "description": "MCP server to perform RAG on the Archive Ouverte HAL metadata.",
        "url": "https://mcp-hal-metadata.ifca-deployments.cloud.ai4eosc.eu/mcp",
    },
}

BASE_URL = "https://vllm.cloud.ai4eosc.eu"


def register_server(server_key: str, headers: dict, mcp_auth: str) -> None:
    """Register a single MCP server in LiteLLM and run a health check."""
    config = SERVERS[server_key]

    payload = {
        "server_name": config["name"],
        "description": config["description"],
        "url": config["url"],
        "transport": "http",
        "allow_all_keys": True,
        "static_headers": {
            "Authorization": f"Basic {mcp_auth}"
        }
    }

    print(f"\n{'=' * 60}")
    print(f"Registering MCP server '{config['name']}' ({config['url']})...")

    response = requests.post(
        f"{BASE_URL}/v1/mcp/server",
        json=payload,
        headers=headers,
    )

    print(f"Status code: {response.status_code}")
    resp_data = response.json()
    # print(json.dumps(resp_data, indent=2))

    # Health check
    server_id = resp_data.get("server_id") if isinstance(resp_data, dict) else None
    health_params = {"server_ids": [server_id]} if server_id else None

    health_resp = requests.get(
        f"{BASE_URL}/v1/mcp/server/health",
        headers=headers,
        params=health_params,
    )

    print(f"Health check status code: {health_resp.status_code}")
    status = health_resp.json()[0]['status']
    indicator = "🟢" if status == 'healthy' else "🔴"
    print(f"Health check status: {indicator} {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register MCP servers in LiteLLM")
    parser.add_argument(
        "--server",
        choices=[*SERVERS.keys(), "all"],
        default="all",
        help="Which MCP server to register (default: all)",
    )
    args = parser.parse_args()

    litellm_key = os.environ["LITELLM_REGISTRATION_KEY"]
    mcp_auth = os.environ["MCP_AUTH"]

    headers = {
        "Authorization": f"Bearer {litellm_key}",
        "Content-Type": "application/json",
    }

    server_keys = list(SERVERS.keys()) if args.server == "all" else [args.server]
    for server_key in server_keys:
        register_server(server_key, headers=headers, mcp_auth=mcp_auth)
