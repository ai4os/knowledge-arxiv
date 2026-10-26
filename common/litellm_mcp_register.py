"""
Registers the MCP in LiteLLM
"""

import os

from dotenv import load_dotenv
import requests


load_dotenv()

MCP_NAME = "arxiv_metadata"
MCP_DESCRIPTION = "MCP server to perform RAG on the full arXiv metadata."
MCP_ENDPOINT = "https://mcp-arxiv-metadata.ifca-deployments.cloud.ai4eosc.eu/mcp"


LITELLM_KEY = os.environ["LITELLM_KEY"]
MCP_AUTH = os.environ["MCP_AUTH"]

BASE_URL = "https://vllm.cloud.ai4eosc.eu"

payload = {
    "server_name": MCP_NAME,
    "description": MCP_DESCRIPTION,
    "url": MCP_ENDPOINT,
    "transport": "http",
    "allow_all_keys": True,
    "static_headers": {
        "Authorization": f"Basic {MCP_AUTH}"
    }
}
headers = {
    "Authorization": f"Bearer {LITELLM_KEY}",
    "Content-Type": "application/json",
}

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

print(f"\nHealth check status code: {health_resp.status_code}")
status = health_resp.json()[0]['status']
indicator = "🟢" if status == 'healthy' else "🔴"
print(f"\nHealth check status: {indicator} {status}")