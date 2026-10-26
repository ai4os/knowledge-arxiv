"""Shared client factories and helpers for embeddings (OpenAI-compatible) and Milvus."""

import os

from dotenv import load_dotenv
import httpx
from openai import OpenAI
from pymilvus import MilvusClient


load_dotenv()

# Shared embedding constants
EMBEDDINGS_SIZE = 2560  # specific to Qwen3-Embeddings-4B
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "AI4EOSC/default-embeddings")
MAX_TEXT_LENGTH = 8192  # Safe byte length for title + authors + abstract
BATCH_SIZE = 10  # batch size for computing embeddings in OpenAI API calls

DEFAULT_OPENAI_BASE_URL = "https://vllm.cloud.ai4eosc.eu/v1"


def get_openai_client(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    """Initialize OpenAI client for embeddings."""
    base_url = (
        base_url
        or os.getenv("EMBEDDINGS_BASE_URL")
        or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    )
    api_key = api_key or os.getenv("LITELLM_KEY") or os.getenv("OPENAI_API_KEY", "dummy-key")
    # We define the client to silence the INFO logs that break the progress bar
    http_client = httpx.Client(transport=httpx.HTTPTransport(retries=3))
    return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)


def get_milvus_client() -> MilvusClient:
    """Initialize Milvus client from environment variables."""
    milvus_uri = os.getenv("MILVUS_URI", "./milvus.db")
    milvus_pwd = os.getenv("MILVUS_PWD")
    milvus_user = os.getenv("MILVUS_USER", "root")

    if milvus_uri.startswith("http://") or milvus_uri.startswith("https://") or ":" in milvus_uri:
        return MilvusClient(uri=milvus_uri, user=milvus_user, password=milvus_pwd)
    return MilvusClient(uri=milvus_uri)


def get_embedding(openai_client: OpenAI, text: str, model: str = EMBEDDINGS_MODEL) -> list[float]:
    """Generate vector embedding for the search query."""
    response = openai_client.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding
