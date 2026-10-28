# Knowledge Sources → Milvus → MCP

Pipelines for downloading, parsing, embedding, and indexing scholarly metadata snapshots into a Milvus vector database, and exposing them through MCP servers.

Supported knowledge sources live under `sources/<source>/`:

| Source | Description | Milvus collection |
|---|---|---|
| [`arxiv_metadata`](sources/arxiv_metadata/) | Full [arXiv metadata snapshot](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) | `arxiv_metadata_v2` |
| [`csic_metadata`](sources/csic_metadata/) | [DIGITAL.CSIC metadata snapshot](https://huggingface.co/datasets/ifca-advanced-computing/DIGITAL.CSIC-metadata-snapshot) | `digital_csic_metadata_v1` |
| [`hal_metadata`](sources/hal_metadata/) | [HAL metadata snapshot](https://huggingface.co/datasets/ifca-advanced-computing/HAL-metadata-snapshot) | `hal_metadata_v1` |

Shared code lives in [`common/`](common/):

- [`common/clients.py`](common/clients.py) — OpenAI-compatible embeddings client, Milvus client, `get_embedding`, and shared embedding constants.
- [`common/utils.py`](common/utils.py) — logging setup, collection-name sanitization, ISO date normalization, Milvus LIKE escaping, and `get_existing_ids`.
- [`common/mcp_testing.py`](common/mcp_testing.py) — shared helpers for the local MCP integration tests.
- [`common/hf_download.py`](common/hf_download.py) / [`common/litellm_mcp_register.py`](common/litellm_mcp_register.py) — dataset download and MCP registration.


## Installation

* Install the requirements:

```
pip install -r requirements.txt
```

* Configure your credentials in a `.env` file:
```env
OPENAI_BASE_URL=https://vllm.cloud.ai4eosc.eu/v1
LITELLM_KEY=sk-...
MILVUS_URI=https://milvus.k8s.cloud.ai4eosc.eu
MILVUS_USER=root
MILVUS_PWD=...
MCP_AUTH=...
```


## Usage

### 1. Download Snapshot Datasets

Download the metadata snapshots from Hugging Face (arXiv and DIGITAL.CSIC):
```bash
python common/hf_download.py
```

### 2. Ingest Data into Milvus

#### arXiv metadata

The arXiv snapshot contains duplicate IDs, so deduplicate first (keeping the record with the most recent `update_date`):
```bash
# In-place deduplication of parquet files (with optional backup)
python sources/arxiv_metadata/hf_postprocess.py --in-place --backup

# Or export to a separate directory
python sources/arxiv_metadata/hf_postprocess.py --output-dir data/arxiv_metadata_snapshot/data_clean
```

Then embed and upload papers:
```bash
python sources/arxiv_metadata/add_milvus.py
```
You can configure filters in `sources/arxiv_metadata/add_milvus.py` (e.g. `CATEGORY_FILTER = "astro-ph.CO"`, `START_DATE`, `END_DATE`, `LIMIT`).

#### DIGITAL.CSIC metadata

The DIGITAL.CSIC snapshot has unique IDs, so no postprocessing is needed — ingest directly:
```bash
python sources/csic_metadata/add_milvus.py
```
You can configure filters in `sources/csic_metadata/add_milvus.py` (e.g. `SUBJECT_FILTER`, `TYPE_FILTER`, `START_YEAR`, `END_YEAR`, `LIMIT`).

#### HAL metadata

The HAL snapshot has unique IDs, so no postprocessing is needed — ingest directly:
```bash
python sources/hal_metadata/add_milvus.py
```
You can configure filters in `sources/hal_metadata/add_milvus.py` (e.g. `DOMAIN_FILTER`, `TYPE_FILTER` — short type like `"article"`, `START_YEAR`, `END_YEAR`, `LIMIT`).

### 3. Run an MCP Server to expose the collection

You can run a Model Context Protocol (MCP) server per source to allow LLMs and MCP clients to query the corresponding Milvus collection:

```bash
# Default (stdio transport)
python sources/<source>/mcp_server.py

# Streamable HTTP transport
python sources/<source>/mcp_server.py --transport streamable-http --host 0.0.0.0 --port 3001
```

To test a local MCP server (requires the collection to be ingested already):

```bash
python sources/arxiv_metadata/test_mcp_local.py
python sources/csic_metadata/test_mcp_local.py
python sources/hal_metadata/test_mcp_local.py
```

### 4. Register the MCP servers in LiteLLM

Once the MCP servers are deployed, register them in LiteLLM so they are available to LLM clients:

```bash
# Register all servers
python common/litellm_mcp_register.py

# Or register a single one
python common/litellm_mcp_register.py --server arxiv_metadata
python common/litellm_mcp_register.py --server csic_metadata
python common/litellm_mcp_register.py --server hal_metadata
```


## Adding a new knowledge source

1. Add a downloader for the snapshot in `common/hf_download.py` (if hosted on Hugging Face).
2. Create `sources/<source>/add_milvus.py` adapted to the source schema (primary key, metadata fields, filters), reusing the shared clients and helpers from `common/`. If the snapshot has duplicate IDs, also add a postprocessing script.
3. Create `sources/<source>/mcp_server.py` exposing a `search` tool over the source collection, and `sources/<source>/test_mcp_local.py` to test it.
4. Add the MCP server configuration to `SERVERS` in `common/litellm_mcp_register.py`.
