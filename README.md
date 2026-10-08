# arXiv Metadata Knowledge Base

A pipeline for downloading, parsing, embedding, and indexing the entire arXiv metadata snapshot dataset into Milvus vector database.


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
```


## Usage

### 1. Download & Deduplicate Snapshot Dataset

Download the arXiv metadata snapshot from Hugging Face:
```bash
python utils/hf_download.py
```

Remove duplicate arXiv IDs (keeping the record with the most recent `update_date`):
```bash
# In-place deduplication of parquet files (with optional backup)
python utils/hf_postprocess.py --in-place --backup

# Or export to a separate directory
python utils/hf_postprocess.py --output-dir data/arxiv_metadata_snapshot/data_clean
```

### 2. Ingest Data into Milvus
To embed and upload papers:
```bash
python add_milvus.py
```
You can configure filters in `add_milvus.py` (e.g. `CATEGORY_FILTER = "astro-ph.CO"`, `START_DATE`, `END_DATE`, `LIMIT`).


### 3. Run an MCP Server to expose the collection

You can run the Model Context Protocol (MCP) server to allow LLMs and MCP clients to query the Milvus vector database:

```bash
# Default (stdio transport)
python mcp_server.py

# Streamable HTTP transport
python mcp_server.py --transport streamable-http --host 0.0.0.0 --port 3001
```

To test the local MCP server:

```bash
python tests/test_mcp_local.py
```
