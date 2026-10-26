"""
Upload arXiv metadata snapshot dataset to Milvus.
Reference implementation adapted from reference.py with schema including `arxiv_id`.
"""

import datetime
from email.utils import parsedate_to_datetime
import sys
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import pandas as pd
from pymilvus import DataType, MilvusClient
import tqdm

from common.clients import (
    BATCH_SIZE,
    EMBEDDINGS_MODEL,
    EMBEDDINGS_SIZE,
    MAX_TEXT_LENGTH,
    get_milvus_client,
    get_openai_client,
)
from common.utils import (
    ensure_vector_index,
    get_existing_ids,
    sanitize_collection_name,
    setup_logging,
)

# Source-specific configuration constants
COLLECTION_NAME = "arxiv_metadata_v2"
DEFAULT_DATA_PATH = str(Path(__file__).resolve().parents[2] / "data/arxiv_metadata_snapshot/data_clean/*.parquet")

logger = setup_logging("add_milvus", silence_http_libs=True)


def parse_rfc2822_to_iso(date_str: Optional[str]) -> Optional[str]:
    """Parse an RFC 2822 date string (e.g. 'Fri, 21 Mar 2008 14:16:43 GMT') into ISO 8601 UTC string."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        dt = parsedate_to_datetime(date_str.strip())
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc)
        else:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        logger.debug(f"Failed to parse RFC 2822 date '{date_str}': {e}")
        return None


def extract_version_dates(versions: Any) -> Tuple[Optional[str], Optional[str]]:
    """Extract submitted (v1 / first version) and updated (latest version) ISO dates from versions list."""
    if versions is None:
        return None, None

    # Handle numpy arrays, lists, or structured dicts
    if hasattr(versions, "tolist"):
        versions = versions.tolist()

    if not isinstance(versions, (list, tuple)) or len(versions) == 0:
        return None, None

    first_entry = versions[0]
    last_entry = versions[-1]

    submitted_raw = (
        first_entry.get("created")
        if isinstance(first_entry, dict)
        else getattr(first_entry, "created", None)
    )
    updated_raw = (
        last_entry.get("created")
        if isinstance(last_entry, dict)
        else getattr(last_entry, "created", None)
    )

    submitted_iso = parse_rfc2822_to_iso(submitted_raw)
    updated_iso = parse_rfc2822_to_iso(updated_raw)
    return submitted_iso, updated_iso


def format_authors_text(raw_authors: Any) -> str:
    """Format authors to 'First Author et al.' if multiple authors, or single author."""
    if not raw_authors:
        return ""

    if hasattr(raw_authors, "tolist"):
        raw_authors = raw_authors.tolist()

    if len(raw_authors) == 0:
        return ""

    first_author = f"{str(raw_authors[0][1] or '').strip()} {str(raw_authors[0][0] or '').strip()}".strip()
    if len(raw_authors) == 1:
        return first_author
    return f"{first_author} et al." if first_author else ""


def format_authors_field(raw_authors: Any) -> str:
    """Convert raw authors_parsed (e.g., [['Li', 'Chang', ''], ...]) into a comma-separated string '<firstname> <lastname>, ...'."""
    if not raw_authors:
        return ""

    if hasattr(raw_authors, "tolist"):
        raw_authors = raw_authors.tolist()

    # Keep at max 20 authors to avoid bloating the database with giant list of names in papers from collaborations
    max_authors = 20
    authors_list: List[str] = [
        f"{str(a[1] or '').strip()} {str(a[0] or '').strip()}".strip()
        for a in raw_authors[:max_authors]
    ]
    return ", ".join(name for name in authors_list if name)


def ensure_collection(milvus_client: MilvusClient, collection_name: str = COLLECTION_NAME):
    """Ensure the Milvus collection and its schema/index exist."""
    collection_name = sanitize_collection_name(collection_name)

    if collection_name not in milvus_client.list_collections():
        logger.info(f"Creating collection '{collection_name}' with arXiv metadata schema...")
        schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)

        # Primary Key: arxiv_id
        schema.add_field(
            field_name="arxiv_id", datatype=DataType.VARCHAR, max_length=64, is_primary=True
        )

        # Embeddings Vector
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDINGS_SIZE
        )

        # Original text (Title + Authors + Abstract)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=MAX_TEXT_LENGTH,
        )

        # Filename
        schema.add_field(
            field_name="filename", datatype=DataType.VARCHAR, max_length=512
        )

        # URL
        schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=2048)

        # arXiv Categories
        schema.add_field(
            field_name="arxiv_categories", datatype=DataType.VARCHAR, max_length=512
        )

        # Submitted Date (ISO 8601 UTC string e.g. 2008-03-21T14:16:43Z)
        schema.add_field(
            field_name="submitted", datatype=DataType.VARCHAR, max_length=32
        )

        # Updated Date (ISO 8601 UTC string e.g. 2026-09-24T15:42:17Z)
        schema.add_field(
            field_name="updated", datatype=DataType.VARCHAR, max_length=32
        )

        # Has DOI flag
        schema.add_field(field_name="has_doi", datatype=DataType.BOOL)

        # Comma-separated authors: "<firstname> <lastname>, <firstname> <lastname>, ..."
        schema.add_field(
            field_name="authors", datatype=DataType.VARCHAR, max_length=2048
        )

        # Create Index for fast vector similarity search
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="HNSW",
        )

        milvus_client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )
        logger.info(f"Collection '{collection_name}' created successfully.")

    # Ensure the vector index exists (missing if created by an earlier interrupted run)
    ensure_vector_index(milvus_client, collection_name, logger=logger)

    # Load collection into memory
    milvus_client.load_collection(collection_name=collection_name)
    logger.info(f"Collection '{collection_name}' loaded.")


def process_and_upload(
    data_path: str = DEFAULT_DATA_PATH,
    collection_name: str = COLLECTION_NAME,
    category_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    overwrite_existing: bool = False,
    batch_size: int = BATCH_SIZE,
):
    """Read arXiv metadata from parquet, compute embeddings, and upsert into Milvus."""
    start_time = time.time()
    collection_name = sanitize_collection_name(collection_name)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client()

    ensure_collection(milvus_client, collection_name=collection_name)

    # Check for existing papers in Milvus to skip redundant embedding computations
    existing_ids = (
        set()
        if overwrite_existing
        else get_existing_ids(milvus_client, collection_name, pk_field="arxiv_id", logger=logger)
    )

    # Build SQL query with DuckDB
    con = duckdb.connect()
    where_clauses = ["id IS NOT NULL"]
    if category_filter:
        where_clauses.append(f"categories LIKE '%{category_filter}%'")
    if start_date:
        where_clauses.append(f"update_date >= '{start_date}'")
    if end_date:
        where_clauses.append(f"update_date <= '{end_date}'")

    where_stmt = " AND ".join(where_clauses)
    limit_stmt = f"LIMIT {limit}" if limit else ""

    # Compute total records to process
    if limit:
        count_query = f"""
            SELECT count(*)
            FROM (
                SELECT 1
                FROM '{data_path}'
                WHERE {where_stmt}
                LIMIT {limit}
            )
        """
    else:
        count_query = f"""
            SELECT count(*)
            FROM '{data_path}'
            WHERE {where_stmt}
        """

    logger.info(f"Counting total papers to process from {data_path}...")
    total_in_subset = con.execute(count_query).fetchone()[0]
    logger.info(f"Total papers in query subset: {total_in_subset:,}")

    # Determine total_to_process by intersecting query subset IDs with existing_ids in Milvus
    if overwrite_existing or not existing_ids or total_in_subset == 0:
        total_to_process = total_in_subset
    else:
        # Fetch just the IDs from the subset using DuckDB to count how many need embedding
        ids_query = f"""
            SELECT id AS arxiv_id
            FROM '{data_path}'
            WHERE {where_stmt}
            {limit_stmt}
        """
        subset_df = con.execute(ids_query).fetchdf()
        already_in_milvus = subset_df["arxiv_id"].astype(str).str.strip().isin(existing_ids).sum()
        total_to_process = total_in_subset - int(already_in_milvus)

    logger.info(
        f"Papers to embed and upload (excluding already existing): {total_to_process:,} "
        f"(out of {total_in_subset:,} in subset)"
    )

    query = f"""
        SELECT
            id AS arxiv_id,
            title,
            abstract,
            authors,
            categories,
            doi,
            versions,
            authors_parsed,
            strftime(update_date, '%Y-%m-%d') AS update_date
        FROM '{data_path}'
        WHERE {where_stmt}
        {limit_stmt}
    """

    logger.info(f"Querying arXiv dataset from {data_path}...")
    cursor = con.cursor()
    cursor.execute(query)

    # Fetch rows in chunks
    fetch_batch_size = 500
    total_processed = 0
    total_skipped = 0
    total_upserted = 0

    while True:
        rows = cursor.fetchmany(fetch_batch_size)
        if not rows:
            break

        columns = [desc[0] for desc in cursor.description]
        df_batch = pd.DataFrame(rows, columns=columns)

        # Prepare text and items for embeddings (filtering out already existing papers)
        pending_items = []
        for _, row_data in df_batch.iterrows():
            arxiv_id = str(row_data["arxiv_id"]).strip()

            # Skip if already in Milvus and not forcing overwrite
            if not overwrite_existing and arxiv_id in existing_ids:
                total_skipped += 1
                continue

            title = str(row_data["title"] or "").strip()
            abstract = str(row_data["abstract"] or "").strip()
            categories = str(row_data["categories"] or "").strip()
            raw_authors_parsed = row_data.get("authors_parsed")
            authors_text = format_authors_text(raw_authors_parsed)
            authors_field = format_authors_field(raw_authors_parsed)
            url = f"https://arxiv.org/abs/{arxiv_id}"
            filename = f"{arxiv_id}.parquet"

            raw_doi = row_data.get("doi")
            has_doi = bool(raw_doi is not None and str(raw_doi).strip() and str(raw_doi).strip().lower() != "none")

            submitted_iso, updated_iso = extract_version_dates(row_data.get("versions"))
            # Fallback for updated date to update_date if versions created timestamp is missing
            if not updated_iso and row_data.get("update_date"):
                updated_iso = f"{row_data['update_date']}T00:00:00Z"


            full_text = f"Title: {title}\nAuthors: {authors_text}\n\nAbstract: {abstract}"

            # Check and crop text if it exceeds MAX_TEXT_LENGTH (byte-safe truncation)
            encoded = full_text.encode("utf-8")
            if len(encoded) > MAX_TEXT_LENGTH:
                logger.warning(
                    f"Paper {arxiv_id} text exceeds MAX_TEXT_LENGTH ({len(encoded)} > {MAX_TEXT_LENGTH} bytes). Truncating."
                )
                full_text = encoded[:MAX_TEXT_LENGTH].decode("utf-8", errors="ignore")

            pending_items.append(
                {
                    "arxiv_id": arxiv_id,
                    "arxiv_categories": categories,
                    "text": full_text,
                    "filename": filename,
                    "url": url,
                    "submitted": submitted_iso or "",
                    "updated": updated_iso or "",
                    "has_doi": has_doi,
                    "authors": authors_field,
                }
            )

        total_processed += len(pending_items)
        pct = f" ({(total_processed / total_to_process * 100):.1f}%)" if total_to_process > 0 else ""
        logger.info(
            f"Progress: {total_processed}/{total_to_process} papers processed{pct} "
            f"(remaining: {max(0, total_to_process - total_processed)})"
        )

        if not pending_items:
            continue

        # Compute embeddings and upsert in sub-batches
        data_to_upsert = []
        for i in tqdm.tqdm(
            range(0, len(pending_items), batch_size),
            desc=f"  Embedding batch of {len(pending_items)} new papers: ",
            leave=False,
        ):
            sub_batch = pending_items[i : i + batch_size]
            batch_texts = [item["text"] for item in sub_batch]

            try:
                response = openai_client.embeddings.create(
                    model=EMBEDDINGS_MODEL, input=batch_texts
                )

                for j, emb_data in enumerate(response.data):
                    item = sub_batch[j]
                    data_to_upsert.append(
                        {
                            "arxiv_id": item["arxiv_id"],
                            "vector": emb_data.embedding,
                            "text": item["text"],
                            "filename": item["filename"],
                            "url": item["url"],
                            "arxiv_categories": item["arxiv_categories"],
                            "submitted": item["submitted"],
                            "updated": item["updated"],
                            "has_doi": item["has_doi"],
                            "authors": item["authors"],
                        }
                    )
                    # Add to existing_ids to prevent re-processing within the same run
                    existing_ids.add(item["arxiv_id"])
            except Exception as e:
                logger.error(f"Error computing embeddings: {e}")

        # Upsert into Milvus
        if data_to_upsert:
            milvus_client.upsert(collection_name=collection_name, data=data_to_upsert)
            total_upserted += len(data_to_upsert)

    elapsed_time = time.time() - start_time
    logger.info(
        f"⏱️ Finished processing {total_processed}/{total_to_process} papers: "
        f"{total_upserted} upserted, {total_skipped} skipped (already existed) "
        f"in '{collection_name}' in {elapsed_time:.2f}s"
    )


if __name__ == "__main__":
    # Parameters configuration
    DATA_PATH = DEFAULT_DATA_PATH
    COLLECTION = COLLECTION_NAME
    CATEGORY_FILTER = None # "astro-ph.CO"    # e.g., "cs.CL", "cs.AI", "quant-ph", or None for all
    START_DATE = None # "2026-09-05"         # e.g., "2024-01-01" or None
    END_DATE = None # "2026-09-10"           # e.g., "2024-12-31" or None
    LIMIT = None              # e.g., 100 for testing, or None for all
    OVERWRITE_EXISTING = False  # Set True to recompute embeddings & overwrite existing papers in Milvus
    BATCH_SIZE_OVERRIDE = BATCH_SIZE

    process_and_upload(
        data_path=DATA_PATH,
        collection_name=COLLECTION,
        category_filter=CATEGORY_FILTER,
        start_date=START_DATE,
        end_date=END_DATE,
        limit=LIMIT,
        overwrite_existing=OVERWRITE_EXISTING,
        batch_size=BATCH_SIZE_OVERRIDE,
    )
