"""
Upload DIGITAL.CSIC metadata snapshot dataset to Milvus.
Adapted from sources/arxiv_metadata/add_milvus.py to the DIGITAL.CSIC schema
(Handle `id` as primary key; no dedup preprocessing needed since IDs are unique).
"""

import sys
from pathlib import Path
import time
from typing import Optional

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
    format_authors_field,
    format_authors_text,
    get_existing_ids,
    sanitize_collection_name,
    setup_logging,
    timestamp_to_iso,
    to_str_list,
)

# Source-specific configuration constants
COLLECTION_NAME = "digital_csic_metadata_v1"
DEFAULT_DATA_PATH = "data/csic_metadata_snapshot/data/*.parquet"

logger = setup_logging("add_milvus", silence_http_libs=True)


def ensure_collection(milvus_client: MilvusClient, collection_name: str = COLLECTION_NAME):
    """Ensure the Milvus collection and its schema/index exist."""
    collection_name = sanitize_collection_name(collection_name)

    if collection_name not in milvus_client.list_collections():
        logger.info(f"Creating collection '{collection_name}' with DIGITAL.CSIC metadata schema...")
        schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)

        # Primary Key: csic_id (Handle, e.g. '10261/1399')
        schema.add_field(
            field_name="csic_id", datatype=DataType.VARCHAR, max_length=64, is_primary=True
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

        # URL (handle_uri, link to the original record)
        schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=2048)

        # Comma-separated subjects (dc.subject)
        schema.add_field(
            field_name="subjects", datatype=DataType.VARCHAR, max_length=2048
        )

        # Comma-separated authors ("First Last", converted from the stored 'Last, First')
        schema.add_field(
            field_name="authors", datatype=DataType.VARCHAR, max_length=2048
        )

        # Item type (dc.type, e.g. 'artículo', 'documento de trabajo')
        schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=128)

        # Language (dc.language, e.g. 'eng', 'spa')
        schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=32)

        # Date issued (dc.date.issued, variable precision e.g. '2004-11' or '2005-10-03')
        schema.add_field(
            field_name="date_issued", datatype=DataType.VARCHAR, max_length=32
        )

        # Publication year (nullable: a small number of records lack a year)
        schema.add_field(field_name="year", datatype=DataType.INT16, nullable=True)

        # Last modification of the record (OAI-PMH datestamp, ISO 8601 UTC string)
        schema.add_field(
            field_name="updated", datatype=DataType.VARCHAR, max_length=32
        )

        # Has DOI flag
        schema.add_field(field_name="has_doi", datatype=DataType.BOOL)

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
    subject_filter: Optional[str] = None,
    type_filter: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    limit: Optional[int] = None,
    overwrite_existing: bool = False,
    batch_size: int = BATCH_SIZE,
):
    """Read DIGITAL.CSIC metadata from parquet, compute embeddings, and upsert into Milvus."""
    start_time = time.time()
    collection_name = sanitize_collection_name(collection_name)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client()

    ensure_collection(milvus_client, collection_name=collection_name)

    # Check for existing items in Milvus to skip redundant embedding computations
    existing_ids = (
        set()
        if overwrite_existing
        else get_existing_ids(milvus_client, collection_name, pk_field="csic_id", logger=logger)
    )

    # Build SQL query with DuckDB
    con = duckdb.connect()
    where_clauses = ["id IS NOT NULL"]
    if subject_filter:
        where_clauses.append(f"list_contains(subjects, '{subject_filter}')")
    if type_filter:
        where_clauses.append(f"type = '{type_filter}'")
    if start_year:
        where_clauses.append(f"year >= {int(start_year)}")
    if end_year:
        where_clauses.append(f"year <= {int(end_year)}")

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

    logger.info(f"Counting total items to process from {data_path}...")
    total_in_subset = con.execute(count_query).fetchone()[0]
    logger.info(f"Total items in query subset: {total_in_subset:,}")

    # Determine total_to_process by intersecting query subset IDs with existing_ids in Milvus
    if overwrite_existing or not existing_ids or total_in_subset == 0:
        total_to_process = total_in_subset
    else:
        # Fetch just the IDs from the subset using DuckDB to count how many need embedding
        ids_query = f"""
            SELECT id AS csic_id
            FROM '{data_path}'
            WHERE {where_stmt}
            {limit_stmt}
        """
        subset_df = con.execute(ids_query).fetchdf()
        already_in_milvus = subset_df["csic_id"].astype(str).str.strip().isin(existing_ids).sum()
        total_to_process = total_in_subset - int(already_in_milvus)

    logger.info(
        f"Items to embed and upload (excluding already existing): {total_to_process:,} "
        f"(out of {total_in_subset:,} in subset)"
    )

    query = f"""
        SELECT
            id AS csic_id,
            handle_uri,
            title,
            abstract,
            authors,
            subjects,
            type,
            language,
            date_issued,
            year,
            datestamp,
            doi
        FROM '{data_path}'
        WHERE {where_stmt}
        {limit_stmt}
    """

    logger.info(f"Querying DIGITAL.CSIC dataset from {data_path}...")
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

        # Prepare text and items for embeddings (filtering out already existing items)
        pending_items = []
        for _, row_data in df_batch.iterrows():
            csic_id = str(row_data["csic_id"]).strip()

            # Skip if already in Milvus and not forcing overwrite
            if not overwrite_existing and csic_id in existing_ids:
                total_skipped += 1
                continue

            title = str(row_data["title"] or "").strip()
            abstract = str(row_data["abstract"] or "").strip()
            subjects = ", ".join(to_str_list(row_data.get("subjects")))
            authors_text = format_authors_text(row_data.get("authors"))
            authors_field = format_authors_field(row_data.get("authors"))
            url = str(row_data.get("handle_uri") or "").strip() or f"http://hdl.handle.net/{csic_id}"

            raw_doi = row_data.get("doi")
            has_doi = bool(raw_doi is not None and str(raw_doi).strip() and str(raw_doi).strip().lower() != "none")

            updated_iso = timestamp_to_iso(row_data.get("datestamp"))

            raw_year = row_data.get("year")
            year = int(raw_year) if raw_year is not None and not pd.isna(raw_year) else None

            full_text = f"Title: {title}\nAuthors: {authors_text}\n\nAbstract: {abstract}"

            # Check and crop text if it exceeds MAX_TEXT_LENGTH (byte-safe truncation)
            encoded = full_text.encode("utf-8")
            if len(encoded) > MAX_TEXT_LENGTH:
                logger.warning(
                    f"Item {csic_id} text exceeds MAX_TEXT_LENGTH ({len(encoded)} > {MAX_TEXT_LENGTH} bytes). Truncating."
                )
                full_text = encoded[:MAX_TEXT_LENGTH].decode("utf-8", errors="ignore")

            pending_items.append(
                {
                    "csic_id": csic_id,
                    "text": full_text,
                    "url": url,
                    "subjects": subjects,
                    "authors": authors_field,
                    "type": str(row_data.get("type") or "").strip(),
                    "language": str(row_data.get("language") or "").strip(),
                    "date_issued": str(row_data.get("date_issued") or "").strip(),
                    "year": year,
                    "updated": updated_iso or "",
                    "has_doi": has_doi,
                }
            )

        total_processed += len(pending_items)
        pct = f" ({(total_processed / total_to_process * 100):.1f}%)" if total_to_process > 0 else ""
        logger.info(
            f"Progress: {total_processed}/{total_to_process} items processed{pct} "
            f"(remaining: {max(0, total_to_process - total_processed)})"
        )

        if not pending_items:
            continue

        # Compute embeddings and upsert in sub-batches
        data_to_upsert = []
        for i in tqdm.tqdm(
            range(0, len(pending_items), batch_size),
            desc=f"  Embedding batch of {len(pending_items)} new items: ",
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
                            "csic_id": item["csic_id"],
                            "vector": emb_data.embedding,
                            "text": item["text"],
                            "url": item["url"],
                            "subjects": item["subjects"],
                            "authors": item["authors"],
                            "type": item["type"],
                            "language": item["language"],
                            "date_issued": item["date_issued"],
                            "year": item["year"],
                            "updated": item["updated"],
                            "has_doi": item["has_doi"],
                        }
                    )
                    # Add to existing_ids to prevent re-processing within the same run
                    existing_ids.add(item["csic_id"])
            except Exception as e:
                logger.error(f"Error computing embeddings: {e}")

        # Upsert into Milvus
        if data_to_upsert:
            milvus_client.upsert(collection_name=collection_name, data=data_to_upsert)
            total_upserted += len(data_to_upsert)

    elapsed_time = time.time() - start_time
    logger.info(
        f"⏱️ Finished processing {total_processed}/{total_to_process} items: "
        f"{total_upserted} upserted, {total_skipped} skipped (already existed) "
        f"in '{collection_name}' in {elapsed_time:.2f}s"
    )


if __name__ == "__main__":
    # Parameters configuration
    DATA_PATH = DEFAULT_DATA_PATH
    COLLECTION = COLLECTION_NAME
    SUBJECT_FILTER = None  # e.g., "Genome evolution", or None for all
    TYPE_FILTER = None     # e.g., "artículo", "documento de trabajo", or None for all
    START_YEAR = None      # e.g., 2000 or None
    END_YEAR = None        # e.g., 2020 or None
    LIMIT = None           # e.g., 100 for testing, or None for all
    OVERWRITE_EXISTING = False  # Set True to recompute embeddings & overwrite existing items in Milvus
    BATCH_SIZE_OVERRIDE = BATCH_SIZE

    process_and_upload(
        data_path=DATA_PATH,
        collection_name=COLLECTION,
        subject_filter=SUBJECT_FILTER,
        type_filter=TYPE_FILTER,
        start_year=START_YEAR,
        end_year=END_YEAR,
        limit=LIMIT,
        overwrite_existing=OVERWRITE_EXISTING,
        batch_size=BATCH_SIZE_OVERRIDE,
    )
