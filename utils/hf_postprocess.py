"""
Original HF dataset has duplicated entries [1].
We need to postprocess it in order to remove them.

This script deduplicates arXiv metadata parquet files by arXiv ID (`id`),
keeping the record with the most recent `update_date`.

[1]: https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot/discussions/3
"""

import argparse
import glob
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Optional

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hf_postprocess")


def resolve_default_paths() -> Path:
    """Resolve default data directory relative to repository root or script location."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Check candidates for data directory
    candidates = [
        repo_root / "data" / "arxiv_metadata_snapshot" / "data",
        script_dir / ".." / "data" / "arxiv_metadata_snapshot" / "data",
        Path("data/arxiv_metadata_snapshot/data"),
    ]
    for candidate in candidates:
        if candidate.resolve().exists():
            return candidate.resolve()

    return (repo_root / "data" / "arxiv_metadata_snapshot" / "data").resolve()


def deduplicate_parquet_files(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    in_place: bool = False,
    backup: bool = False,
    num_shards: Optional[int] = None,
    compression: str = "zstd",
) -> bool:
    """
    Deduplicate parquet files by arXiv `id`, keeping the entry with the most recent `update_date`.

    Args:
        input_dir: Path to directory containing original parquet files.
        output_dir: Path to directory to store deduplicated parquet files.
        in_place: If True, replace the original files in `input_dir` with the clean files.
        backup: If True and in_place is True, back up original files before replacing.
        num_shards: Number of output parquet shards. Defaults to number of input files.
        compression: Parquet compression codec (e.g. 'zstd', 'snappy').
    """
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return False

    input_files = sorted(glob.glob(str(input_dir / "*.parquet")))
    if not input_files:
        logger.error(f"No parquet files found in {input_dir}")
        return False

    num_input_files = len(input_files)
    shards_to_write = num_shards if num_shards and num_shards > 0 else num_input_files
    logger.info(f"Found {num_input_files} parquet files in {input_dir}")

    # Determine target output directory
    if in_place:
        target_dir = input_dir
    elif output_dir:
        target_dir = output_dir.resolve()
    else:
        target_dir = input_dir.parent / f"{input_dir.name}_clean"

    logger.info(f"Target output directory: {target_dir}")

    # Connect to DuckDB
    con = duckdb.connect()

    # Step 1: Scan and inspect duplicates
    input_glob = str(input_dir / "*.parquet")
    logger.info(f"Analyzing dataset records from: {input_glob}")

    stats = con.sql(f"""
        SELECT
            count(*) AS total_count,
            count(DISTINCT id) AS distinct_count
        FROM read_parquet('{input_glob}')
    """).fetchone()

    total_count, distinct_count = stats[0], stats[1]
    duplicate_count = total_count - distinct_count

    logger.info(f"Total records:    {total_count:,}")
    logger.info(f"Distinct IDs:     {distinct_count:,}")
    logger.info(f"Duplicate records: {duplicate_count:,}")

    if duplicate_count == 0:
        logger.info("No duplicated arXiv IDs found. Dataset is already clean!")
        if not in_place and target_dir != input_dir and not target_dir.exists():
            logger.info("Copying clean parquet files to target directory...")
            target_dir.mkdir(parents=True, exist_ok=True)
            for f in input_files:
                shutil.copy2(f, target_dir / Path(f).name)
        return True

    # Show sample duplicate IDs for diagnostic visibility
    sample_dups = con.sql(f"""
        SELECT id, count(*) AS count
        FROM read_parquet('{input_glob}')
        GROUP BY id
        HAVING count(*) > 1
        ORDER BY count DESC
        LIMIT 5
    """).df()
    logger.info(f"Sample duplicate arXiv IDs:\n{sample_dups.to_string(index=False)}")

    # Step 2: Create deduplicated view/table keeping latest update_date
    logger.info("Deduplicating records (keeping latest update_date for each arXiv ID)...")
    start_time = time.time()

    # We partition clean rows across output shards deterministically
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE deduplicated_arxiv AS
        SELECT
            * EXCLUDE (row_num),
            (ROW_NUMBER() OVER () - 1) % {shards_to_write} AS shard_id
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY id
                    ORDER BY update_date DESC NULLS LAST
                ) AS row_num
            FROM read_parquet('{input_glob}')
        )
        WHERE row_num = 1
    """)

    dedup_count = con.sql("SELECT count(*) FROM deduplicated_arxiv").fetchone()[0]
    logger.info(f"Deduplication finished in {time.time() - start_time:.2f}s. Clean records: {dedup_count:,}")
    assert dedup_count == distinct_count, f"Mismatch: expected {distinct_count}, got {dedup_count}"

    # Step 3: Write deduplicated shards to a staging directory
    stage_dir = Path(tempfile.mkdtemp(prefix="arxiv_dedup_stage_"))
    logger.info(f"Writing {shards_to_write} deduplicated parquet shard(s) to staging directory: {stage_dir}")

    write_start = time.time()
    for shard_idx in range(shards_to_write):
        shard_filename = f"train-{shard_idx:05d}-of-{shards_to_write:05d}.parquet"
        shard_dest = stage_dir / shard_filename
        logger.info(f"Exporting shard {shard_idx + 1}/{shards_to_write}: {shard_filename}...")
        con.sql(f"""
            COPY (
                SELECT * EXCLUDE (shard_id)
                FROM deduplicated_arxiv
                WHERE shard_id = {shard_idx}
            ) TO '{shard_dest}' (FORMAT PARQUET, COMPRESSION '{compression}')
        """)

    logger.info(f"Finished writing parquet files in {time.time() - write_start:.2f}s.")

    # Step 4: Validate exported parquet files
    logger.info("Validating exported parquet files...")
    stage_glob = str(stage_dir / "*.parquet")
    val_stats = con.sql(f"""
        SELECT
            count(*) AS total_count,
            count(DISTINCT id) AS distinct_count
        FROM read_parquet('{stage_glob}')
    """).fetchone()

    val_total, val_distinct = val_stats[0], val_stats[1]
    logger.info(f"Validation total: {val_total:,}, distinct IDs: {val_distinct:,}")

    if val_total != distinct_count or val_total != val_distinct:
        logger.error(
            f"Validation failed! Expected {distinct_count} records with 0 duplicates, "
            f"but found {val_total} records and {val_distinct} distinct IDs."
        )
        shutil.rmtree(stage_dir, ignore_errors=True)
        return False

    logger.info("Validation passed! Zero duplicates in exported parquet files.")

    # Step 5: Finalize output (in-place replacement or copy to target_dir)
    if in_place:
        if backup:
            backup_dir = input_dir.parent / f"{input_dir.name}_backup_{int(time.time())}"
            logger.info(f"Creating backup of original files in: {backup_dir}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            for f in input_files:
                shutil.move(f, backup_dir / Path(f).name)
        else:
            logger.info("Removing original duplicate-containing parquet files...")
            for f in input_files:
                os.remove(f)

        logger.info(f"Moving clean parquet files into {input_dir}...")
        for clean_file in sorted(stage_dir.glob("*.parquet")):
            shutil.move(str(clean_file), str(input_dir / clean_file.name))
        shutil.rmtree(stage_dir, ignore_errors=True)
        logger.info(f"In-place replacement complete! {input_dir} now contains {shards_to_write} clean shards.")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Moving clean parquet files into {target_dir}...")
        # Clear existing parquet files in target_dir if any
        for old_f in target_dir.glob("*.parquet"):
            os.remove(old_f)
        for clean_file in sorted(stage_dir.glob("*.parquet")):
            shutil.move(str(clean_file), str(target_dir / clean_file.name))
        shutil.rmtree(stage_dir, ignore_errors=True)
        logger.info(f"Clean parquet files ready in: {target_dir}")

    logger.info(f"Done! Successfully removed {duplicate_count:,} duplicate records.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate arXiv metadata parquet files by arXiv ID, keeping the latest update_date."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the input parquet files. Defaults to data/arxiv_metadata_snapshot/data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save clean parquet files. Defaults to '<input-dir>_clean' unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Safely replace the original parquet files in-place with the deduplicated files.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="When using --in-place, back up original files into '<input-dir>_backup_<timestamp>'.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of output parquet files (shards). Defaults to the number of input files.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "none"],
        help="Parquet compression codec (default: zstd).",
    )

    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir else resolve_default_paths()

    success = deduplicate_parquet_files(
        input_dir=input_dir,
        output_dir=args.output_dir,
        in_place=args.in_place,
        backup=args.backup,
        num_shards=args.num_shards,
        compression=args.compression,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
