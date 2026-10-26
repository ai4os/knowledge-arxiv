"""Download datasets from Hugging Face."""

from pathlib import Path

from huggingface_hub import snapshot_download


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    snapshot_download(
        repo_id="librarian-bots/arxiv-metadata-snapshot",
        repo_type="dataset",
        local_dir=DATA_DIR / "arxiv_metadata_snapshot",
    )

    snapshot_download(
        repo_id="ifca-advanced-computing/DIGITAL.CSIC-metadata-snapshot",
        repo_type="dataset",
        local_dir=DATA_DIR / "csic_metadata_snapshot",
    )

    snapshot_download(
        repo_id="ifca-advanced-computing/HAL-metadata-snapshot",
        repo_type="dataset",
        local_dir=DATA_DIR / "hal_metadata_snapshot",
    )
