"""Download the arXiv metadata snapshot dataset from Hugging Face.

Dataset: https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot
"""

from huggingface_hub import snapshot_download

REPO_ID = "librarian-bots/arxiv-metadata-snapshot"
LOCAL_DIR = "../data/arxiv_metadata_snapshot"

if __name__ == "__main__":
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )
