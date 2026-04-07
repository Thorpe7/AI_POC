"""Upload HuggingFace model weights to S3 for use with the multi-model endpoint.

One-time setup per model. Downloads from HF Hub, uploads each file to an S3 prefix.

Usage:
    python src/upload_weights.py \
        --model-id google/medgemma-1.5-4b-it \
        --s3-uri s3://bucket/weights/medgemma/ \
        [--hf-token $HF_TOKEN]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download HF model weights and upload to S3."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID (e.g. google/medgemma-1.5-4b-it)",
    )
    parser.add_argument(
        "--s3-uri",
        required=True,
        help="S3 URI prefix for weight upload (e.g. s3://bucket/weights/medgemma/)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace auth token (default: $HF_TOKEN or $MED_GEM_TOKEN)",
    )
    parser.add_argument(
        "--work-dir",
        default="./weight_staging",
        help="Temporary local directory for downloads (default: ./weight_staging)",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Don't delete staging directory after upload",
    )
    return parser.parse_args(argv)


def get_hf_token(cli_token: str | None) -> str | None:
    """Resolve HuggingFace token from CLI arg or environment."""
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("MED_GEM_TOKEN")


def download_from_hf(model_id: str, local_dir: Path, token: str | None) -> Path:
    """Download model snapshot from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID.
        local_dir: Local directory to download into.
        token: HuggingFace auth token, or None.

    Returns:
        Path to the download directory.
    """
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    from huggingface_hub.errors import GatedRepoError  # type: ignore[import-not-found]

    ignore_patterns = ["*.gguf", "*.md", ".gitattributes"]
    print(f"Downloading {model_id} to {local_dir} ...")

    if not token:
        print(
            "Warning: No HuggingFace token provided. Gated models will fail.\n"
            "Set HF_TOKEN or MED_GEM_TOKEN, or pass --hf-token.",
            file=sys.stderr,
        )

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=token,
            ignore_patterns=ignore_patterns,
        )
    except GatedRepoError:
        print(
            f"Error: Access denied — {model_id} is a gated model.\n"
            f"1. Accept the license at https://huggingface.co/{model_id}\n"
            f"2. Provide a token via --hf-token or HF_TOKEN env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Download complete: {local_dir}")
    return local_dir


def upload_to_s3(local_dir: Path, s3_uri: str) -> int:
    """Upload all files from a local directory to an S3 prefix.

    Args:
        local_dir: Directory containing files to upload.
        s3_uri: S3 URI prefix (e.g. s3://bucket/weights/medgemma/).

    Returns:
        Number of files uploaded.
    """
    import re

    import boto3  # type: ignore[import-untyped]

    match = re.match(r"^s3://([^/]+)/(.+?)/?$", s3_uri)
    if not match:
        print(f"Error: Invalid S3 URI: {s3_uri}", file=sys.stderr)
        sys.exit(1)

    bucket = match.group(1)
    prefix = match.group(2).rstrip("/")
    s3 = boto3.client("s3")

    count = 0
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        # Skip HF cache metadata
        if ".cache" in file_path.parts:
            continue

        relative = file_path.relative_to(local_dir)
        s3_key = f"{prefix}/{relative}"

        print(f"Uploading {relative} → s3://{bucket}/{s3_key}")
        s3.upload_file(str(file_path), bucket, s3_key)
        count += 1

    return count


def main(argv: list[str] | None = None) -> None:
    """Download HF weights and upload to S3."""
    args = parse_args(argv)

    token = get_hf_token(args.hf_token)
    work_dir = Path(args.work_dir).resolve()
    local_dir = work_dir / "model"

    work_dir.mkdir(parents=True, exist_ok=True)

    download_from_hf(args.model_id, local_dir, token)
    count = upload_to_s3(local_dir, args.s3_uri)

    print(f"\nDone! Uploaded {count} files to {args.s3_uri}")

    if not args.keep_staging:
        import shutil

        shutil.rmtree(work_dir)
        print(f"Cleaned up staging directory: {work_dir}")


if __name__ == "__main__":
    main()
