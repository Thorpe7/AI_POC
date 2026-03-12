"""Package MedGemma model weights and inference code into a model.tar.gz for SageMaker."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

REQUIRED_DISK_GB = 18
MODEL_PAGE_URL = "https://huggingface.co/google/medgemma-4b"
IGNORE_PATTERNS = ["*.gguf", "*.md", ".gitattributes"]
CODE_REQUIREMENTS = """\
# HF DLC includes torch, transformers, accelerate, Pillow.
# Only add deps missing from the container image.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Package MedGemma model artifacts into a SageMaker-compatible tar archive."
    )
    parser.add_argument(
        "--model-id",
        default="google/medgemma-1.5-4b-it",
        help="HuggingFace model ID (default: google/medgemma-1.5-4b-it)",
    )
    parser.add_argument(
        "--output",
        default="./model.tar.gz",
        help="Output archive path (default: ./model.tar.gz)",
    )
    parser.add_argument(
        "--work-dir",
        default="./model_staging",
        help="Temporary staging directory (default: ./model_staging)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace auth token (default: $MED_GEM_TOKEN env var)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip gzip compression (faster, but SageMaker requires .tar.gz)",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Don't delete staging directory after packaging",
    )
    return parser.parse_args(argv)


def get_hf_token(cli_token: str | None) -> str:
    """Resolve the HuggingFace token from CLI arg or environment variable."""
    token = cli_token or os.environ.get("MED_GEM_TOKEN")
    if not token:
        print(
            "Error: No HuggingFace token provided.\n"
            "Set MED_GEM_TOKEN environment variable or pass --hf-token.\n"
            f"You must also accept the model license at {MODEL_PAGE_URL}",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def check_disk_space(path: Path, required_gb: float) -> None:
    """Verify sufficient disk space is available."""
    stat = shutil.disk_usage(path.parent if path.parent.exists() else Path.cwd())
    free_gb = stat.free / (1024**3)
    if free_gb < required_gb:
        print(
            f"Error: Insufficient disk space. Need ~{required_gb:.0f} GB, "
            f"have {free_gb:.1f} GB free.",
            file=sys.stderr,
        )
        sys.exit(1)


def download_model(model_id: str, staging_dir: Path, token: str) -> Path:
    """Download model snapshot from HuggingFace Hub."""
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    from huggingface_hub.errors import GatedRepoError  # type: ignore[import-not-found]

    model_dir = staging_dir / "model"
    print(f"Downloading {model_id} to {model_dir} ...")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            token=token,
            ignore_patterns=IGNORE_PATTERNS,
        )
    except GatedRepoError:
        print(
            f"Error: Access denied — {model_id} is a gated model.\n"
            f"Accept the license at {MODEL_PAGE_URL} then retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Download complete: {model_dir}")
    return model_dir


def stage_code(model_dir: Path) -> None:
    """Copy inference code and write requirements into the code/ subdirectory."""
    code_dir = model_dir / "code"
    code_dir.mkdir(exist_ok=True)

    inference_src = Path(__file__).resolve().parent / "inference.py"
    if not inference_src.exists():
        print(f"Error: {inference_src} not found.", file=sys.stderr)
        sys.exit(1)

    shutil.copy2(inference_src, code_dir / "inference.py")
    (code_dir / "requirements.txt").write_text(CODE_REQUIREMENTS)
    print(f"Staged code/ directory: {code_dir}")


def clean_staging(model_dir: Path) -> None:
    """Remove HuggingFace cache metadata from staging directory."""
    hf_cache = model_dir / ".cache"
    if hf_cache.exists():
        shutil.rmtree(hf_cache)
        print("Cleaned .cache/ from staging directory")


def create_archive(model_dir: Path, output_path: Path, *, compress: bool) -> None:
    """Create tar archive with model artifacts at the root level."""
    mode = "w:gz" if compress else "w"
    suffix = "tar.gz" if compress else "tar"
    print(f"Creating {suffix} archive at {output_path} ...")

    with tarfile.open(str(output_path), mode) as tar:  # type: ignore[call-overload]
        for item in sorted(model_dir.iterdir()):
            tar.add(str(item), arcname=item.name)

    size_gb = output_path.stat().st_size / (1024**3)
    print(f"Archive created: {output_path} ({size_gb:.2f} GB)")


def count_archive_files(output_path: Path) -> int:
    """Count files in the archive."""
    with tarfile.open(output_path, "r:*") as tar:
        return len(tar.getnames())


def main(argv: list[str] | None = None) -> None:
    """Package MedGemma model for SageMaker deployment."""
    args = parse_args(argv)

    token = get_hf_token(args.hf_token)
    output_path = Path(args.output).resolve()
    staging_dir = Path(args.work_dir).resolve()

    check_disk_space(staging_dir, REQUIRED_DISK_GB)

    staging_dir.mkdir(parents=True, exist_ok=True)

    model_dir = download_model(args.model_id, staging_dir, token)
    stage_code(model_dir)
    clean_staging(model_dir)
    create_archive(model_dir, output_path, compress=not args.no_compress)

    file_count = count_archive_files(output_path)
    print(f"\nDone! {file_count} files archived.")

    if not args.keep_staging:
        shutil.rmtree(staging_dir)
        print(f"Cleaned up staging directory: {staging_dir}")


if __name__ == "__main__":
    main()
