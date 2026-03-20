"""Package inference code and configs into a lightweight model.tar.gz for SageMaker.

No model weights are included — weights are stored separately in S3 and downloaded
at inference time by the ModelRegistry. This keeps the tarball small (KB, not GB).

Tarball structure:
    configs/
        medgemma.json
        (future models...)
    code/
        inference.py          <- shim (from src/shims/hf_inference.py)
        requirements.txt
        handlers/
            __init__.py
            base.py
            hf_handler.py
            registry.py
            utils.py
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

CODE_REQUIREMENTS = """\
# HF DLC includes torch, transformers, accelerate, Pillow.
# Only add deps missing from the container image.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Package multi-model inference code and configs into a SageMaker tarball."
    )
    parser.add_argument(
        "--output",
        default="./model.tar.gz",
        help="Output archive path (default: ./model.tar.gz)",
    )
    parser.add_argument(
        "--configs-dir",
        default=None,
        help="Path to configs directory (default: src/configs/ relative to this script)",
    )
    parser.add_argument(
        "--work-dir",
        default="./model_staging",
        help="Temporary staging directory (default: ./model_staging)",
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


def stage_code(staging_dir: Path, configs_dir: Path) -> None:
    """Assemble the tarball directory structure under staging_dir.

    Creates:
        staging_dir/configs/*.json       (at tarball root)
        staging_dir/code/inference.py    (shim)
        staging_dir/code/requirements.txt
        staging_dir/code/handlers/       (handler package)
    """
    src_root = Path(__file__).resolve().parent

    # --- configs/ at tarball root ---
    dest_configs = staging_dir / "configs"
    if dest_configs.exists():
        shutil.rmtree(dest_configs)
    shutil.copytree(
        configs_dir,
        dest_configs,
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    # --- code/ directory ---
    code_dir = staging_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    # Copy shim as inference.py (SageMaker entry point)
    shim_src = src_root / "shims" / "hf_inference.py"
    if not shim_src.exists():
        raise FileNotFoundError(f"Shim not found: {shim_src}")
    shutil.copy2(shim_src, code_dir / "inference.py")

    # Copy handlers package
    handlers_src = src_root / "handlers"
    if not handlers_src.exists():
        raise FileNotFoundError(f"Handlers package not found: {handlers_src}")
    dest_handlers = code_dir / "handlers"
    if dest_handlers.exists():
        shutil.rmtree(dest_handlers)
    shutil.copytree(
        handlers_src,
        dest_handlers,
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    # Write requirements.txt
    (code_dir / "requirements.txt").write_text(CODE_REQUIREMENTS)

    print(f"Staged tarball structure under {staging_dir}")


def create_archive(staging_dir: Path, output_path: Path, *, compress: bool) -> None:
    """Create tar archive with all staged content at the root level."""
    mode = "w:gz" if compress else "w"
    suffix = "tar.gz" if compress else "tar"
    print(f"Creating {suffix} archive at {output_path} ...")

    with tarfile.open(str(output_path), mode) as tar:  # type: ignore[call-overload]
        for item in sorted(staging_dir.iterdir()):
            tar.add(str(item), arcname=item.name)

    size_kb = output_path.stat().st_size / 1024
    print(f"Archive created: {output_path} ({size_kb:.1f} KB)")


def count_archive_files(output_path: Path) -> int:
    """Count files in the archive."""
    with tarfile.open(output_path, "r:*") as tar:
        return len(tar.getnames())


def main(argv: list[str] | None = None) -> None:
    """Package multi-model inference code for SageMaker deployment."""
    args = parse_args(argv)

    output_path = Path(args.output).resolve()
    staging_dir = Path(args.work_dir).resolve()

    # Resolve configs directory
    if args.configs_dir:
        configs_dir = Path(args.configs_dir).resolve()
    else:
        configs_dir = Path(__file__).resolve().parent / "configs"

    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    # Clean and create staging dir
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    stage_code(staging_dir, configs_dir)
    create_archive(staging_dir, output_path, compress=not args.no_compress)

    file_count = count_archive_files(output_path)
    print(f"\nDone! {file_count} files archived.")

    if not args.keep_staging:
        shutil.rmtree(staging_dir)
        print(f"Cleaned up staging directory: {staging_dir}")


if __name__ == "__main__":
    main()
