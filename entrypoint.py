"""Entrypoint for the KServe inference container."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from kserve import ModelServer

from src.kserve_model_class import KserveModelHandler

CONFIG_DIR = Path(__file__).parent / "src" / "handlers" / "handler_configs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("entrypoint")


def load_config(model_name: str) -> dict:
    """Load the handler config JSON for the given model name."""
    config_path = CONFIG_DIR / f"{model_name}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {config_path}")
    with open(config_path) as f:
        return json.load(f)


def dump_mounts() -> None:
    """Log /proc/mounts entries relevant to the model_dir path."""
    try:
        with open("/proc/mounts") as f:
            lines = f.readlines()
    except OSError as e:
        log.error("could not read /proc/mounts: %s", e)
        return

    log.info("/proc/mounts (%d entries total):", len(lines))
    for line in lines:
        if any(tok in line for tok in ("/data", "xnat", "pvc", "persistent")):
            log.info("  %s", line.rstrip())


def walk_ancestry(target: Path) -> None:
    """Walk from / down to target, logging contents at each level.

    Makes it obvious at which level the path stops existing — e.g. does
    /data exist but /data/xnat missing (mount not landing), or does
    /data/xnat/archive/AI-POC exist but the subpath to the weights is
    absent (mount OK but wrong content)?
    """
    parts = target.parts
    current = Path(parts[0])
    for part in parts[1:]:
        if not current.exists():
            log.error("ancestor does not exist: %s", current)
            return
        if not current.is_dir():
            log.error("ancestor is not a directory: %s", current)
            return
        try:
            entries = sorted(current.iterdir())
        except PermissionError as e:
            log.error("PermissionError listing %s: %s", current, e)
            return
        names = [e.name + ("/" if e.is_dir() else "") for e in entries[:50]]
        log.info("contents of %s (%d entries): %s", current, len(entries), names)
        current = current / part

    if current.exists():
        log.info("target %s EXISTS", current)
    else:
        log.error("target %s does NOT exist (but parent did)", current)


def verify_model_dir_readable(model_dir: Path) -> None:
    """Verify the container can read the model weights directory.

    Runs before any model-loading library touches the path so a permission
    block surfaces here with a clear error instead of downstream as an
    unrelated HuggingFace repo-id validation error.
    """
    log.info(
        "process identity: uid=%d gid=%d groups=%s",
        os.geteuid(),
        os.getegid(),
        os.getgroups(),
    )
    log.info("checking model_dir: %s", model_dir)

    dump_mounts()

    if not model_dir.exists():
        log.error("model_dir missing — dumping ancestor tree:")
        walk_ancestry(model_dir)
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    st = model_dir.stat()
    log.info(
        "model_dir stat: mode=%o owner_uid=%d owner_gid=%d is_dir=%s",
        st.st_mode & 0o7777,
        st.st_uid,
        st.st_gid,
        model_dir.is_dir(),
    )

    entries = list(model_dir.iterdir())
    log.info("model_dir contains %d entries", len(entries))
    for entry in entries[:20]:
        try:
            est = entry.stat()
            log.info(
                "  %s mode=%o uid=%d gid=%d size=%d",
                entry.name,
                est.st_mode & 0o7777,
                est.st_uid,
                est.st_gid,
                est.st_size,
            )
        except PermissionError as e:
            log.error("  %s PermissionError on stat: %s", entry.name, e)

    config_json = model_dir / "config.json"
    if not config_json.exists():
        log.warning("config.json not present in model_dir — unusual for HF layout")
        return

    with open(config_json, "rb") as f:
        head = f.read(64)
    log.info("read %d bytes from %s", len(head), config_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Model key (e.g. medgemma)")
    args = parser.parse_args()

    log.info("starting container for model_name=%s", args.model_name)

    config = load_config(args.model_name)
    log.info("loaded config: %s", config)

    verify_model_dir_readable(Path(config["model_dir"]))

    log.info("initializing KserveModelHandler")
    handler = KserveModelHandler(
        name=args.model_name,
        config=config,
    )
    log.info("handler ready, starting ModelServer")

    ModelServer().start(models=[handler])
