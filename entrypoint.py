"""Entrypoint for the KServe inference container."""

import os

# OpenCV prepends `.../cv2/../../lib64` to LD_LIBRARY_PATH on import, which can
# shadow the system CUDA libs at dlopen time (vllm#35608). Strip it before any
# import path that cascades into torch/vllm.
_ORIG_LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", "")
if "/cv2/" in _ORIG_LD_LIBRARY_PATH:
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p for p in _ORIG_LD_LIBRARY_PATH.split(":") if "/cv2/" not in p
    )

import argparse
import logging
import sys

from kserve import ModelServer

from src.kserve_model_class import KserveModelHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("entrypoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--handler_type", required=True, help="Handler family (e.g. vllm, segmentation)")
    parser.add_argument("--model_name", required=True, help="Model key (e.g. medgemma)")
    parser.add_argument("--model_dir", required=True, help="Path to model weights on the mounted volume")
    parser.add_argument("--dtype", default="bfloat16", help="vLLM load dtype")
    parser.add_argument("--max_model_len", type=int, default=32768, help="vLLM max context length")
    args = parser.parse_args()

    if _ORIG_LD_LIBRARY_PATH != os.environ.get("LD_LIBRARY_PATH", ""):
        log.info(
            "sanitized LD_LIBRARY_PATH (stripped cv2 prefix): before=%s after=%s",
            _ORIG_LD_LIBRARY_PATH,
            os.environ.get("LD_LIBRARY_PATH", ""),
        )

    config = {
        "handler_type": args.handler_type,
        "model_name": args.model_name,
        "model_dir": args.model_dir,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
    }
    log.info("starting container with config: %s", config)

    log.info("initializing KserveModelHandler")
    handler = KserveModelHandler(
        name=args.model_name,
        config=config,
    )
    log.info("handler ready, starting ModelServer")

    ModelServer().start(models=[handler])
