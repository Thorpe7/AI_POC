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


def _str_bool(s: str) -> bool:
    # argparse type=bool is a trap — any non-empty string (incl. "false") is True.
    return s.strip().lower() in ("1", "true", "yes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--handler_type", required=True, help="Handler family (vllm, totalseg)")
    parser.add_argument("--model_name", required=True, help="Model key (e.g. medgemma, llava-med, totalseg)")
    parser.add_argument("--model_dir", default=None, help="Path to model weights on the mounted volume (vllm handlers only)")
    parser.add_argument("--dtype", default="bfloat16", help="vLLM load dtype")
    parser.add_argument("--max_model_len", type=int, default=32768, help="vLLM max context length")
    parser.add_argument("--image_size", type=int, default=896, help="Square edge length for preprocessed DICOM images")
    parser.add_argument("--max_images", type=int, default=32, help="Max images per request (also sets vLLM limit_mm_per_prompt)")
    parser.add_argument("--output_root", default="/data/xnat/archive/AI-POC/resources/output", help="Root directory for handler output (totalseg handler)")
    parser.add_argument("--task_default", default="total", help="Default TotalSegmentator task")
    parser.add_argument("--fast_default", type=_str_bool, default=True, help="Default value for the 'fast' request field (totalseg handler)")
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
        "image_size": args.image_size,
        "max_images": args.max_images,
        "output_root": args.output_root,
        "task_default": args.task_default,
        "fast_default": args.fast_default,
    }
    log.info("starting container with config: %s", config)

    log.info("initializing KserveModelHandler")
    handler = KserveModelHandler(
        name=args.model_name,
        config=config,
    )
    log.info("handler ready, starting ModelServer")

    ModelServer().start(models=[handler])
