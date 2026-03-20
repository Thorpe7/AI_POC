"""Model registry with on-demand loading, LRU caching, and GPU memory management."""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import torch

from handlers.hf_handler import HFHandler
from handlers.utils import parse_s3_uri

logger = logging.getLogger(__name__)

REQUIRED_CONFIG_FIELDS = {
    "auto_class",
    "processor_class",
    "dtype",
    "device_map",
    "input_modality",
    "generation_params",
    "output_key",
    "estimated_gpu_memory_gb",
    "weights_s3_uri",
}

GPU_USABLE_FRACTION = 0.85


class ModelRegistry:
    """Manages on-demand model loading with LRU eviction when GPU memory is constrained.

    Models are loaded from S3 on first request, cached in GPU memory, and evicted
    (least recently used first) when a new model requires more memory than is available.
    Weight files are cached on disk even after GPU eviction to avoid re-downloading.
    """

    def __init__(self, configs_dir: str | Path) -> None:
        configs_dir = Path(configs_dir)
        if not configs_dir.is_dir():
            raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

        self.configs: dict[str, dict[str, Any]] = {}
        self._load_configs(configs_dir)

        if not self.configs:
            raise ValueError(f"No valid config files found in {configs_dir}")

        # LRU cache: oldest items first, newest (most recently used) at the end
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.handlers: dict[str, HFHandler] = {}
        self.weights_cache_dir = Path(os.environ.get("WEIGHTS_CACHE_DIR", "/tmp/model_weights"))
        self.weights_cache_dir.mkdir(parents=True, exist_ok=True)

        # GPU memory tracking
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            self.total_gpu_memory_gb = total * GPU_USABLE_FRACTION
        else:
            self.total_gpu_memory_gb = 0.0

        self.used_gpu_memory_gb = 0.0
        self._lock = threading.Lock()

        logger.info(
            "Registry initialized: %d configs, %.1f GB usable GPU",
            len(self.configs),
            self.total_gpu_memory_gb,
        )

    def _load_configs(self, configs_dir: Path) -> None:
        """Scan directory for *.json config files and validate required fields."""
        for path in sorted(configs_dir.glob("*.json")):
            model_id = path.stem
            try:
                with open(path) as f:
                    config = json.load(f)
                missing = REQUIRED_CONFIG_FIELDS - set(config.keys())
                if missing:
                    logger.warning("Skipping %s: missing fields %s", model_id, missing)
                    continue
                self.configs[model_id] = config
                logger.info("Loaded config: %s (%s)", model_id, config.get("display_name", ""))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", path.name, exc)

    def get_or_load(self, model_id: str) -> tuple[HFHandler, dict[str, Any]]:
        """Return the handler and loaded model dict for the given model ID.

        On cache miss: ensures GPU memory, downloads weights from S3 if needed,
        loads the model, and caches the result. On cache hit: moves the entry
        to the end of the LRU order.

        Args:
            model_id: Client-facing model identifier (matches config filename).

        Returns:
            Tuple of (handler, model_dict).

        Raises:
            ValueError: If model_id is not in the known configs.
            RuntimeError: If model exceeds total GPU memory or S3 download fails.
        """
        if model_id not in self.configs:
            raise ValueError(
                f"Unknown model: '{model_id}'. "
                f"Available: {list(self.configs.keys())}"
            )

        with self._lock:
            # Cache hit
            if model_id in self.cache:
                self.cache.move_to_end(model_id)
                logger.info("Cache hit: %s", model_id)
                return self.handlers[model_id], self.cache[model_id]

            # Cache miss — load the model
            config = self.configs[model_id]
            needed_gb = config["estimated_gpu_memory_gb"]

            self._ensure_memory(needed_gb)

            local_dir = self._download_weights(model_id, config)

            handler = HFHandler(config)
            try:
                model_dict = handler.model_fn(str(local_dir))
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM loading %s, evicting another model and retrying", model_id)
                if self.cache:
                    self._evict_lru()
                    model_dict = handler.model_fn(str(local_dir))
                else:
                    raise RuntimeError(
                        f"Out of GPU memory loading '{model_id}' and no models to evict"
                    )

            self.cache[model_id] = model_dict
            self.handlers[model_id] = handler
            self.used_gpu_memory_gb += needed_gb

            logger.info(
                "Loaded %s (%.1f GB used / %.1f GB total)",
                model_id,
                self.used_gpu_memory_gb,
                self.total_gpu_memory_gb,
            )
            return handler, model_dict

    def _ensure_memory(self, needed_gb: float) -> None:
        """Evict LRU models until there is enough estimated GPU memory.

        Raises:
            RuntimeError: If the model is larger than total available GPU memory.
        """
        if needed_gb > self.total_gpu_memory_gb and self.total_gpu_memory_gb > 0:
            raise RuntimeError(
                f"Model requires {needed_gb:.1f} GB but only "
                f"{self.total_gpu_memory_gb:.1f} GB total GPU available"
            )

        while (
            self.cache
            and (self.total_gpu_memory_gb - self.used_gpu_memory_gb) < needed_gb
        ):
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict the least recently used model from GPU memory."""
        evicted_id, model_dict = self.cache.popitem(last=False)
        evicted_gb = self.configs[evicted_id]["estimated_gpu_memory_gb"]

        self._unload_model(model_dict)
        del self.handlers[evicted_id]
        self.used_gpu_memory_gb -= evicted_gb

        logger.info("Evicted %s (freed %.1f GB)", evicted_id, evicted_gb)

    @staticmethod
    def _unload_model(model_dict: dict[str, Any]) -> None:
        """Move model to CPU and free GPU memory."""
        model = model_dict.get("model")
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
            del model_dict["model"]

        processor = model_dict.get("processor")
        if processor is not None:
            del model_dict["processor"]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _download_weights(self, model_id: str, config: dict[str, Any]) -> Path:
        """Download model weights from S3 to a local cache directory.

        Skips download if the local directory already exists and is non-empty.

        Args:
            model_id: Model identifier used as the local subdirectory name.
            config: Model config containing ``weights_s3_uri``.

        Returns:
            Path to the local directory containing weight files.

        Raises:
            RuntimeError: If S3 download fails.
        """
        local_dir = self.weights_cache_dir / model_id
        if local_dir.exists() and any(local_dir.iterdir()):
            logger.info("Using cached weights for %s at %s", model_id, local_dir)
            return local_dir

        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            bucket, prefix = parse_s3_uri(config["weights_s3_uri"])
            s3 = boto3.client("s3")
            paginator = s3.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # Derive local filename relative to the prefix
                    relative = key[len(prefix) :].lstrip("/")
                    if not relative:
                        continue

                    local_file = local_dir / relative
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    logger.info("Downloading s3://%s/%s → %s", bucket, key, local_file)
                    s3.download_file(bucket, key, str(local_file))

        except Exception as exc:
            raise RuntimeError(
                f"Failed to download weights for '{model_id}' "
                f"from {config['weights_s3_uri']}: {exc}"
            ) from exc

        return local_dir

    def get_config(self, model_id: str) -> dict[str, Any]:
        """Return the config dict for a model ID.

        Raises:
            ValueError: If model_id is not known.
        """
        if model_id not in self.configs:
            raise ValueError(f"Unknown model: '{model_id}'")
        return self.configs[model_id]

    def is_known_model(self, model_id: str) -> bool:
        """Return True if the model ID has a valid config."""
        return model_id in self.configs

    def get_available_models(self) -> list[dict[str, str]]:
        """Return a list of available model summaries for API consumers."""
        return [
            {"model_id": mid, "display_name": cfg.get("display_name", mid)}
            for mid, cfg in self.configs.items()
        ]
