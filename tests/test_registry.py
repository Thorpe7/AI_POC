"""Tests for the ModelRegistry: init, get_or_load, eviction, and weight download."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── TestRegistryInit ──


class TestRegistryInit:
    """Tests for registry initialization from config files."""

    def test_loads_configs(self, configs_dir: Path, sample_medgemma_config: dict[str, Any]) -> None:
        """Registry loads and parses valid config files."""
        with (
            patch("handlers.registry.torch") as mock_torch,
            patch("handlers.registry.boto3"),
            patch.dict(os.environ, {"WEIGHTS_CACHE_DIR": str(configs_dir.parent / "w")}),
        ):
            mock_torch.cuda.is_available.return_value = False
            (configs_dir.parent / "w").mkdir(exist_ok=True)

            from handlers.registry import ModelRegistry

            registry = ModelRegistry(configs_dir)

        assert "medgemma" in registry.configs
        assert registry.configs["medgemma"]["display_name"] == "MedGemma 4B"

    def test_fails_on_empty_dir(self, tmp_path: Path) -> None:
        """Registry raises ValueError when no configs found."""
        empty = tmp_path / "empty_configs"
        empty.mkdir()

        with (
            patch("handlers.registry.torch") as mock_torch,
            patch("handlers.registry.boto3"),
            patch.dict(os.environ, {"WEIGHTS_CACHE_DIR": str(tmp_path / "w")}),
        ):
            mock_torch.cuda.is_available.return_value = False
            (tmp_path / "w").mkdir(exist_ok=True)

            from handlers.registry import ModelRegistry

            with pytest.raises(ValueError, match="No valid config"):
                ModelRegistry(empty)

    def test_fails_on_missing_dir(self, tmp_path: Path) -> None:
        """Registry raises FileNotFoundError for non-existent directory."""
        from handlers.registry import ModelRegistry

        with pytest.raises(FileNotFoundError):
            ModelRegistry(tmp_path / "nonexistent")

    def test_skips_invalid_config(
        self, tmp_path: Path, sample_medgemma_config: dict[str, Any]
    ) -> None:
        """Registry skips configs missing required fields but loads valid ones."""
        configs = tmp_path / "configs"
        configs.mkdir()
        # Valid config
        (configs / "medgemma.json").write_text(json.dumps(sample_medgemma_config))
        # Invalid config (missing most fields)
        (configs / "broken.json").write_text(json.dumps({"display_name": "Broken"}))

        with (
            patch("handlers.registry.torch") as mock_torch,
            patch("handlers.registry.boto3"),
            patch.dict(os.environ, {"WEIGHTS_CACHE_DIR": str(tmp_path / "w")}),
        ):
            mock_torch.cuda.is_available.return_value = False
            (tmp_path / "w").mkdir(exist_ok=True)

            from handlers.registry import ModelRegistry

            registry = ModelRegistry(configs)

        assert "medgemma" in registry.configs
        assert "broken" not in registry.configs

    def test_reports_available_models(self, mock_registry: object) -> None:
        """get_available_models returns model summaries."""
        models = mock_registry.get_available_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "medgemma"
        assert models[0]["display_name"] == "MedGemma 4B"


# ── TestRegistryGetOrLoad ──


class TestRegistryGetOrLoad:
    """Tests for get_or_load with mocked S3 and HFHandler."""

    def test_cache_miss_downloads_and_loads(self, mock_registry: object) -> None:
        """Cache miss triggers download, load, and cache."""
        mock_handler = MagicMock()
        mock_model_dict = {"model": MagicMock(), "processor": MagicMock()}
        mock_handler.model_fn.return_value = mock_model_dict

        with (
            patch.object(
                mock_registry, "_download_weights", return_value=Path("/tmp/w")
            ) as mock_dl,
            patch("handlers.registry.HFHandler", return_value=mock_handler),
        ):
            handler, model_dict = mock_registry.get_or_load("medgemma")

        mock_dl.assert_called_once()
        assert "medgemma" in mock_registry.cache
        assert model_dict == mock_model_dict

    def test_cache_hit_returns_immediately(self, mock_registry: object) -> None:
        """Cache hit returns without downloading."""
        mock_model_dict = {"model": MagicMock(), "processor": MagicMock()}
        mock_handler = MagicMock()
        mock_registry.cache["medgemma"] = mock_model_dict
        mock_registry.handlers["medgemma"] = mock_handler

        with patch.object(mock_registry, "_download_weights") as mock_dl:
            handler, model_dict = mock_registry.get_or_load("medgemma")

        mock_dl.assert_not_called()
        assert model_dict == mock_model_dict
        assert handler == mock_handler

    def test_cache_hit_updates_lru_order(self, mock_registry: object) -> None:
        """Cache hit moves the entry to the end (MRU position)."""
        # Pre-populate cache with two models
        mock_registry.cache["model_a"] = {"model": MagicMock()}
        mock_registry.handlers["model_a"] = MagicMock()
        mock_registry.cache["medgemma"] = {"model": MagicMock()}
        mock_registry.handlers["medgemma"] = MagicMock()
        # Add model_a config
        mock_registry.configs["model_a"] = {
            **mock_registry.configs["medgemma"],
            "display_name": "Model A",
        }

        mock_registry.get_or_load("model_a")

        # model_a should now be at the end (MRU)
        keys = list(mock_registry.cache.keys())
        assert keys[-1] == "model_a"

    def test_eviction_when_memory_insufficient(
        self, mock_registry: object, sample_medgemma_config: dict[str, Any]
    ) -> None:
        """When memory is insufficient, LRU model is evicted."""
        # Fill the registry with a fake model using most memory
        mock_registry.used_gpu_memory_gb = 15.0
        mock_registry.cache["old_model"] = {"model": MagicMock(), "processor": MagicMock()}
        mock_registry.handlers["old_model"] = MagicMock()
        mock_registry.configs["old_model"] = {
            **sample_medgemma_config,
            "estimated_gpu_memory_gb": 15.0,
        }

        mock_handler = MagicMock()
        mock_handler.model_fn.return_value = {"model": MagicMock(), "processor": MagicMock()}

        with (
            patch.object(mock_registry, "_download_weights", return_value=Path("/tmp/w")),
            patch("handlers.registry.HFHandler", return_value=mock_handler),
            patch.object(mock_registry, "_unload_model") as mock_unload,
        ):
            mock_registry.get_or_load("medgemma")

        # old_model should have been evicted
        assert "old_model" not in mock_registry.cache
        mock_unload.assert_called_once()
        assert "medgemma" in mock_registry.cache

    def test_model_exceeds_total_gpu_raises(self, mock_registry: object) -> None:
        """Model requiring more than total GPU memory raises RuntimeError."""
        # Override config to require absurd memory
        mock_registry.configs["medgemma"]["estimated_gpu_memory_gb"] = 999.0

        with pytest.raises(RuntimeError, match="total GPU available"):
            mock_registry.get_or_load("medgemma")

    def test_unknown_model_raises(self, mock_registry: object) -> None:
        """Unknown model ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            mock_registry.get_or_load("nonexistent")

    def test_s3_failure_propagates(self, mock_registry: object) -> None:
        """S3 download failure raises RuntimeError."""
        with patch.object(
            mock_registry,
            "_download_weights",
            side_effect=RuntimeError("S3 error"),
        ):
            with pytest.raises(RuntimeError, match="S3 error"):
                mock_registry.get_or_load("medgemma")


# ── TestRegistryEviction ──


class TestRegistryEviction:
    """Tests for LRU eviction behavior."""

    def test_evicts_lru_specifically(
        self, mock_registry: object, sample_medgemma_config: dict[str, Any]
    ) -> None:
        """Eviction removes the least recently used model (first in OrderedDict)."""
        mock_registry.cache["model_a"] = {"model": MagicMock(), "processor": MagicMock()}
        mock_registry.handlers["model_a"] = MagicMock()
        mock_registry.configs["model_a"] = {
            **sample_medgemma_config,
            "estimated_gpu_memory_gb": 5.0,
        }
        mock_registry.cache["model_b"] = {"model": MagicMock(), "processor": MagicMock()}
        mock_registry.handlers["model_b"] = MagicMock()
        mock_registry.configs["model_b"] = {
            **sample_medgemma_config,
            "estimated_gpu_memory_gb": 5.0,
        }
        mock_registry.used_gpu_memory_gb = 10.0

        with patch.object(mock_registry, "_unload_model"):
            mock_registry._evict_lru()

        # model_a was first (LRU), should be evicted
        assert "model_a" not in mock_registry.cache
        assert "model_b" in mock_registry.cache

    def test_memory_accounting_correct(
        self, mock_registry: object, sample_medgemma_config: dict[str, Any]
    ) -> None:
        """Eviction correctly updates used_gpu_memory_gb."""
        mock_registry.cache["model_a"] = {"model": MagicMock()}
        mock_registry.handlers["model_a"] = MagicMock()
        mock_registry.configs["model_a"] = {
            **sample_medgemma_config,
            "estimated_gpu_memory_gb": 8.0,
        }
        mock_registry.used_gpu_memory_gb = 8.0

        with patch.object(mock_registry, "_unload_model"):
            mock_registry._evict_lru()

        assert mock_registry.used_gpu_memory_gb == 0.0


# ── TestRegistryWeightDownload ──


class TestRegistryWeightDownload:
    """Tests for _download_weights with mocked boto3."""

    def test_downloads_s3_prefix(
        self, mock_registry: object, sample_medgemma_config: dict[str, Any]
    ) -> None:
        """Downloads all files under an S3 prefix."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "weights/medgemma/config.json"},
                    {"Key": "weights/medgemma/model.safetensors"},
                ]
            }
        ]

        with patch("handlers.registry.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            result = mock_registry._download_weights("medgemma", sample_medgemma_config)

        assert result == mock_registry.weights_cache_dir / "medgemma"
        assert mock_s3.download_file.call_count == 2

    def test_skips_if_cached(
        self, mock_registry: object, sample_medgemma_config: dict[str, Any]
    ) -> None:
        """Skips S3 download if local cache exists and is non-empty."""
        local_dir = mock_registry.weights_cache_dir / "medgemma"
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")

        with patch("handlers.registry.boto3") as mock_boto3:
            result = mock_registry._download_weights("medgemma", sample_medgemma_config)

        mock_boto3.client.assert_not_called()
        assert result == local_dir
