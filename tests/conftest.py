"""Shared fixtures for inference handler tests."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Ensure src/ is importable for handler tests
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ── Old inference.py fixtures (kept for backward compat) ──


@pytest.fixture
def sample_image_base64() -> str:
    """Create a tiny 8x8 PNG image encoded as a base64 string."""
    img = Image.new("RGB", (8, 8), color=(128, 0, 64))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def sample_request_body(sample_image_base64: str) -> str:
    """JSON request body containing both text and an image."""
    return json.dumps({"text": "Describe this medical image.", "image": sample_image_base64})


@pytest.fixture
def sample_text_only_request_body() -> str:
    """JSON request body containing only text."""
    return json.dumps({"text": "What are the symptoms of pneumonia?"})


@pytest.fixture
def mock_model_dict() -> dict[str, Any]:
    """Mock model and processor for testing predict_fn."""
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = MagicMock()

    mock_processor = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(
        side_effect=lambda key: MagicMock(shape=(1, 10)) if key == "input_ids" else MagicMock()
    )
    mock_inputs.to.return_value = mock_inputs
    mock_processor.apply_chat_template.return_value = mock_inputs
    mock_processor.decode.return_value = "Mock generated response."

    return {"model": mock_model, "processor": mock_processor}


# ── New multi-model fixtures ──


@pytest.fixture
def sample_medgemma_config() -> dict[str, Any]:
    """Config dict matching the medgemma.json schema."""
    return {
        "display_name": "MedGemma 4B",
        "hf_model_id": "google/medgemma-1.5-4b-it",
        "weights_s3_uri": "s3://test-bucket/weights/medgemma/",
        "auto_class": "AutoModelForImageTextToText",
        "processor_class": "AutoProcessor",
        "dtype": "bfloat16",
        "device_map": "auto",
        "input_modality": "text+image",
        "use_chat_template": True,
        "generation_params": {"max_new_tokens": 512, "do_sample": False},
        "output_key": "generated_text",
        "estimated_gpu_memory_gb": 9.0,
        "requires_hf_token": True,
    }


@pytest.fixture
def configs_dir(tmp_path: Path, sample_medgemma_config: dict[str, Any]) -> Path:
    """Temporary directory with a medgemma.json config file."""
    configs = tmp_path / "configs"
    configs.mkdir()
    config_file = configs / "medgemma.json"
    config_file.write_text(json.dumps(sample_medgemma_config))
    return configs


@pytest.fixture
def mock_registry(configs_dir: Path) -> object:
    """ModelRegistry with mocked GPU detection and no real S3."""
    with (
        patch("handlers.registry.torch") as mock_torch,
        patch("handlers.registry.boto3"),
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_mem=24 * (1024**3)  # 24 GB
        )
        mock_torch.cuda.OutOfMemoryError = RuntimeError
        mock_torch.cuda.empty_cache = MagicMock()

        # Need to set WEIGHTS_CACHE_DIR to a temp directory
        weights_dir = configs_dir.parent / "weights"
        weights_dir.mkdir()
        with patch.dict(os.environ, {"WEIGHTS_CACHE_DIR": str(weights_dir)}):
            from handlers.registry import ModelRegistry

            registry = ModelRegistry(configs_dir)

        return registry
