"""Shared fixtures for inference handler tests."""

from __future__ import annotations

import base64
import io
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image


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
