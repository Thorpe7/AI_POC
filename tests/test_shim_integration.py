"""Integration tests for the multi-model SageMaker shim."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shims.hf_inference import input_fn, model_fn, output_fn, predict_fn


class TestModelFn:
    """Tests for shim model_fn."""

    def test_creates_registry(self, configs_dir: Path) -> None:
        """model_fn returns a ModelRegistry initialized from configs."""
        with (
            patch("handlers.registry.torch") as mock_torch,
            patch("handlers.registry.boto3"),
            patch.dict("os.environ", {"WEIGHTS_CACHE_DIR": str(configs_dir.parent / "w")}),
        ):
            mock_torch.cuda.is_available.return_value = False
            (configs_dir.parent / "w").mkdir(exist_ok=True)

            # model_fn expects configs/ at model_dir root
            model_dir = configs_dir.parent
            registry = model_fn(str(model_dir))

        from handlers.registry import ModelRegistry

        assert isinstance(registry, ModelRegistry)
        assert registry.is_known_model("medgemma")


class TestInputFn:
    """Tests for shim input_fn."""

    def test_valid_request_with_model(self, sample_image_base64: str) -> None:
        """Parses request with model, text, and image fields."""
        body = json.dumps({
            "model": "medgemma",
            "text": "Describe this.",
            "image": sample_image_base64,
        })
        result = input_fn(body, "application/json")
        assert result["_model_id"] == "medgemma"
        assert result["text"] == "Describe this."
        assert "image" in result

    def test_missing_model_field(self) -> None:
        """Raises ValueError when model field is missing."""
        body = json.dumps({"text": "Hello"})
        with pytest.raises(ValueError, match="model"):
            input_fn(body, "application/json")

    def test_missing_text_field(self) -> None:
        """Raises ValueError when text field is missing."""
        body = json.dumps({"model": "medgemma"})
        with pytest.raises(ValueError, match="text"):
            input_fn(body, "application/json")

    def test_unsupported_content_type(self) -> None:
        """Raises ValueError for non-JSON content type."""
        with pytest.raises(ValueError, match="Unsupported content type"):
            input_fn("{}", "text/plain")


class TestPredictFn:
    """Tests for shim predict_fn."""

    def test_routes_to_correct_handler(self) -> None:
        """predict_fn calls registry.get_or_load with the correct model_id."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = "Generated text"
        mock_model_dict = {"model": MagicMock()}
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, mock_model_dict)

        input_data = {"_model_id": "medgemma", "text": "Hello"}
        prediction, model_id = predict_fn(input_data, mock_registry)

        mock_registry.get_or_load.assert_called_once_with("medgemma")
        assert prediction == "Generated text"
        assert model_id == "medgemma"

    def test_unknown_model_raises(self) -> None:
        """predict_fn raises ValueError for unknown model."""
        mock_registry = MagicMock()
        mock_registry.is_known_model.return_value = False
        mock_registry.get_available_models.return_value = [{"model_id": "medgemma"}]

        input_data = {"_model_id": "unknown", "text": "Hello"}
        with pytest.raises(ValueError, match="Unknown model"):
            predict_fn(input_data, mock_registry)

    def test_pops_model_id_from_input(self) -> None:
        """predict_fn removes _model_id before passing to handler."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = "ok"
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {})

        input_data = {"_model_id": "medgemma", "text": "Hello"}
        predict_fn(input_data, mock_registry)

        # handler.predict_fn should receive input WITHOUT _model_id
        handler_input = mock_handler.predict_fn.call_args[0][0]
        assert "_model_id" not in handler_input


class TestOutputFn:
    """Tests for shim output_fn."""

    def test_serializes_prediction(self) -> None:
        """output_fn returns JSON with generated_text."""
        result = output_fn(("Hello world", "medgemma"), "application/json")
        parsed = json.loads(result)
        assert parsed == {"generated_text": "Hello world"}

    def test_unsupported_accept_type(self) -> None:
        """output_fn raises ValueError for unsupported accept types."""
        with pytest.raises(ValueError, match="Unsupported accept type"):
            output_fn(("text", "model"), "text/xml")

    def test_wildcard_accept(self) -> None:
        """output_fn accepts wildcard MIME type."""
        result = output_fn(("Hello", "medgemma"), "*/*")
        parsed = json.loads(result)
        assert parsed["generated_text"] == "Hello"


class TestEndToEndFlow:
    """Full flow through shim functions with mocked registry."""

    def test_full_flow(self, sample_image_base64: str) -> None:
        """model_fn → input_fn → predict_fn → output_fn."""
        # Setup mock registry
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = "Diagnosis: healthy"
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {"model": MagicMock()})

        # input_fn
        body = json.dumps({
            "model": "medgemma",
            "text": "Describe this image.",
            "image": sample_image_base64,
        })
        parsed = input_fn(body, "application/json")
        assert parsed["_model_id"] == "medgemma"

        # predict_fn
        prediction, model_id = predict_fn(parsed, mock_registry)
        assert prediction == "Diagnosis: healthy"
        assert model_id == "medgemma"

        # output_fn
        result = output_fn((prediction, model_id), "application/json")
        parsed_output = json.loads(result)
        assert parsed_output == {"generated_text": "Diagnosis: healthy"}
