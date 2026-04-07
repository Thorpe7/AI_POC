"""Tests for the HFHandler config-driven handler."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from handlers.hf_handler import HFHandler


class TestHFHandlerInit:
    """Tests for handler construction and config validation."""

    def test_valid_config(self, sample_medgemma_config: dict[str, Any]) -> None:
        """Handler accepts a complete config."""
        handler = HFHandler(sample_medgemma_config)
        assert handler.config == sample_medgemma_config

    def test_missing_fields_raises(self) -> None:
        """Handler raises ValueError on incomplete config."""
        with pytest.raises(ValueError, match="missing required fields"):
            HFHandler({"auto_class": "AutoModel"})


class TestHFHandlerModelFn:
    """Tests for model_fn with mocked transformers."""

    @patch("handlers.hf_handler.transformers")
    def test_dynamic_class_resolution(
        self,
        mock_transformers: MagicMock,
        sample_medgemma_config: dict[str, Any],
    ) -> None:
        """model_fn uses getattr to resolve auto_class and processor_class."""
        mock_model_cls = MagicMock()
        mock_proc_cls = MagicMock()
        mock_transformers.AutoModelForImageTextToText = mock_model_cls
        mock_transformers.AutoProcessor = mock_proc_cls

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        handler = HFHandler(sample_medgemma_config)
        result = handler.model_fn("/tmp/weights")

        mock_model_cls.from_pretrained.assert_called_once()
        mock_proc_cls.from_pretrained.assert_called_once_with("/tmp/weights")
        assert "model" in result
        assert "processor" in result
        mock_model.eval.assert_called_once()

    @patch("handlers.hf_handler.transformers")
    @patch("handlers.hf_handler.torch")
    def test_configured_dtype(
        self,
        mock_torch: MagicMock,
        mock_transformers: MagicMock,
        sample_medgemma_config: dict[str, Any],
    ) -> None:
        """model_fn applies the dtype from config."""
        mock_torch.bfloat16 = "bf16_sentinel"
        mock_torch.float16 = "f16_sentinel"
        mock_torch.float32 = "f32_sentinel"

        mock_cls = MagicMock()
        mock_transformers.AutoModelForImageTextToText = mock_cls
        mock_transformers.AutoProcessor = MagicMock()

        handler = HFHandler(sample_medgemma_config)
        handler.model_fn("/tmp/weights")

        call_kwargs = mock_cls.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == "bf16_sentinel"


class TestHFHandlerPredictFn:
    """Tests for predict_fn."""

    def test_chat_template_path(
        self, sample_medgemma_config: dict[str, Any], mock_model_dict: dict[str, Any]
    ) -> None:
        """predict_fn uses chat template when use_chat_template is True."""
        handler = HFHandler(sample_medgemma_config)
        input_data = {"text": "What is this?"}

        handler.predict_fn(input_data, mock_model_dict)

        mock_model_dict["processor"].apply_chat_template.assert_called_once()

    def test_direct_tokenization_path(
        self, sample_medgemma_config: dict[str, Any], mock_model_dict: dict[str, Any]
    ) -> None:
        """predict_fn uses direct processor call when use_chat_template is False."""
        config = {**sample_medgemma_config, "use_chat_template": False}
        handler = HFHandler(config)

        mock_processor = mock_model_dict["processor"]
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(
            side_effect=lambda key: MagicMock(shape=(1, 10))
            if key == "input_ids"
            else MagicMock()
        )
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        handler.predict_fn({"text": "What is this?"}, mock_model_dict)

        mock_processor.assert_called_once()
        assert not mock_processor.apply_chat_template.called

    def test_generation_params_passed(
        self, sample_medgemma_config: dict[str, Any], mock_model_dict: dict[str, Any]
    ) -> None:
        """predict_fn passes generation_params to model.generate."""
        handler = HFHandler(sample_medgemma_config)
        handler.predict_fn({"text": "Test"}, mock_model_dict)

        generate_kwargs = mock_model_dict["model"].generate.call_args[1]
        assert generate_kwargs["max_new_tokens"] == 512
        assert generate_kwargs["do_sample"] is False

    def test_pre_built_messages_path(
        self, sample_medgemma_config: dict[str, Any], mock_model_dict: dict[str, Any]
    ) -> None:
        """predict_fn uses input_data['messages'] directly when present."""
        handler = HFHandler(sample_medgemma_config)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": [{"type": "text", "text": "Follow-up"}]},
        ]

        handler.predict_fn({"messages": messages}, mock_model_dict)

        mock_model_dict["processor"].apply_chat_template.assert_called_once()
        call_args = mock_model_dict["processor"].apply_chat_template.call_args
        assert call_args[0][0] == messages


class TestHFHandlerOutputFn:
    """Tests for output_fn."""

    def test_custom_output_key(self, sample_medgemma_config: dict[str, Any]) -> None:
        """output_fn uses the configured output_key."""
        handler = HFHandler(sample_medgemma_config)
        result = json.loads(handler.output_fn("test output", "application/json"))
        assert result == {"generated_text": "test output"}

    def test_non_default_output_key(self, sample_medgemma_config: dict[str, Any]) -> None:
        """output_fn uses a custom output_key when configured."""
        config = {**sample_medgemma_config, "output_key": "result"}
        handler = HFHandler(config)
        result = json.loads(handler.output_fn("test output", "application/json"))
        assert result == {"result": "test output"}

    def test_unsupported_accept_type(self, sample_medgemma_config: dict[str, Any]) -> None:
        """output_fn raises ValueError for unsupported accept types."""
        handler = HFHandler(sample_medgemma_config)
        with pytest.raises(ValueError, match="Unsupported accept type"):
            handler.output_fn("test", "text/xml")
