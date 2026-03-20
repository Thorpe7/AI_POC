"""Config-driven HuggingFace Transformers handler."""

from __future__ import annotations

import logging
from typing import Any

import torch
import transformers

from handlers.base import BaseHandler
from handlers.utils import decode_base64_image

logger = logging.getLogger(__name__)

REQUIRED_CONFIG_KEYS = {
    "auto_class",
    "processor_class",
    "dtype",
    "device_map",
    "input_modality",
    "generation_params",
    "output_key",
    "estimated_gpu_memory_gb",
}


class HFHandler(BaseHandler):
    """Handler for any HuggingFace Transformers model, driven by a JSON config."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Check that required config fields are present."""
        missing = REQUIRED_CONFIG_KEYS - set(self._config.keys())
        if missing:
            raise ValueError(f"Config missing required fields: {missing}")

    @property
    def config(self) -> dict[str, Any]:
        """Return the handler's config dictionary."""
        return self._config

    def model_fn(self, model_dir: str) -> dict[str, Any]:
        """Load model and processor from a local weights directory.

        Uses dynamic class resolution via ``getattr(transformers, config["auto_class"])``.

        Args:
            model_dir: Path to directory containing downloaded model weights.

        Returns:
            Dictionary with 'model' and 'processor' keys.
        """
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._config["dtype"], torch.bfloat16)

        auto_cls = getattr(transformers, self._config["auto_class"])
        processor_cls = getattr(transformers, self._config["processor_class"])

        processor = processor_cls.from_pretrained(model_dir)
        model = auto_cls.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map=self._config["device_map"],
        )
        model.eval()

        logger.info("Loaded %s from %s (dtype=%s)", self._config["auto_class"], model_dir, dtype)
        return {"model": model, "processor": processor}

    def input_fn(self, request_body: str | bytes, request_content_type: str) -> dict[str, Any]:
        """Parse request body — not used directly by the shim, kept for interface compliance."""
        import json

        if request_content_type != "application/json":
            raise ValueError(f"Unsupported content type: {request_content_type}")

        payload = json.loads(request_body)
        if "text" not in payload:
            raise ValueError("Request must include a 'text' field")

        result: dict[str, Any] = {"text": payload["text"]}
        if "image" in payload:
            result["image"] = decode_base64_image(payload["image"])
        return result

    def predict_fn(self, input_data: dict[str, Any], model_dict: dict[str, Any]) -> str:
        """Run inference using the loaded model.

        Builds chat-format messages if ``use_chat_template`` is set, otherwise
        passes text directly to the processor.

        Args:
            input_data: Dictionary with 'text' and optional 'image'.
            model_dict: Dictionary with 'model' and 'processor'.

        Returns:
            Generated text string.
        """
        model = model_dict["model"]
        processor = model_dict["processor"]

        if self._config.get("use_chat_template", False):
            content: list[dict[str, Any]] = []
            if "image" in input_data:
                content.append({"type": "image", "image": input_data["image"]})
            content.append({"type": "text", "text": input_data["text"]})
            messages = [{"role": "user", "content": content}]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)
        else:
            inputs = processor(
                text=input_data["text"],
                images=input_data.get("image"),
                return_tensors="pt",
            ).to(model.device)

        gen_params = self._config.get("generation_params", {})
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_params)

        input_len = inputs["input_ids"].shape[-1]
        generated_text: str = processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        return generated_text

    def output_fn(self, prediction: Any, accept: str) -> str:  # noqa: ANN401
        """Serialize prediction to JSON using the configured output_key."""
        import json

        if accept not in ("application/json", "*/*"):
            raise ValueError(f"Unsupported accept type: {accept}")

        output_key = self._config.get("output_key", "generated_text")
        return json.dumps({output_key: prediction})
