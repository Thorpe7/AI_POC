"""EmbarkLab's handler class for vLLM-supported models."""

import logging
from pathlib import Path
from typing import Any

from PIL import Image
from vllm import LLM, SamplingParams

from src.handlers.utils.dicom_preprocessing import preprocess_dicom_series

log = logging.getLogger(__name__)


class VLLMHandler:
    """Custom model handler using vLLM for inference."""

    def __init__(self, config: dict[str, Any]):
        self.model_name = config["model_name"]
        self.model_dir = Path(config["model_dir"])
        self.dtype = config.get("dtype", "float16")
        self.max_model_len = config.get("max_model_len", 32768)
        self.model: LLM | None = None

    def model_fn(self) -> None:
        """Load model via vLLM from mounted directory."""
        log.info(
            "loading vLLM model: dir=%s dtype=%s max_model_len=%d",
            self.model_dir,
            self.dtype,
            self.max_model_len,
        )
        self.model = LLM(
            model=str(self.model_dir),
            dtype=self.dtype,
            max_model_len=self.max_model_len,
        )
        log.info("vLLM model loaded")

    def input_fn(self, request: dict[str, Any]) -> dict[str, Any]:
        """Parse request and preprocess DICOM series or 2D images.

        Expected request format:
            {
                "prompt": "Describe any findings.",
                "dicom_dir": "/data/path/to/series/",  # for 3D volumes
                "image_paths": ["/data/path/to/img.png"],  # for 2D images
                "max_new_tokens": 512  # optional
            }
        """
        prompt = request["prompt"]
        max_new_tokens = request.get("max_new_tokens", 512)
        slice_info = None

        if "dicom_dir" in request:
            images, total, used = preprocess_dicom_series(Path(request["dicom_dir"]))
            slice_info = {"total_slices": total, "used_slices": used}
        else:
            images = []
            for path in request.get("image_paths", []):
                images.append(Image.open(path).convert("RGB"))

        # Build chat messages with interleaved images
        content: list[dict[str, Any]] = []
        for i, img in enumerate(images):
            content.append({"type": "image_url", "image_url": {"url": img}})
            content.append({"type": "text", "text": f"SLICE {i}"})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        return {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "slice_info": slice_info,
        }

    def predict_fn(self, input_data: dict[str, Any]) -> str:
        """Run inference via vLLM chat."""
        if self.model is None:
            raise RuntimeError("model_fn() must be called before predict_fn()")

        params = SamplingParams(max_tokens=input_data["max_new_tokens"])
        outputs = self.model.chat(input_data["messages"], sampling_params=params)

        return outputs[0].outputs[0].text

    def output_fn(self, prediction: str, slice_info: dict | None = None) -> dict[str, Any]:
        """Format the response, including slice sampling info if applicable."""
        result: dict[str, Any] = {"generated_text": prediction}
        if slice_info is not None:
            result["slice_info"] = slice_info
        return result
