"""EmbarkLab's handler class for vLLM-supported models."""

import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image
from vllm import LLM, SamplingParams

from src.handlers.utils.dicom_preprocessing import preprocess_dicom_series

log = logging.getLogger(__name__)

ALLOWED_TEXT_ROOT = Path("/data/xnat/archive")
MAX_TEXT_CHARS_PER_FILE = 50_000
# MedGemma emits <unused94>...<unused95> thinking blocks; strip them (including
# an unclosed block when generation hits max_tokens mid-thought).
THINKING_BLOCK_RE = re.compile(r"<unused94>.*?(<unused95>|$)", re.DOTALL)


class VLLMHandler:
    """Custom model handler using vLLM for inference."""

    def __init__(self, config: dict[str, Any]):
        self.model_name = config["model_name"]
        self.model_dir = Path(config["model_dir"])
        self.dtype = config.get("dtype", "bfloat16")
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
                "slice_range": [start, end],  # optional, 0-indexed inclusive
                "image_paths": ["/data/path/to/img.png"],  # for 2D images
                "text_paths": ["/data/path/to/notes.txt"],  # optional text docs
                "max_new_tokens": 512  # optional
            }
        """
        prompt = request["prompt"]
        max_new_tokens = request.get("max_new_tokens", 512)
        slice_info = None
        used_indices: list[int] = []

        if "dicom_dir" in request:
            slice_range = request.get("slice_range")
            if slice_range is not None:
                if not (isinstance(slice_range, (list, tuple)) and len(slice_range) == 2):
                    raise ValueError("slice_range must be [start, end]")
                slice_range = (int(slice_range[0]), int(slice_range[1]))
            images, total, used_indices = preprocess_dicom_series(
                Path(request["dicom_dir"]), slice_range=slice_range
            )
            slice_info = {
                "total_slices": total,
                "used_slices": len(used_indices),
                "used_indices": used_indices,
            }
        else:
            if "slice_range" in request:
                raise ValueError("slice_range requires dicom_dir")
            images = []
            for path in request.get("image_paths", []):
                images.append(Image.open(path).convert("RGB"))

        text_docs: list[str] = []
        for path_str in request.get("text_paths", []):
            path = Path(path_str).resolve()
            if not path.is_relative_to(ALLOWED_TEXT_ROOT):
                raise ValueError(f"text_paths entry outside {ALLOWED_TEXT_ROOT}: {path}")
            if path.suffix.lower() != ".txt":
                raise ValueError(f"text_paths only supports .txt files: {path}")
            body = path.read_text(encoding="utf-8")
            if len(body) > MAX_TEXT_CHARS_PER_FILE:
                raise ValueError(
                    f"text file exceeds {MAX_TEXT_CHARS_PER_FILE} chars ({len(body)}): {path}"
                )
            text_docs.append(body)

        # Build chat messages: images -> text docs -> prompt
        slice_labels = used_indices if used_indices else list(range(len(images)))
        content: list[dict[str, Any]] = []
        for i, img in enumerate(images):
            content.append({"type": "image_url", "image_url": {"url": img}})
            content.append({"type": "text", "text": f"SLICE {slice_labels[i]}"})
        for i, body in enumerate(text_docs):
            content.append({"type": "text", "text": f"DOCUMENT {i}:\n{body}"})
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

        raw = outputs[0].outputs[0].text
        return THINKING_BLOCK_RE.sub("", raw).lstrip()

    def output_fn(self, prediction: str, slice_info: dict | None = None) -> dict[str, Any]:
        """Format the response, including slice sampling info if applicable."""
        result: dict[str, Any] = {"generated_text": prediction}
        if slice_info is not None:
            result["slice_info"] = slice_info
        return result
