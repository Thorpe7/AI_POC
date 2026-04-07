"""SageMaker inference handlers for MedGemma medical vision-language model."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_NAME = "google/medgemma-4b"


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load MedGemma model and processor from the model directory.

    Args:
        model_dir: Path to the directory containing model artifacts.

    Returns:
        Dictionary with 'model' and 'processor' keys.
    """
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return {"model": model, "processor": processor}


def input_fn(request_body: str | bytes, request_content_type: str) -> dict[str, Any]:
    """Parse incoming request body into a structured input dictionary.

    Expects JSON with a required "text" field and an optional "image" field
    containing a base64-encoded PNG or JPEG image.

    Args:
        request_body: Raw request body as string or bytes.
        request_content_type: MIME type of the request.

    Returns:
        Dictionary with 'text' (str) and optionally 'image' (PIL.Image.Image).

    Raises:
        ValueError: If content type is unsupported, text is missing, or image is invalid.
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)

    if "text" not in payload:
        raise ValueError("Request must include a 'text' field")

    result: dict[str, Any] = {"text": payload["text"]}

    if "image" in payload:
        image_bytes = base64.b64decode(payload["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result["image"] = image

    return result


def predict_fn(input_data: dict[str, Any], model_dict: dict[str, Any]) -> str:
    """Run inference on the parsed input using the loaded model.

    Builds chat-format messages, applies the processor chat template,
    and generates output tokens.

    Args:
        input_data: Dictionary from input_fn with 'text' and optional 'image'.
        model_dict: Dictionary from model_fn with 'model' and 'processor'.

    Returns:
        Generated text string from the model.
    """
    model = model_dict["model"]
    processor = model_dict["processor"]

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

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    input_len = inputs["input_ids"].shape[-1]
    generated_text: str = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return generated_text


def output_fn(prediction: str, accept: str) -> str:
    """Serialize the prediction into the requested response format.

    Args:
        prediction: Generated text from predict_fn.
        accept: Accepted MIME type for the response.

    Returns:
        JSON string with the generated text.

    Raises:
        ValueError: If accept type is not supported.
    """
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")

    return json.dumps({"generated_text": prediction})
