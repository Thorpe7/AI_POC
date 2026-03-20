"""Multi-model SageMaker inference shim.

Bridges SageMaker's 4-function contract (model_fn, input_fn, predict_fn, output_fn)
to the ModelRegistry, which handles on-demand model loading with LRU eviction.

The ``model`` field in each request selects which model config to use.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ensure code/ directory is importable (SageMaker extracts tarball to model_dir)
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from handlers.registry import ModelRegistry  # noqa: E402
from handlers.utils import decode_base64_image  # noqa: E402


def model_fn(model_dir: str) -> ModelRegistry:
    """Initialize the model registry from config files in the tarball.

    SageMaker calls this once at container startup. The configs directory
    lives at ``model_dir/configs/`` (tarball root, outside ``code/``).

    Args:
        model_dir: Path to the extracted model.tar.gz root.

    Returns:
        Initialized ModelRegistry instance.
    """
    configs_dir = Path(model_dir) / "configs"
    logger.info("Initializing registry from %s", configs_dir)
    return ModelRegistry(configs_dir)


def input_fn(request_body: str | bytes, request_content_type: str) -> dict[str, Any]:
    """Parse incoming request and extract model routing information.

    Expects JSON with a required ``model`` field to select the model config,
    a required ``text`` field, and an optional ``image`` field (base64-encoded).

    The model ID is stored as ``_model_id`` in the returned dict for
    ``predict_fn`` to pop and use for routing.

    Args:
        request_body: Raw request body.
        request_content_type: MIME type of the request.

    Returns:
        Parsed input dict with ``_model_id``, ``text``, and optional ``image``.

    Raises:
        ValueError: If content type is unsupported or required fields are missing.
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)

    if "model" not in payload:
        raise ValueError("Request must include a 'model' field")
    if "text" not in payload:
        raise ValueError("Request must include a 'text' field")

    result: dict[str, Any] = {
        "_model_id": payload["model"],
        "text": payload["text"],
    }

    if "image" in payload:
        result["image"] = decode_base64_image(payload["image"])

    return result


def predict_fn(input_data: dict[str, Any], registry: Any) -> tuple[str, str]:  # noqa: ANN401
    """Route to the correct model handler via the registry.

    Pops ``_model_id`` from input_data, calls ``registry.get_or_load()`` to
    obtain the handler and loaded model, then delegates to ``handler.predict_fn()``.

    Args:
        input_data: Parsed input from input_fn (with ``_model_id``).
        registry: ModelRegistry instance returned by model_fn.

    Returns:
        Tuple of (prediction_text, model_id) so output_fn can look up
        the correct config for output formatting.
    """
    model_id = input_data.pop("_model_id")

    if not registry.is_known_model(model_id):
        raise ValueError(
            f"Unknown model: '{model_id}'. "
            f"Available: {[m['model_id'] for m in registry.get_available_models()]}"
        )

    handler, model_dict = registry.get_or_load(model_id)
    prediction = handler.predict_fn(input_data, model_dict)

    return prediction, model_id


def output_fn(prediction: tuple[str, str], accept: str) -> str:
    """Serialize the prediction using the per-model output_key.

    Note: SageMaker passes output of ``predict_fn`` directly here. We don't
    have access to the registry, so we use a default output key. The model-specific
    output_key from config is used when the registry is available via the handler.

    For the shim, we use a simple default since ``predict_fn`` returns a
    ``(prediction_text, model_id)`` tuple.

    Args:
        prediction: Tuple of (generated_text, model_id) from predict_fn.
        accept: Accepted MIME type for the response.

    Returns:
        JSON-encoded response string.
    """
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")

    generated_text, _model_id = prediction
    return json.dumps({"generated_text": generated_text})
