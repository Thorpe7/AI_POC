"""Multi-model SageMaker inference shim with chat session support.

Bridges SageMaker's 4-function contract (model_fn, input_fn, predict_fn, output_fn)
to the ModelRegistry, which handles on-demand model loading with LRU eviction.

The ``model`` field in each request selects which model config to use.
Models with ``"supports_chat": true`` in their config get multi-turn session
management; all others are zero-shot.
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

from handlers.chat_sessions import ChatSessionManager  # noqa: E402
from handlers.job_manager import JobManager  # noqa: E402
from handlers.registry import ModelRegistry  # noqa: E402
from handlers.utils import decode_base64_image  # noqa: E402


def model_fn(model_dir: str) -> dict[str, Any]:
    """Initialize the model registry and session manager.

    SageMaker calls this once at container startup. The configs directory
    lives at ``model_dir/configs/`` (tarball root, outside ``code/``).

    Args:
        model_dir: Path to the extracted model.tar.gz root.

    Returns:
        Context dict with ``registry`` and ``session_manager``.
    """
    configs_dir = Path(model_dir) / "configs"
    logger.info("Initializing registry from %s", configs_dir)
    return {
        "registry": ModelRegistry(configs_dir),
        "session_manager": ChatSessionManager(),
        "job_manager": JobManager(),
    }


def input_fn(request_body: str | bytes, request_content_type: str) -> dict[str, Any]:
    """Parse incoming request and extract model routing information.

    Expects JSON with a required ``model`` field to select the model config,
    a required ``text`` field (unless ``clear_session`` is true), and optional
    ``image`` (base64), ``session_id`` (str), and ``clear_session`` (bool) fields.

    Args:
        request_body: Raw request body.
        request_content_type: MIME type of the request.

    Returns:
        Parsed input dict with ``_model_id``, ``text``, and optional fields.

    Raises:
        ValueError: If content type is unsupported or required fields are missing.
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)

    if "model" not in payload:
        raise ValueError("Request must include a 'model' field")

    clear_session = payload.get("clear_session", False)
    if (
        not clear_session
        and "poll_job_id" not in payload
        and "text" not in payload
        and "volume_s3_uri" not in payload
    ):
        raise ValueError("Request must include 'text' or 'volume_s3_uri'")

    result: dict[str, Any] = {"_model_id": payload["model"]}

    if "text" in payload:
        result["text"] = payload["text"]

    if "session_id" in payload:
        result["_session_id"] = payload["session_id"]

    if clear_session:
        result["_clear_session"] = True

    if "image" in payload:
        result["image"] = decode_base64_image(payload["image"])

    # Pass through Merlin-specific fields
    if "mode" in payload:
        result["mode"] = payload["mode"]
    if "volume_s3_uri" in payload:
        result["volume_s3_uri"] = payload["volume_s3_uri"]
    if "query_texts" in payload:
        result["query_texts"] = payload["query_texts"]

    # Background job fields
    if "background" in payload:
        result["_background"] = payload["background"]
    if "poll_job_id" in payload:
        result["_poll_job_id"] = payload["poll_job_id"]

    return result


def predict_fn(input_data: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ANN401
    """Route to the correct model handler, managing chat sessions when applicable.

    For models with ``supports_chat`` in config: creates/retrieves sessions,
    builds conversation history, and returns response with ``session_id``
    and ``turn`` count.

    For zero-shot models: delegates directly to the handler.

    Args:
        input_data: Parsed input from input_fn (with ``_model_id``).
        context: Dict with ``registry`` and ``session_manager`` from model_fn.

    Returns:
        Dict with ``generated_text`` and optionally ``session_id``/``turn``.
    """
    registry: ModelRegistry = context["registry"]
    session_manager: ChatSessionManager = context["session_manager"]
    job_manager: JobManager = context["job_manager"]

    # 1. Poll — no model loading needed
    poll_job_id: str | None = input_data.pop("_poll_job_id", None)
    if poll_job_id:
        input_data.pop("_model_id", None)
        return job_manager.poll(poll_job_id)

    model_id = input_data.pop("_model_id")
    session_id: str | None = input_data.pop("_session_id", None)
    clear_session: bool = input_data.pop("_clear_session", False)

    # Handle clear_session — no inference needed
    if clear_session:
        if session_id:
            session_manager.delete_session(session_id)
        return {
            "generated_text": "Session cleared.",
            "session_id": session_id or "",
        }

    if not registry.is_known_model(model_id):
        raise ValueError(
            f"Unknown model: '{model_id}'. "
            f"Available: {[m['model_id'] for m in registry.get_available_models()]}"
        )

    config = registry.get_config(model_id)

    # 2. Background submit — pre-load model, then run in thread
    background: bool = input_data.pop("_background", False)
    if background:
        handler, model_dict = registry.get_or_load(model_id)
        job_id = job_manager.submit(handler.predict_fn, input_data, model_dict)
        return {"job_id": job_id, "status": "submitted"}

    supports_chat = config.get("supports_chat", False)

    if supports_chat:
        max_turns = config.get("max_chat_turns", 20)

        # Get existing session or create a new one
        session = None
        if session_id:
            session = session_manager.get_session(session_id)

        if session is None:
            session_id = session_manager.create_session(max_turns=max_turns)
            session = session_manager.get_session(session_id)

        # Build user content and add to session
        content: list[dict[str, Any]] = []
        if "image" in input_data:
            content.append({"type": "image", "image": input_data["image"]})
        content.append({"type": "text", "text": input_data["text"]})

        session.add_user_message(content)  # type: ignore[union-attr]

        # Pass full history to handler
        handler, model_dict = registry.get_or_load(model_id)
        prediction = handler.predict_fn({"messages": session.messages}, model_dict)  # type: ignore[union-attr]

        session.add_assistant_message(prediction)  # type: ignore[union-attr]

        return {
            "generated_text": prediction,
            "session_id": session_id,
            "turn": session.turn_count,  # type: ignore[union-attr]
        }

    # Zero-shot path
    handler, model_dict = registry.get_or_load(model_id)
    prediction = handler.predict_fn(input_data, model_dict)

    # Handlers like MerlinHandler return a dict directly; wrap string results
    if isinstance(prediction, dict):
        return prediction
    return {"generated_text": prediction}


def output_fn(prediction: dict[str, Any], accept: str) -> str:  # noqa: ANN401
    """Serialize the prediction dict to JSON.

    The prediction dict always contains ``generated_text``, and for chat
    models also includes ``session_id`` and ``turn``.

    Args:
        prediction: Dict from predict_fn.
        accept: Accepted MIME type for the response.

    Returns:
        JSON-encoded response string.
    """
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")

    return json.dumps(prediction)
