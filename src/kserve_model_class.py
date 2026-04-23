"""Custom KServe model class that wraps Embark's custom model handlers.
"""

import kserve

from typing import Any

from src.handlers.vllm_handler import VLLMHandler

HANDLER_REGISTRY: dict[str, type] = {
    "vllm": VLLMHandler,
}


class KserveModelHandler(kserve.Model):
    """Custom model handler for KServe that routes to the appropriate handler."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, return_response_headers=True)
        handler_type = config["handler_type"]
        if handler_type not in HANDLER_REGISTRY:
            raise ValueError(
                f"Unknown handler_type: {handler_type}. Available: {list(HANDLER_REGISTRY.keys())}"
            )
        handler_cls = HANDLER_REGISTRY[handler_type]
        self.handler = handler_cls(config=config)
        self.handler.model_fn()
        self.ready = True

    def preprocess(  # type: ignore[override]
        self, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        return self.handler.input_fn(payload)

    def predict(  # type: ignore[override]
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        result = self.handler.predict_fn(payload)
        return self.handler.output_fn(result, slice_info=payload.get("slice_info"))

    def postprocess(  # type: ignore[override]
        self,
        outputs: dict[str, str],
        headers: dict[str, str] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        return outputs