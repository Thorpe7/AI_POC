"""Custom KServe model class that wraps Embark's custom model handlers."""

import kserve

from typing import Any


def _load_handler(handler_type: str) -> type:
    """Lazy-import the handler class so each container only pulls in its own deps."""
    if handler_type == "vllm":
        from src.handlers.vllm_handler import VLLMHandler
        return VLLMHandler
    if handler_type == "totalseg":
        from src.handlers.totalseg_handler import TotalSegmentatorHandler
        return TotalSegmentatorHandler
    raise ValueError(
        f"Unknown handler_type: {handler_type}. Available: vllm, totalseg"
    )


class KserveModelHandler(kserve.Model):
    """Custom model handler for KServe that routes to the appropriate handler."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, return_response_headers=True)
        handler_cls = _load_handler(config["handler_type"])
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
