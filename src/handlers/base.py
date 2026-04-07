"""Abstract base class for all model handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseHandler(ABC):
    """Base interface that every model handler must implement.

    Subclasses provide model-specific logic for loading, preprocessing,
    inference, and output formatting.
    """

    @abstractmethod
    def model_fn(self, model_dir: str) -> dict[str, Any]:
        """Load model artifacts from a local directory.

        Args:
            model_dir: Path to the directory containing model weights.

        Returns:
            Dictionary of loaded model components (e.g. model, processor).
        """

    @abstractmethod
    def input_fn(self, request_body: str | bytes, request_content_type: str) -> dict[str, Any]:
        """Parse raw request into a structured input dictionary.

        Args:
            request_body: Raw request payload.
            request_content_type: MIME type of the request.

        Returns:
            Parsed input dictionary ready for predict_fn.
        """

    @abstractmethod
    def predict_fn(self, input_data: dict[str, Any], model_dict: dict[str, Any]) -> Any:  # noqa: ANN401
        """Run inference on parsed input using loaded model components.

        Args:
            input_data: Parsed input from input_fn.
            model_dict: Loaded model components from model_fn.

        Returns:
            Raw prediction result.
        """

    @abstractmethod
    def output_fn(self, prediction: Any, accept: str) -> str:  # noqa: ANN401
        """Serialize prediction into the requested response format.

        Args:
            prediction: Raw prediction from predict_fn.
            accept: Accepted MIME type for the response.

        Returns:
            Serialized response string.
        """
