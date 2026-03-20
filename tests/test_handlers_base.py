"""Tests for the BaseHandler abstract base class."""

from __future__ import annotations

from typing import Any

import pytest

from handlers.base import BaseHandler


class TestBaseHandlerABC:
    """Verify ABC enforcement."""

    def test_cannot_instantiate_abc(self) -> None:
        """BaseHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseHandler()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        """A complete concrete subclass can be instantiated."""

        class ConcreteHandler(BaseHandler):
            def model_fn(self, model_dir: str) -> dict[str, Any]:
                return {}

            def input_fn(  # noqa: ANN401
                self, request_body: str | bytes, request_content_type: str
            ) -> dict[str, Any]:
                return {}

            def predict_fn(  # noqa: ANN401
                self, input_data: dict[str, Any], model_dict: dict[str, Any]
            ) -> object:
                return None

            def output_fn(
                self, prediction: object, accept: str
            ) -> str:
                return ""

        handler = ConcreteHandler()
        assert isinstance(handler, BaseHandler)

    def test_incomplete_subclass_fails(self) -> None:
        """A subclass missing abstract methods cannot be instantiated."""

        class PartialHandler(BaseHandler):
            def model_fn(self, model_dir: str) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError):
            PartialHandler()  # type: ignore[abstract]
