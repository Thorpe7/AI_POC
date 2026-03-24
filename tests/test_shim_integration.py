"""Integration tests for the multi-model SageMaker shim."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from handlers.chat_sessions import ChatSessionManager
from handlers.job_manager import JobManager
from shims.hf_inference import input_fn, model_fn, output_fn, predict_fn


class TestModelFn:
    """Tests for shim model_fn."""

    def test_creates_registry_and_session_manager(self, configs_dir: Path) -> None:
        """model_fn returns a dict with registry and session_manager."""
        with (
            patch("handlers.registry.torch") as mock_torch,
            patch("handlers.registry.boto3"),
            patch.dict("os.environ", {"WEIGHTS_CACHE_DIR": str(configs_dir.parent / "w")}),
        ):
            mock_torch.cuda.is_available.return_value = False
            (configs_dir.parent / "w").mkdir(exist_ok=True)

            model_dir = configs_dir.parent
            context = model_fn(str(model_dir))

        from handlers.registry import ModelRegistry

        assert isinstance(context, dict)
        assert "registry" in context
        assert "session_manager" in context
        assert isinstance(context["registry"], ModelRegistry)
        assert isinstance(context["session_manager"], ChatSessionManager)
        assert isinstance(context["job_manager"], JobManager)
        assert context["registry"].is_known_model("medgemma")


class TestInputFn:
    """Tests for shim input_fn."""

    def test_valid_request_with_model(self, sample_image_base64: str) -> None:
        """Parses request with model, text, and image fields."""
        body = json.dumps({
            "model": "medgemma",
            "text": "Describe this.",
            "image": sample_image_base64,
        })
        result = input_fn(body, "application/json")
        assert result["_model_id"] == "medgemma"
        assert result["text"] == "Describe this."
        assert "image" in result

    def test_missing_model_field(self) -> None:
        """Raises ValueError when model field is missing."""
        body = json.dumps({"text": "Hello"})
        with pytest.raises(ValueError, match="model"):
            input_fn(body, "application/json")

    def test_missing_text_and_volume_field(self) -> None:
        """Raises ValueError when neither text nor volume_s3_uri is present."""
        body = json.dumps({"model": "medgemma"})
        with pytest.raises(ValueError, match="text.*volume_s3_uri"):
            input_fn(body, "application/json")

    def test_unsupported_content_type(self) -> None:
        """Raises ValueError for non-JSON content type."""
        with pytest.raises(ValueError, match="Unsupported content type"):
            input_fn("{}", "text/plain")

    def test_session_id_extracted(self) -> None:
        """session_id is stored as _session_id."""
        body = json.dumps({
            "model": "medgemma",
            "text": "Hello",
            "session_id": "abc-123",
        })
        result = input_fn(body, "application/json")
        assert result["_session_id"] == "abc-123"

    def test_clear_session_no_text_required(self) -> None:
        """clear_session requests don't require a text field."""
        body = json.dumps({
            "model": "medgemma",
            "session_id": "abc-123",
            "clear_session": True,
        })
        result = input_fn(body, "application/json")
        assert result["_clear_session"] is True
        assert "text" not in result


class TestPredictFn:
    """Tests for shim predict_fn."""

    def test_routes_to_correct_handler(self) -> None:
        """predict_fn calls registry.get_or_load with the correct model_id."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = "Generated text"
        mock_model_dict = {"model": MagicMock()}
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, mock_model_dict)
        mock_registry.get_config.return_value = {"supports_chat": False}

        context = {"registry": mock_registry, "session_manager": ChatSessionManager(), "job_manager": JobManager()}
        input_data = {"_model_id": "medgemma", "text": "Hello"}
        result = predict_fn(input_data, context)

        mock_registry.get_or_load.assert_called_once_with("medgemma")
        assert result["generated_text"] == "Generated text"

    def test_unknown_model_raises(self) -> None:
        """predict_fn raises ValueError for unknown model."""
        mock_registry = MagicMock()
        mock_registry.is_known_model.return_value = False
        mock_registry.get_available_models.return_value = [{"model_id": "medgemma"}]

        context = {"registry": mock_registry, "session_manager": ChatSessionManager(), "job_manager": JobManager()}
        input_data = {"_model_id": "unknown", "text": "Hello"}
        with pytest.raises(ValueError, match="Unknown model"):
            predict_fn(input_data, context)

    def test_pops_model_id_from_input(self) -> None:
        """predict_fn removes _model_id before passing to handler."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = "ok"
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {})
        mock_registry.get_config.return_value = {"supports_chat": False}

        context = {"registry": mock_registry, "session_manager": ChatSessionManager(), "job_manager": JobManager()}
        input_data = {"_model_id": "medgemma", "text": "Hello"}
        predict_fn(input_data, context)

        handler_input = mock_handler.predict_fn.call_args[0][0]
        assert "_model_id" not in handler_input


class TestOutputFn:
    """Tests for shim output_fn."""

    def test_serializes_prediction(self) -> None:
        """output_fn returns JSON with generated_text."""
        result = output_fn({"generated_text": "Hello world"}, "application/json")
        parsed = json.loads(result)
        assert parsed == {"generated_text": "Hello world"}

    def test_serializes_chat_prediction(self) -> None:
        """output_fn preserves session_id and turn in chat responses."""
        prediction = {
            "generated_text": "Hello",
            "session_id": "abc-123",
            "turn": 1,
        }
        result = output_fn(prediction, "application/json")
        parsed = json.loads(result)
        assert parsed == prediction

    def test_unsupported_accept_type(self) -> None:
        """output_fn raises ValueError for unsupported accept types."""
        with pytest.raises(ValueError, match="Unsupported accept type"):
            output_fn({"generated_text": "text"}, "text/xml")

    def test_wildcard_accept(self) -> None:
        """output_fn accepts wildcard MIME type."""
        result = output_fn({"generated_text": "Hello"}, "*/*")
        parsed = json.loads(result)
        assert parsed["generated_text"] == "Hello"


class TestChatFlow:
    """Tests for multi-turn chat session management in predict_fn."""

    @staticmethod
    def _make_context(
        supports_chat: bool = True,
        max_chat_turns: int = 20,
    ) -> dict[str, Any]:
        """Build a context dict with mocked registry."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = "Assistant response"
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {"model": MagicMock()})
        mock_registry.get_config.return_value = {
            "supports_chat": supports_chat,
            "max_chat_turns": max_chat_turns,
        }
        return {
            "registry": mock_registry,
            "session_manager": ChatSessionManager(),
            "job_manager": JobManager(),
        }

    def test_new_session_created(self) -> None:
        """First request to a chat model creates a session and returns session_id."""
        context = self._make_context()
        input_data = {"_model_id": "medgemma", "text": "Hello"}

        result = predict_fn(input_data, context)

        assert "session_id" in result
        assert result["turn"] == 1
        assert result["generated_text"] == "Assistant response"

    def test_continue_with_session_id(self) -> None:
        """Subsequent requests with session_id continue the same session."""
        context = self._make_context()

        # First turn
        result1 = predict_fn({"_model_id": "medgemma", "text": "Hello"}, context)
        session_id = result1["session_id"]

        # Second turn
        result2 = predict_fn(
            {"_model_id": "medgemma", "text": "Follow-up", "_session_id": session_id},
            context,
        )

        assert result2["session_id"] == session_id
        assert result2["turn"] == 2

    def test_multi_turn_history_passed_to_handler(self) -> None:
        """Handler receives full conversation history at time of call."""
        context = self._make_context()
        mock_handler = context["registry"].get_or_load.return_value[0]

        # Capture messages at call time (before assistant response is appended)
        captured: list[list[dict[str, Any]]] = []

        def capture_and_respond(input_data: dict[str, Any], model_dict: Any) -> str:
            captured.append([m.copy() for m in input_data["messages"]])
            return "Assistant response"

        mock_handler.predict_fn.side_effect = capture_and_respond

        # First turn
        result1 = predict_fn({"_model_id": "medgemma", "text": "Hello"}, context)
        session_id = result1["session_id"]

        # Second turn
        predict_fn(
            {"_model_id": "medgemma", "text": "Follow-up", "_session_id": session_id},
            context,
        )

        # Second call should have received: user1, assistant1, user2
        messages = captured[1]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_image_in_first_turn_only(self) -> None:
        """Image content is included in the first turn's message."""
        context = self._make_context()
        mock_handler = context["registry"].get_or_load.return_value[0]

        image_sentinel = MagicMock()
        result1 = predict_fn(
            {"_model_id": "medgemma", "text": "What's this?", "image": image_sentinel},
            context,
        )
        session_id = result1["session_id"]

        # Second turn without image
        predict_fn(
            {"_model_id": "medgemma", "text": "Tell me more", "_session_id": session_id},
            context,
        )

        second_call_input = mock_handler.predict_fn.call_args[0][0]
        messages = second_call_input["messages"]

        # First user message should have image + text
        first_content = messages[0]["content"]
        assert any(item.get("type") == "image" for item in first_content)

        # Third message (second user) should have text only
        third_content = messages[2]["content"]
        assert len(third_content) == 1
        assert third_content[0]["type"] == "text"

    def test_clear_session(self) -> None:
        """clear_session deletes the session and returns confirmation."""
        context = self._make_context()

        # Create a session
        result1 = predict_fn({"_model_id": "medgemma", "text": "Hello"}, context)
        session_id = result1["session_id"]

        # Clear it
        result2 = predict_fn(
            {"_model_id": "medgemma", "_session_id": session_id, "_clear_session": True},
            context,
        )

        assert result2["generated_text"] == "Session cleared."
        assert result2["session_id"] == session_id

        # Session is gone — next request creates a new one
        result3 = predict_fn(
            {"_model_id": "medgemma", "text": "Hello again", "_session_id": session_id},
            context,
        )
        assert result3["session_id"] != session_id
        assert result3["turn"] == 1

    def test_non_chat_model_ignores_session_id(self) -> None:
        """Zero-shot models return no session_id even if one was sent."""
        context = self._make_context(supports_chat=False)

        result = predict_fn(
            {"_model_id": "merlin", "text": "Analyze this", "_session_id": "ignored"},
            context,
        )

        assert "session_id" not in result
        assert "turn" not in result
        assert result["generated_text"] == "Assistant response"

    def test_expired_session_starts_fresh(self) -> None:
        """An expired session_id creates a new session transparently."""
        context = self._make_context()

        # Create a session
        result1 = predict_fn({"_model_id": "medgemma", "text": "Hello"}, context)
        old_session_id = result1["session_id"]

        # Expire it
        session_manager = context["session_manager"]
        session = session_manager._sessions[old_session_id]
        session.last_active = datetime.now(timezone.utc) - timedelta(hours=2)

        # Request with expired session_id
        result2 = predict_fn(
            {"_model_id": "medgemma", "text": "Hello again", "_session_id": old_session_id},
            context,
        )

        # Should get a new session
        assert result2["session_id"] != old_session_id
        assert result2["turn"] == 1


class TestEndToEndFlow:
    """Full flow through shim functions with mocked registry."""

    def test_full_zero_shot_flow(self, sample_image_base64: str) -> None:
        """model_fn → input_fn → predict_fn → output_fn for zero-shot models."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = "Diagnosis: healthy"
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {"model": MagicMock()})
        mock_registry.get_config.return_value = {"supports_chat": False}

        context = {"registry": mock_registry, "session_manager": ChatSessionManager(), "job_manager": JobManager()}

        # input_fn
        body = json.dumps({
            "model": "medgemma",
            "text": "Describe this image.",
            "image": sample_image_base64,
        })
        parsed = input_fn(body, "application/json")
        assert parsed["_model_id"] == "medgemma"

        # predict_fn
        result = predict_fn(parsed, context)
        assert result["generated_text"] == "Diagnosis: healthy"

        # output_fn
        output = output_fn(result, "application/json")
        parsed_output = json.loads(output)
        assert parsed_output == {"generated_text": "Diagnosis: healthy"}

    def test_full_chat_flow(self, sample_image_base64: str) -> None:
        """input_fn → predict_fn → output_fn for multi-turn chat."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = "I see a chest X-ray."
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {"model": MagicMock()})
        mock_registry.get_config.return_value = {
            "supports_chat": True,
            "max_chat_turns": 20,
        }

        context = {"registry": mock_registry, "session_manager": ChatSessionManager(), "job_manager": JobManager()}

        # Turn 1: image + text
        body1 = json.dumps({
            "model": "medgemma",
            "text": "What's in this scan?",
            "image": sample_image_base64,
        })
        parsed1 = input_fn(body1, "application/json")
        result1 = predict_fn(parsed1, context)

        assert result1["generated_text"] == "I see a chest X-ray."
        assert "session_id" in result1
        assert result1["turn"] == 1

        output1 = output_fn(result1, "application/json")
        parsed_output1 = json.loads(output1)
        assert parsed_output1["session_id"] == result1["session_id"]

        # Turn 2: text-only continuation
        mock_handler.predict_fn.return_value = "No signs of malignancy."
        body2 = json.dumps({
            "model": "medgemma",
            "text": "Is it malignant?",
            "session_id": result1["session_id"],
        })
        parsed2 = input_fn(body2, "application/json")
        result2 = predict_fn(parsed2, context)

        assert result2["generated_text"] == "No signs of malignancy."
        assert result2["session_id"] == result1["session_id"]
        assert result2["turn"] == 2


class TestMerlinShimFlow:
    """Tests for Merlin model routing through the shim."""

    def test_input_fn_with_volume_s3_uri(self) -> None:
        """input_fn accepts volume_s3_uri instead of text."""
        body = json.dumps({
            "model": "merlin",
            "mode": "findings",
            "volume_s3_uri": "s3://bucket/vol.nii.gz",
        })
        result = input_fn(body, "application/json")
        assert result["_model_id"] == "merlin"
        assert result["mode"] == "findings"
        assert result["volume_s3_uri"] == "s3://bucket/vol.nii.gz"

    def test_input_fn_passes_query_texts(self) -> None:
        """input_fn passes through query_texts for retrieval mode."""
        body = json.dumps({
            "model": "merlin",
            "mode": "retrieval",
            "volume_s3_uri": "s3://bucket/vol.nii.gz",
            "query_texts": ["ascites", "lesion"],
        })
        result = input_fn(body, "application/json")
        assert result["query_texts"] == ["ascites", "lesion"]

    def test_predict_fn_dict_passthrough(self) -> None:
        """predict_fn returns dict results directly (no generated_text wrapping)."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        merlin_result = {"mode": "findings", "findings": {"ascites": 0.87}}
        mock_handler.predict_fn.return_value = merlin_result
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {"model_dir": "/tmp"})
        mock_registry.get_config.return_value = {"supports_chat": False}

        context = {"registry": mock_registry, "session_manager": ChatSessionManager(), "job_manager": JobManager()}
        input_data = {
            "_model_id": "merlin",
            "mode": "findings",
            "volume_s3_uri": "s3://bucket/vol.nii.gz",
        }
        result = predict_fn(input_data, context)

        assert result == merlin_result
        assert "generated_text" not in result

    def test_output_fn_with_merlin_dict(self) -> None:
        """output_fn serializes Merlin dict responses."""
        prediction = {"mode": "findings", "findings": {"ascites": 0.87}}
        result = output_fn(prediction, "application/json")
        parsed = json.loads(result)
        assert parsed == prediction

    def test_full_merlin_zero_shot_flow(self) -> None:
        """Full input_fn → predict_fn → output_fn flow for Merlin."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = {
            "mode": "report",
            "report_text": "FINDINGS: Normal.",
        }
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {"model_dir": "/tmp"})
        mock_registry.get_config.return_value = {"supports_chat": False}

        context = {"registry": mock_registry, "session_manager": ChatSessionManager(), "job_manager": JobManager()}

        body = json.dumps({
            "model": "merlin",
            "mode": "report",
            "volume_s3_uri": "s3://bucket/vol.nii.gz",
        })
        parsed = input_fn(body, "application/json")
        result = predict_fn(parsed, context)

        assert result["mode"] == "report"
        assert result["report_text"] == "FINDINGS: Normal."

        output = output_fn(result, "application/json")
        parsed_output = json.loads(output)
        assert parsed_output["report_text"] == "FINDINGS: Normal."


class TestBackgroundJobFlow:
    """Tests for background submit + poll pattern."""

    def test_input_fn_accepts_background_field(self) -> None:
        """input_fn passes through the background flag."""
        body = json.dumps({
            "model": "merlin",
            "mode": "findings",
            "volume_s3_uri": "s3://bucket/vol.nii.gz",
            "background": True,
        })
        result = input_fn(body, "application/json")
        assert result["_background"] is True

    def test_input_fn_accepts_poll_job_id(self) -> None:
        """input_fn passes through poll_job_id."""
        body = json.dumps({
            "model": "merlin",
            "poll_job_id": "abc-123",
        })
        result = input_fn(body, "application/json")
        assert result["_poll_job_id"] == "abc-123"

    def test_input_fn_poll_no_text_required(self) -> None:
        """poll_job_id requests don't require text or volume_s3_uri."""
        body = json.dumps({
            "model": "merlin",
            "poll_job_id": "abc-123",
        })
        # Should not raise
        result = input_fn(body, "application/json")
        assert "_poll_job_id" in result
        assert "text" not in result

    def test_submit_returns_job_id(self) -> None:
        """Background submit returns a job_id and 'submitted' status."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.predict_fn.return_value = {"mode": "findings", "findings": {}}
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {"model_dir": "/tmp"})
        mock_registry.get_config.return_value = {"supports_chat": False}

        context = {"registry": mock_registry, "session_manager": ChatSessionManager(), "job_manager": JobManager()}

        input_data = {
            "_model_id": "merlin",
            "mode": "findings",
            "volume_s3_uri": "s3://bucket/vol.nii.gz",
            "_background": True,
        }
        result = predict_fn(input_data, context)
        assert result["status"] == "submitted"
        assert "job_id" in result

    def test_poll_returns_completed_result(self) -> None:
        """Poll returns 'completed' with the handler result after job finishes."""
        import time

        mock_registry = MagicMock()
        mock_handler = MagicMock()
        expected = {"mode": "findings", "findings": {"ascites": 0.87}}
        mock_handler.predict_fn.return_value = expected
        mock_registry.is_known_model.return_value = True
        mock_registry.get_or_load.return_value = (mock_handler, {"model_dir": "/tmp"})
        mock_registry.get_config.return_value = {"supports_chat": False}

        jm = JobManager()
        context = {"registry": mock_registry, "session_manager": ChatSessionManager(), "job_manager": jm}

        # Submit
        input_data = {
            "_model_id": "merlin",
            "mode": "findings",
            "volume_s3_uri": "s3://bucket/vol.nii.gz",
            "_background": True,
        }
        submit_result = predict_fn(input_data, context)
        job_id = submit_result["job_id"]

        # Poll until completed
        for _ in range(50):
            poll_result = predict_fn({"_poll_job_id": job_id, "_model_id": "merlin"}, context)
            if poll_result["status"] == "completed":
                break
            time.sleep(0.05)

        assert poll_result["status"] == "completed"
        assert poll_result["result"] == expected

    def test_poll_unknown_job(self) -> None:
        """Polling a non-existent job returns 'not_found'."""
        context = {
            "registry": MagicMock(),
            "session_manager": ChatSessionManager(),
            "job_manager": JobManager(),
        }
        result = predict_fn({"_poll_job_id": "no-such-id", "_model_id": "merlin"}, context)
        assert result["status"] == "not_found"
