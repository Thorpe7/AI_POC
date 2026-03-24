"""Tests for the MerlinHandler."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from handlers.merlin_handler import MAX_LOADED_MODES, VALID_MODES, MerlinHandler


class TestMerlinHandlerInit:
    """Tests for handler construction and config validation."""

    def test_valid_config(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        assert handler.config == sample_merlin_config

    def test_missing_fields_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required fields"):
            MerlinHandler({"display_name": "Bad"})

    def test_valid_modes_match_config(self, sample_merlin_config: dict[str, Any]) -> None:
        assert set(sample_merlin_config["valid_modes"]) == VALID_MODES


class TestMerlinHandlerModelFn:
    """Tests for model_fn."""

    def test_returns_model_dir_and_empty_modes(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        result = handler.model_fn("/tmp/weights")
        assert result["model_dir"] == "/tmp/weights"
        assert isinstance(result["loaded_modes"], OrderedDict)
        assert len(result["loaded_modes"]) == 0


class TestMerlinHandlerInputFn:
    """Tests for input_fn validation."""

    def test_valid_findings_request(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        body = json.dumps({
            "mode": "findings",
            "volume_s3_uri": "s3://bucket/vol.nii.gz",
        })
        result = handler.input_fn(body, "application/json")
        assert result["mode"] == "findings"
        assert result["volume_s3_uri"] == "s3://bucket/vol.nii.gz"

    def test_invalid_mode_raises(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        body = json.dumps({"mode": "invalid", "volume_s3_uri": "s3://b/v.nii.gz"})
        with pytest.raises(ValueError, match="Invalid mode"):
            handler.input_fn(body, "application/json")

    def test_missing_volume_uri_raises(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        body = json.dumps({"mode": "findings"})
        with pytest.raises(ValueError, match="volume_s3_uri"):
            handler.input_fn(body, "application/json")

    def test_retrieval_requires_query_texts(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        body = json.dumps({"mode": "retrieval", "volume_s3_uri": "s3://b/v.nii.gz"})
        with pytest.raises(ValueError, match="query_texts"):
            handler.input_fn(body, "application/json")

    def test_retrieval_with_query_texts(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        body = json.dumps({
            "mode": "retrieval",
            "volume_s3_uri": "s3://b/v.nii.gz",
            "query_texts": ["ascites"],
        })
        result = handler.input_fn(body, "application/json")
        assert result["query_texts"] == ["ascites"]

    def test_unsupported_content_type(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        with pytest.raises(ValueError, match="Unsupported content type"):
            handler.input_fn("{}", "text/plain")


class TestMerlinHandlerPredictFn:
    """Tests for predict_fn with mocked S3 and merlin imports."""

    @staticmethod
    def _make_handler_and_model_dict(
        config: dict[str, Any],
    ) -> tuple[MerlinHandler, dict[str, Any]]:
        handler = MerlinHandler(config)
        model_dict = handler.model_fn("/tmp/weights")
        return handler, model_dict

    @patch("handlers.merlin_handler.boto3")
    def test_findings_mode(
        self, mock_boto3: MagicMock, sample_merlin_config: dict[str, Any]
    ) -> None:
        handler, model_dict = self._make_handler_and_model_dict(sample_merlin_config)
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_merlin = MagicMock()
        # _predict_findings calls model.model.encode_image(volume)
        mock_merlin.model.encode_image.return_value = (
            torch.tensor([[0.1, 0.2]]),
            torch.tensor([[0.3, 0.4]]),
        )

        with (
            patch.object(handler, "_preprocess_volume", return_value=torch.zeros(1, 1, 4, 4, 4)),
            patch.object(handler, "_get_or_load_mode", return_value=mock_merlin),
        ):
            result = handler.predict_fn(
                {"mode": "findings", "volume_s3_uri": "s3://bucket/vol.nii.gz"},
                model_dict,
            )

        assert result["mode"] == "findings"
        assert "findings" in result
        assert isinstance(result["findings"], dict)
        mock_s3.delete_object.assert_not_called()

    @patch("handlers.merlin_handler.boto3")
    def test_phenotype_mode(
        self, mock_boto3: MagicMock, sample_merlin_config: dict[str, Any]
    ) -> None:
        handler, model_dict = self._make_handler_and_model_dict(sample_merlin_config)
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_merlin = MagicMock()
        mock_merlin.return_value = torch.tensor([[0.9, 0.1]])

        with (
            patch.object(handler, "_preprocess_volume", return_value=torch.zeros(1, 1, 4, 4, 4)),
            patch.object(handler, "_get_or_load_mode", return_value=mock_merlin),
        ):
            result = handler.predict_fn(
                {"mode": "phenotype", "volume_s3_uri": "s3://bucket/vol.nii.gz"},
                model_dict,
            )

        assert result["mode"] == "phenotype"
        assert "phenotypes" in result
        assert "top_k" in result

    @patch("handlers.merlin_handler.boto3")
    def test_retrieval_mode(
        self, mock_boto3: MagicMock, sample_merlin_config: dict[str, Any]
    ) -> None:
        handler, model_dict = self._make_handler_and_model_dict(sample_merlin_config)
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        # Mock the model — _predict_retrieval calls model(volume) and model.model.encode_text
        mock_merlin = MagicMock()
        img_feats = torch.tensor([[0.6, 0.8]])
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        mock_merlin.return_value = img_feats
        text_feats = torch.tensor([[0.6, 0.8], [0.1, 0.9]])
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        mock_merlin.model.encode_text.return_value = text_feats

        with (
            patch.object(handler, "_preprocess_volume", return_value=torch.zeros(1, 1, 4, 4, 4)),
            patch.object(handler, "_get_or_load_mode", return_value=mock_merlin),
        ):
            result = handler.predict_fn(
                {
                    "mode": "retrieval",
                    "volume_s3_uri": "s3://bucket/vol.nii.gz",
                    "query_texts": ["ascites", "lesion"],
                },
                model_dict,
            )

        assert result["mode"] == "retrieval"
        assert len(result["similarity_scores"]) == 2
        assert result["similarity_scores"][0]["query"] == "ascites"

    @patch("handlers.merlin_handler.boto3")
    def test_prediction_mode(
        self, mock_boto3: MagicMock, sample_merlin_config: dict[str, Any]
    ) -> None:
        handler, model_dict = self._make_handler_and_model_dict(sample_merlin_config)
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_merlin = MagicMock()
        mock_merlin.return_value = torch.tensor([[0.45, 0.12, 0.78, 0.33, 0.91, 0.05]])

        with (
            patch.object(handler, "_preprocess_volume", return_value=torch.zeros(1, 1, 4, 4, 4)),
            patch.object(handler, "_get_or_load_mode", return_value=mock_merlin),
        ):
            result = handler.predict_fn(
                {"mode": "prediction", "volume_s3_uri": "s3://bucket/vol.nii.gz"},
                model_dict,
            )

        assert result["mode"] == "prediction"
        assert "risk_scores" in result

    @patch("handlers.merlin_handler.boto3")
    def test_report_mode(
        self, mock_boto3: MagicMock, sample_merlin_config: dict[str, Any]
    ) -> None:
        handler, model_dict = self._make_handler_and_model_dict(sample_merlin_config)
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_merlin = MagicMock()
        mock_merlin.generate.return_value = ["FINDINGS: Normal."]

        with (
            patch.object(handler, "_preprocess_volume", return_value=torch.zeros(1, 1, 4, 4, 4)),
            patch.object(handler, "_get_or_load_mode", return_value=mock_merlin),
        ):
            result = handler.predict_fn(
                {"mode": "report", "volume_s3_uri": "s3://bucket/vol.nii.gz"},
                model_dict,
            )

        assert result["mode"] == "report"
        assert result["report_text"] == "FINDINGS: Normal."


class TestMerlinHandlerS3Lifecycle:
    """Tests for S3 volume handling — volumes are kept for reuse, not deleted."""

    @patch("handlers.merlin_handler.boto3")
    def test_s3_volume_not_deleted_on_success(
        self, mock_boto3: MagicMock, sample_merlin_config: dict[str, Any]
    ) -> None:
        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_merlin = MagicMock()
        mock_merlin.model.encode_image.return_value = (torch.tensor([[0.1]]), torch.tensor([[0.2]]))

        with (
            patch.object(handler, "_preprocess_volume", return_value=torch.zeros(1, 1, 4, 4, 4)),
            patch.object(handler, "_get_or_load_mode", return_value=mock_merlin),
        ):
            handler.predict_fn(
                {"mode": "findings", "volume_s3_uri": "s3://bucket/vol.nii.gz"},
                model_dict,
            )

        mock_s3.delete_object.assert_not_called()

    @patch("handlers.merlin_handler.boto3")
    def test_s3_volume_not_deleted_on_failure(
        self, mock_boto3: MagicMock, sample_merlin_config: dict[str, Any]
    ) -> None:
        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_merlin = MagicMock()
        mock_merlin.model.encode_image.side_effect = RuntimeError("Inference failed")

        with (
            patch.object(handler, "_preprocess_volume", return_value=torch.zeros(1, 1, 4, 4, 4)),
            patch.object(handler, "_get_or_load_mode", return_value=mock_merlin),
            pytest.raises(RuntimeError, match="Inference failed"),
        ):
            handler.predict_fn(
                {"mode": "findings", "volume_s3_uri": "s3://bucket/vol.nii.gz"},
                model_dict,
            )

        mock_s3.delete_object.assert_not_called()


class TestMerlinModeEviction:
    """Tests for internal mode LRU eviction."""

    def test_evicts_oldest_mode_when_full(self, sample_merlin_config: dict[str, Any]) -> None:
        import sys

        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")

        # Pre-populate with MAX_LOADED_MODES modes
        mock_model_a = MagicMock(spec=torch.nn.Module)
        mock_model_b = MagicMock(spec=torch.nn.Module)
        model_dict["loaded_modes"]["findings"] = mock_model_a
        model_dict["loaded_modes"]["phenotype"] = mock_model_b

        assert len(model_dict["loaded_modes"]) == MAX_LOADED_MODES

        # Loading a third mode should evict the oldest (findings)
        mock_new_model = MagicMock()
        mock_merlin_module = MagicMock()
        mock_merlin_module.Merlin.return_value = mock_new_model

        with (
            patch.dict(sys.modules, {"merlin": mock_merlin_module}),
            patch("handlers.merlin_handler.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.empty_cache = MagicMock()
            handler._get_or_load_mode("report", model_dict)

        assert "findings" not in model_dict["loaded_modes"]
        assert "phenotype" in model_dict["loaded_modes"]
        assert "report" in model_dict["loaded_modes"]
        assert len(model_dict["loaded_modes"]) == MAX_LOADED_MODES

    def test_cache_hit_does_not_evict(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")

        mock_model = MagicMock()
        model_dict["loaded_modes"]["findings"] = mock_model

        result = handler._get_or_load_mode("findings", model_dict)
        assert result is mock_model
        assert len(model_dict["loaded_modes"]) == 1


class TestMerlinHandlerOutputFn:
    """Tests for output_fn."""

    def test_serializes_dict(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        result = json.loads(
            handler.output_fn({"mode": "findings", "findings": {"a": 0.5}}, "application/json")
        )
        assert result == {"mode": "findings", "findings": {"a": 0.5}}

    def test_unsupported_accept_type(self, sample_merlin_config: dict[str, Any]) -> None:
        handler = MerlinHandler(sample_merlin_config)
        with pytest.raises(ValueError, match="Unsupported accept type"):
            handler.output_fn({}, "text/xml")


class TestMerlinModeOutputFormats:
    """Verify each mode returns the expected output schema for the notebook renderer."""

    @staticmethod
    def _run_mode(
        handler: MerlinHandler,
        model_dict: dict[str, Any],
        mock_merlin: MagicMock,
        mode: str,
        extra_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        input_data: dict[str, Any] = {
            "mode": mode,
            "volume_s3_uri": "s3://bucket/vol.nii.gz",
        }
        if extra_input:
            input_data.update(extra_input)
        with (
            patch("handlers.merlin_handler.boto3") as mock_boto3,
            patch.object(handler, "_preprocess_volume", return_value=torch.zeros(1, 1, 4, 4, 4)),
            patch.object(handler, "_get_or_load_mode", return_value=mock_merlin),
        ):
            mock_boto3.client.return_value = MagicMock()
            return handler.predict_fn(input_data, model_dict)

    def test_findings_returns_labeled_dict(self, sample_merlin_config: dict[str, Any]) -> None:
        """Findings mode should return {findings: {label: score}} with human-readable labels."""
        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")
        mock_merlin = MagicMock()
        # Simulate 1692-dim output
        ehr_logits = torch.randn(1, 1692)
        mock_merlin.model.encode_image.return_value = (torch.randn(1, 512), ehr_logits)

        result = self._run_mode(handler, model_dict, mock_merlin, "findings")

        assert result["mode"] == "findings"
        assert isinstance(result["findings"], dict)
        assert len(result["findings"]) == 20  # top-K
        # Labels should be human-readable strings, not "feature_N"
        for key in result["findings"]:
            assert not key.startswith("feature_"), f"Expected human label, got {key}"
            assert isinstance(result["findings"][key], float)

    def test_phenotype_returns_labeled_dict_and_top_k(self, sample_merlin_config: dict[str, Any]) -> None:
        """Phenotype mode should return {phenotypes: {label: score}, top_k: [...]}."""
        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")
        mock_merlin = MagicMock()
        mock_merlin.return_value = torch.randn(1, 1692)

        result = self._run_mode(handler, model_dict, mock_merlin, "phenotype")

        assert result["mode"] == "phenotype"
        assert isinstance(result["phenotypes"], dict)
        assert len(result["phenotypes"]) == 20
        assert isinstance(result["top_k"], list)
        assert len(result["top_k"]) == 20
        for item in result["top_k"]:
            assert "name" in item and "score" in item

    def test_retrieval_returns_scored_queries(self, sample_merlin_config: dict[str, Any]) -> None:
        """Retrieval mode should return {similarity_scores: [{query, score}]}."""
        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")
        mock_merlin = MagicMock()
        img_feats = torch.randn(1, 512)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        mock_merlin.return_value = img_feats
        text_feats = torch.randn(3, 512)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        mock_merlin.model.encode_text.return_value = text_feats

        result = self._run_mode(
            handler, model_dict, mock_merlin, "retrieval",
            extra_input={"query_texts": ["ascites", "mass", "effusion"]},
        )

        assert result["mode"] == "retrieval"
        assert len(result["similarity_scores"]) == 3
        for i, q in enumerate(["ascites", "mass", "effusion"]):
            assert result["similarity_scores"][i]["query"] == q
            assert isinstance(result["similarity_scores"][i]["score"], float)

    def test_prediction_returns_risk_scores(self, sample_merlin_config: dict[str, Any]) -> None:
        """Prediction mode should return {risk_scores: [float]}."""
        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")
        mock_merlin = MagicMock()
        mock_merlin.return_value = torch.randn(1, 6)

        result = self._run_mode(handler, model_dict, mock_merlin, "prediction")

        assert result["mode"] == "prediction"
        assert isinstance(result["risk_scores"], dict)
        assert len(result["risk_scores"]) == 6
        assert "All-cause mortality" in result["risk_scores"]

    def test_report_returns_text(self, sample_merlin_config: dict[str, Any]) -> None:
        """Report mode should return {report_text: str}."""
        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")
        mock_merlin = MagicMock()
        mock_merlin.generate.return_value = ["FINDINGS: Bilateral pleural effusions."]

        result = self._run_mode(handler, model_dict, mock_merlin, "report")

        assert result["mode"] == "report"
        assert isinstance(result["report_text"], str)
        assert "pleural effusions" in result["report_text"]


class TestMerlinMetaTensorPatch:
    """Verify the meta-tensor monkey-patches are restored after Merlin construction."""

    def _build_mocks(self, merlin_side_effect=None):
        """Build the mock modules needed for _get_or_load_mode tests."""
        mock_merlin = MagicMock()
        if merlin_side_effect:
            mock_merlin.Merlin.side_effect = merlin_side_effect
        else:
            mock_merlin.Merlin.return_value = MagicMock()

        mock_accelerate = MagicMock()
        mock_accelerate.init_empty_weights = MagicMock()
        mock_tm = MagicMock()
        mock_tm.init_empty_weights = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.AutoConfig.from_pretrained = classmethod(lambda cls, *a, **kw: MagicMock())
        mock_transformers.AutoTokenizer.from_pretrained = classmethod(lambda cls, *a, **kw: MagicMock())
        mock_transformers.AutoModelForCausalLM.from_pretrained = classmethod(lambda cls, *a, **kw: MagicMock())
        mock_transformers.LlamaConfig = MagicMock()
        mock_transformers.LlamaTokenizer = MagicMock()

        return {
            "merlin": mock_merlin,
            "accelerate": mock_accelerate,
            "transformers.modeling_utils": mock_tm,
            "transformers": mock_transformers,
        }

    def test_patches_restore_after_success(self, sample_merlin_config: dict[str, Any]) -> None:
        """All monkey-patches should be reverted after Merlin construction."""
        import sys

        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")

        orig_lsd = torch.nn.Module.load_state_dict
        mocks = self._build_mocks()

        with (
            patch.dict(sys.modules, mocks),
            patch("handlers.merlin_handler.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.empty_cache = MagicMock()
            mock_torch.nn.Module.load_state_dict = orig_lsd
            mock_torch.nn.Module = torch.nn.Module
            handler._get_or_load_mode("report", model_dict)

        # Original should be restored
        assert torch.nn.Module.load_state_dict is orig_lsd

    def test_patches_restore_on_error(self, sample_merlin_config: dict[str, Any]) -> None:
        """All monkey-patches should be reverted even if Merlin() raises."""
        import sys

        handler = MerlinHandler(sample_merlin_config)
        model_dict = handler.model_fn("/tmp/weights")

        orig_lsd = torch.nn.Module.load_state_dict
        mocks = self._build_mocks(merlin_side_effect=RuntimeError("Failed to load"))

        with (
            patch.dict(sys.modules, mocks),
            patch("handlers.merlin_handler.torch") as mock_torch,
            pytest.raises(RuntimeError, match="Failed to load"),
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.empty_cache = MagicMock()
            mock_torch.nn.Module.load_state_dict = orig_lsd
            mock_torch.nn.Module = torch.nn.Module
            handler._get_or_load_mode("report", model_dict)

        # Original should be restored even after error
        assert torch.nn.Module.load_state_dict is orig_lsd


class TestPhenotypeLabels:
    """Verify phenotype label loading."""

    def test_labels_file_exists(self) -> None:
        from handlers.merlin_handler import _get_phenotype_labels
        labels = _get_phenotype_labels()
        assert len(labels) == 1692

    def test_labels_are_strings(self) -> None:
        from handlers.merlin_handler import _get_phenotype_labels
        labels = _get_phenotype_labels()
        assert all(isinstance(l, str) for l in labels)

    def test_known_labels_present(self) -> None:
        from handlers.merlin_handler import _get_phenotype_labels
        labels = _get_phenotype_labels()
        # Spot-check a few known phenotypes from the CSV
        assert "Intestinal infection" in labels
        assert "Septicemia" in labels
        assert "Viral hepatitis C" in labels
