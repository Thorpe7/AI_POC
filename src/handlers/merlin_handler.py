"""Handler for the Merlin 3D CT Vision-Language model (stanfordmimi/Merlin).

Merlin uses a custom architecture (3D ResNet + Longformer) — not HuggingFace
transformers — so it needs its own handler. Supports five operating modes:
findings classification, phenotype classification, cross-modal retrieval,
5-year disease prediction, and report generation.

Volumes are referenced by S3 URI, downloaded on-demand, and deleted after
inference via ``try/finally``.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import torch

from handlers.base import BaseHandler
from handlers.utils import parse_s3_uri

logger = logging.getLogger(__name__)

VALID_MODES = {"findings", "phenotype", "retrieval", "prediction", "report"}

# 1692 phenotype labels from stanfordmimi/Merlin phenotypes.csv.
# Index maps directly to the model's output vector.
_PHENOTYPE_LABELS: list[str] | None = None


def _get_phenotype_labels() -> list[str]:
    """Lazy-load phenotype labels from the configs directory."""
    global _PHENOTYPE_LABELS  # noqa: PLW0603
    if _PHENOTYPE_LABELS is None:
        # Labels file lives alongside the handler in handlers/
        labels_path = Path(__file__).resolve().parent / "phenotype_labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                _PHENOTYPE_LABELS = json.load(f)
            logger.info("Loaded %d phenotype labels", len(_PHENOTYPE_LABELS))
        else:
            logger.warning("phenotype_labels.json not found at %s", labels_path)
            _PHENOTYPE_LABELS = []
    return _PHENOTYPE_LABELS

# Merlin() constructor kwargs for each mode.
# The default constructor (no flags) loads the contrastive model used for
# findings classification and retrieval (ImageEmbedding mode).
MODE_INIT_FLAGS: dict[str, dict[str, bool]] = {
    "findings": {},  # default contrastive model
    "phenotype": {"PhenotypeCls": True},
    "retrieval": {"ImageEmbedding": True},
    "prediction": {"FiveYearPred": True},
    "report": {"RadiologyReport": True},
}

MAX_LOADED_MODES = 2

REQUIRED_CONFIG_KEYS = {
    "estimated_gpu_memory_gb",
    "weights_s3_uri",
    "valid_modes",
}


class MerlinHandler(BaseHandler):
    """Handler for Merlin 3D CT VLM with lazy per-mode loading and S3 volume lifecycle."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._validate_config()

    def _validate_config(self) -> None:
        missing = REQUIRED_CONFIG_KEYS - set(self._config.keys())
        if missing:
            raise ValueError(f"MerlinHandler config missing required fields: {missing}")

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    # -- SageMaker contract --------------------------------------------------

    def model_fn(self, model_dir: str) -> dict[str, Any]:
        """Return a model dict; modes are loaded lazily."""
        return {"model_dir": model_dir, "loaded_modes": OrderedDict()}

    def input_fn(self, request_body: str | bytes, request_content_type: str) -> dict[str, Any]:
        """Parse and validate the incoming request."""
        if request_content_type != "application/json":
            raise ValueError(f"Unsupported content type: {request_content_type}")

        payload = json.loads(request_body)

        mode = payload.get("mode")
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}"
            )

        if "volume_s3_uri" not in payload:
            raise ValueError("Request must include 'volume_s3_uri'")

        if mode == "retrieval" and "query_texts" not in payload:
            raise ValueError("Retrieval mode requires 'query_texts'")

        result: dict[str, Any] = {
            "mode": mode,
            "volume_s3_uri": payload["volume_s3_uri"],
        }
        if "query_texts" in payload:
            result["query_texts"] = payload["query_texts"]
        return result

    def predict_fn(self, input_data: dict[str, Any], model_dict: dict[str, Any]) -> dict[str, Any]:
        """Download volume, preprocess, run mode-specific inference.

        S3 volumes are **not** deleted after inference so the frontend can
        reuse the same upload across multiple mode runs on the same series.
        """
        mode = input_data["mode"]
        volume_s3_uri = input_data["volume_s3_uri"]

        bucket, key = parse_s3_uri(volume_s3_uri)
        s3 = boto3.client("s3")
        local_path: str | None = None

        try:
            # Download volume to temp file
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                local_path = tmp.name
                logger.info("Downloading volume from s3://%s/%s", bucket, key)
                s3.download_file(bucket, key, local_path)

            # Preprocess
            volume_tensor = self._preprocess_volume(local_path)

            # Load mode and run inference
            merlin_model = self._get_or_load_mode(mode, model_dict)
            result = self._dispatch_mode(mode, merlin_model, volume_tensor, input_data)

        finally:
            # Clean up local temp file only — S3 volume is kept for reuse
            if local_path is not None:
                try:
                    Path(local_path).unlink(missing_ok=True)
                except Exception:
                    pass

        return result

    def output_fn(self, prediction: Any, accept: str) -> str:  # noqa: ANN401
        """Serialize the prediction dict to JSON."""
        if accept not in ("application/json", "*/*"):
            raise ValueError(f"Unsupported accept type: {accept}")
        return json.dumps(prediction)

    # -- Internal helpers -----------------------------------------------------

    def _preprocess_volume(self, nifti_path: str) -> torch.Tensor:
        """Load NIfTI, resample to target shape, normalize HU values to [0, 1]."""
        import nibabel as nib  # type: ignore[import-untyped]
        import numpy as np
        from monai.transforms import Resize  # type: ignore[import-untyped]

        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj, dtype=np.float32)

        # HU clipping
        preproc = self._config.get("preprocessing", {})
        hu_min, hu_max = preproc.get("hu_clip_range", [-1024, 3071])
        data = np.clip(data, hu_min, hu_max)

        # Normalize to [0, 1]
        data = (data - hu_min) / (hu_max - hu_min)

        # Convert to tensor: (1, D, H, W) for monai Resize
        tensor = torch.from_numpy(data).unsqueeze(0)

        target_shape = preproc.get("target_shape", [224, 224, 160])
        resize = Resize(spatial_size=target_shape, mode="trilinear")
        tensor = resize(tensor)

        # Add batch dim: (1, 1, D, H, W)
        return tensor.unsqueeze(0)

    @staticmethod
    def _materialize_meta_tensors(module: torch.nn.Module) -> None:
        """Replace any leftover meta-device tensors with real zero tensors.

        Safety net after model construction — if any parameter or buffer is
        still on the ``meta`` device (no backing data), replace it in-place so
        ``.cuda()`` and inference don't crash with
        ``Cannot copy out of meta tensor``.
        """
        for name, param in list(module.named_parameters()):
            if param.is_meta:
                logger.warning("Materializing meta parameter: %s %s", name, param.shape)
                parts = name.split(".")
                target = module
                for part in parts[:-1]:
                    target = getattr(target, part)
                real = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                    requires_grad=param.requires_grad,
                )
                setattr(target, parts[-1], real)

        for name, buf in list(module.named_buffers()):
            if buf.is_meta:
                logger.warning("Materializing meta buffer: %s %s", name, buf.shape)
                parts = name.split(".")
                target = module
                for part in parts[:-1]:
                    target = getattr(target, part)
                target.register_buffer(
                    parts[-1],
                    torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"),
                )

    def _get_or_load_mode(
        self, mode: str, model_dict: dict[str, Any]
    ) -> Any:  # noqa: ANN401
        """Lazy-load a Merlin instance for the requested mode with LRU eviction."""
        loaded_modes: OrderedDict[str, Any] = model_dict["loaded_modes"]

        if mode in loaded_modes:
            loaded_modes.move_to_end(mode)
            logger.info("Merlin mode cache hit: %s", mode)
            return loaded_modes[mode]

        # Evict oldest if at capacity
        while len(loaded_modes) >= MAX_LOADED_MODES:
            evicted_mode, evicted_model = loaded_modes.popitem(last=False)
            logger.info("Evicting Merlin mode: %s", evicted_mode)
            try:
                evicted_model.cpu()
            except Exception:
                pass
            del evicted_model
            torch.cuda.empty_cache()

        from merlin import Merlin  # type: ignore[import-untyped]

        # ---- Patches applied during Merlin() construction ----
        #
        # merlin-vlm calls from_pretrained("StanfordAIMI/RadLLaMA-7b") for
        # the report mode.  Three problems:
        #   1. RadLLaMA's config.json lacks model_type → AutoConfig fails
        #   2. RadLLaMA's tokenizer is a custom CheXagentTokenizer that
        #      requires LlamaTokenizerFast + tokenizer.json (missing)
        #   3. Downloading 27GB at runtime is unacceptable
        #
        # Solution: pre-download RadLLaMA to S3 alongside Merlin weights.
        # The registry downloads it to {model_dir}/radllama-7b/.  We
        # redirect all from_pretrained calls to the local path and fix
        # the config/tokenizer issues.
        #
        # Additionally, transformers 4.51+ creates meta tensors via
        # accelerate.init_empty_weights — we make that a no-op.

        import contextlib

        import accelerate  # type: ignore[import-untyped]
        import transformers.modeling_utils as _tm  # type: ignore[import-untyped]
        from transformers import (  # type: ignore[import-untyped]
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            LlamaConfig,
            LlamaTokenizer,
        )

        # Local path for pre-downloaded RadLLaMA weights
        radllama_local = str(Path(model_dict["model_dir"]) / "radllama-7b")

        def _resolve_radllama(name: str) -> str | None:
            """Return local path if name refers to RadLLaMA, else None."""
            if "radllama" in str(name).lower():
                logger.info("Resolving RadLLaMA → local path: %s", radllama_local)
                return radllama_local
            return None

        # -- 1a. AutoConfig: redirect to LlamaConfig + local path --
        _orig_ac = AutoConfig.from_pretrained.__func__  # type: ignore[attr-defined]

        @classmethod  # type: ignore[misc]
        def _patched_ac(cls: type, name_or_path: str, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
            local = _resolve_radllama(name_or_path)
            if local:
                return LlamaConfig.from_pretrained(local, *a, **kw)
            return _orig_ac(cls, name_or_path, *a, **kw)

        # -- 1b. AutoTokenizer: redirect to LlamaTokenizer + local path --
        _orig_at = AutoTokenizer.from_pretrained.__func__  # type: ignore[attr-defined]

        @classmethod  # type: ignore[misc]
        def _patched_at(cls: type, name_or_path: str, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
            local = _resolve_radllama(name_or_path)
            if local:
                return LlamaTokenizer.from_pretrained(local, *a, **kw)
            return _orig_at(cls, name_or_path, *a, **kw)  # type: ignore[attr-defined]

        # -- 1c. AutoModelForCausalLM: redirect to local path --
        _orig_am = AutoModelForCausalLM.from_pretrained.__func__  # type: ignore[attr-defined]

        @classmethod  # type: ignore[misc]
        def _patched_am(cls: type, name_or_path: str, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
            local = _resolve_radllama(name_or_path)
            if local:
                return _orig_am(cls, local, *a, **kw)  # type: ignore[attr-defined]
            return _orig_am(cls, name_or_path, *a, **kw)  # type: ignore[attr-defined]

        # -- 2. Meta-tensor prevention: no-op init_empty_weights --
        _orig_init_empty = accelerate.init_empty_weights
        _tm_init_empty = getattr(_tm, "init_empty_weights", None)

        @contextlib.contextmanager
        def _noop_init_empty(*a: Any, **kw: Any) -> Any:  # noqa: ANN401
            yield

        # -- 3. load_state_dict: force assign=True --
        _orig_lsd = torch.nn.Module.load_state_dict

        def _assign_lsd(self: torch.nn.Module, state_dict: Any, strict: bool = True, **kw: Any) -> Any:  # noqa: ANN401
            kw["assign"] = True
            return _orig_lsd(self, state_dict, strict=strict, **kw)

        flags = MODE_INIT_FLAGS[mode]
        logger.info("Loading Merlin mode: %s (flags=%s)", mode, flags)

        # Apply all patches
        AutoConfig.from_pretrained = _patched_ac  # type: ignore[assignment]
        AutoTokenizer.from_pretrained = _patched_at  # type: ignore[assignment]
        AutoModelForCausalLM.from_pretrained = _patched_am  # type: ignore[assignment]
        accelerate.init_empty_weights = _noop_init_empty  # type: ignore[assignment]
        if _tm_init_empty is not None:
            _tm.init_empty_weights = _noop_init_empty  # type: ignore[assignment]
        torch.nn.Module.load_state_dict = _assign_lsd  # type: ignore[assignment]
        try:
            merlin_model = Merlin(**flags)
        finally:
            AutoConfig.from_pretrained = classmethod(_orig_ac)  # type: ignore[assignment]
            AutoTokenizer.from_pretrained = classmethod(_orig_at)  # type: ignore[assignment]
            AutoModelForCausalLM.from_pretrained = classmethod(_orig_am)  # type: ignore[assignment]
            accelerate.init_empty_weights = _orig_init_empty  # type: ignore[assignment]
            if _tm_init_empty is not None:
                _tm.init_empty_weights = _tm_init_empty  # type: ignore[assignment]
            torch.nn.Module.load_state_dict = _orig_lsd  # type: ignore[method-assign]

        # Safety net: materialise anything that slipped through
        self._materialize_meta_tensors(merlin_model)

        if torch.cuda.is_available():
            merlin_model = merlin_model.cuda()
        merlin_model.eval()

        loaded_modes[mode] = merlin_model
        return merlin_model

    def _dispatch_mode(
        self,
        mode: str,
        merlin_model: Any,  # noqa: ANN401
        volume_tensor: torch.Tensor,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Route to the correct mode-specific inference method."""
        # Move input tensor to same device as the model
        try:
            device = next(merlin_model.parameters()).device
            volume_tensor = volume_tensor.to(device)
        except (StopIteration, TypeError):
            pass

        if mode == "findings":
            return self._predict_findings(merlin_model, volume_tensor)
        elif mode == "phenotype":
            return self._predict_phenotype(merlin_model, volume_tensor)
        elif mode == "retrieval":
            return self._predict_retrieval(merlin_model, volume_tensor, input_data["query_texts"])
        elif mode == "prediction":
            return self._predict_prediction(merlin_model, volume_tensor)
        elif mode == "report":
            return self._predict_report(merlin_model, volume_tensor)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def _predict_findings(model: Any, volume: torch.Tensor) -> dict[str, Any]:  # noqa: ANN401
        """Default contrastive model — extract findings via the image encoder directly.

        The top-level forward() requires text input, but the image encoder's
        i3_resnet returns (contrastive_features, ehr_features) where ehr_features
        are 1692-class phenotype logits.
        """
        labels = _get_phenotype_labels()
        with torch.inference_mode():
            contrastive_features, ehr_features = model.model.encode_image(volume)
        probs = torch.sigmoid(ehr_features).squeeze(0).cpu()
        top_k = 20
        values, indices = torch.topk(probs, min(top_k, probs.shape[0]))
        findings = {}
        for idx, val in zip(indices, values):
            i = idx.item()
            name = labels[i] if i < len(labels) else f"feature_{i}"
            findings[name] = round(val.item(), 4)
        return {
            "mode": "findings",
            "findings": findings,
        }

    @staticmethod
    def _predict_phenotype(model: Any, volume: torch.Tensor) -> dict[str, Any]:  # noqa: ANN401
        """PhenotypeCls mode — returns top phenotype classifications with labels."""
        labels = _get_phenotype_labels()
        with torch.inference_mode():
            logits = model(volume)
        probs = torch.sigmoid(logits).squeeze(0).cpu()
        top_k = 20
        values, indices = torch.topk(probs, min(top_k, probs.shape[0]))
        phenotypes = {}
        for idx, val in zip(indices, values):
            i = idx.item()
            name = labels[i] if i < len(labels) else f"phenotype_{i}"
            phenotypes[name] = round(val.item(), 4)
        return {"mode": "phenotype", "phenotypes": phenotypes, "top_k": [
            {"name": name, "score": score} for name, score in phenotypes.items()
        ]}

    @staticmethod
    def _predict_retrieval(
        model: Any,  # noqa: ANN401
        volume: torch.Tensor,
        query_texts: list[str],
    ) -> dict[str, Any]:
        """ImageEmbedding mode — compute cosine similarity between image and text."""
        with torch.inference_mode():
            image_features = model(volume)
            # Normalize image features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Encode query texts through the underlying text encoder
            text_features = model.model.encode_text(query_texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            scores = (image_features @ text_features.T).squeeze(0).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]
        results = [
            {"query": q, "score": float(s)} for q, s in zip(query_texts, scores)
        ]
        return {"mode": "retrieval", "similarity_scores": results}

    # Labels for the 5-year disease prediction output vector (from Merlin paper).
    _PREDICTION_LABELS = [
        "All-cause mortality",
        "Lung cancer",
        "COPD",
        "Cardiovascular disease",
        "Type 2 diabetes",
        "Chronic kidney disease",
    ]

    @staticmethod
    def _predict_prediction(model: Any, volume: torch.Tensor) -> dict[str, Any]:  # noqa: ANN401
        """FiveYearPred mode — returns 5-year disease prediction scores."""
        with torch.inference_mode():
            logits = model(volume)
        scores = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        if not isinstance(scores, list):
            scores = [scores]
        labeled = {}
        for i, s in enumerate(scores):
            name = (
                MerlinHandler._PREDICTION_LABELS[i]
                if i < len(MerlinHandler._PREDICTION_LABELS)
                else f"risk_{i}"
            )
            labeled[name] = round(s, 4)
        return {"mode": "prediction", "risk_scores": labeled}

    @staticmethod
    def _predict_report(model: Any, volume: torch.Tensor) -> dict[str, Any]:  # noqa: ANN401
        """RadiologyReport mode — generate a radiology report from the volume."""
        prompt = "Findings and impression for this CT scan:\n###\n"
        with torch.inference_mode():
            texts = model.generate(
                volume,
                text_labels=[prompt],
                max_new_tokens=512,
                do_sample=False,
            )
        report_text = texts[0] if texts else ""
        return {"mode": "report", "report_text": report_text}
