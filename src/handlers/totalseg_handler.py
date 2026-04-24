"""EmbarkLab's handler class for TotalSegmentator CT segmentation."""

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from totalsegmentator.python_api import totalsegmentator

log = logging.getLogger(__name__)

ALLOWED_INPUT_ROOT = Path("/data/xnat/archive/AI-POC")


def _stamp() -> str:
    """UTC timestamp + 8-char UUID, collision-safe at sub-second resolution."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


class TotalSegmentatorHandler:
    """Runs TotalSegmentator on a DICOM series and emits one DICOM SEG file."""

    def __init__(self, config: dict[str, Any]):
        self.model_name = config["model_name"]
        self.task_default = config.get("task_default", "total")
        self.fast_default = bool(config.get("fast_default", True))
        self.output_root = Path(
            config.get(
                "output_root",
                "/data/xnat/archive/AI-POC/resources/output",
            )
        )

    def model_fn(self) -> None:
        # TotalSegmentator lazy-loads nnU-Net weights on the first call.
        return None

    def input_fn(self, request: dict[str, Any]) -> dict[str, Any]:
        if "dicom_dir" not in request:
            raise ValueError("dicom_dir is required")

        dicom_dir = Path(request["dicom_dir"]).resolve()
        if not dicom_dir.is_dir():
            raise ValueError(f"dicom_dir is not a directory: {dicom_dir}")
        if not dicom_dir.is_relative_to(ALLOWED_INPUT_ROOT):
            raise ValueError(
                f"dicom_dir must be under {ALLOWED_INPUT_ROOT}: {dicom_dir}"
            )

        roi_subset = request.get("roi_subset")
        if roi_subset is not None:
            if not isinstance(roi_subset, list) or not all(
                isinstance(x, str) for x in roi_subset
            ):
                raise ValueError("roi_subset must be a list of structure names")

        return {
            "dicom_dir": str(dicom_dir),
            "task": request.get("task", self.task_default),
            "fast": bool(request.get("fast", self.fast_default)),
            "roi_subset": roi_subset,
        }

    def predict_fn(self, parsed: dict[str, Any]) -> dict[str, Any]:
        output_dir = self.output_root / "totalseg" / _stamp()
        output_dir.mkdir(parents=True, exist_ok=False)

        log.info(
            "running totalsegmentator: input=%s output=%s task=%s fast=%s roi_subset=%s",
            parsed["dicom_dir"],
            output_dir,
            parsed["task"],
            parsed["fast"],
            parsed["roi_subset"],
        )

        t0 = time.time()
        totalsegmentator(
            input=parsed["dicom_dir"],
            output=str(output_dir),
            task=parsed["task"],
            fast=parsed["fast"],
            output_type="dicom",
            roi_subset=parsed["roi_subset"],
        )
        elapsed = time.time() - t0
        log.info("totalsegmentator completed in %.1fs", elapsed)

        return {
            "output_dir": str(output_dir),
            "elapsed_s": elapsed,
            "task": parsed["task"],
            "fast": parsed["fast"],
            "roi_subset": parsed["roi_subset"],
        }

    def output_fn(
        self, prediction: dict[str, Any], slice_info: dict | None = None
    ) -> dict[str, Any]:
        output_dir = Path(prediction["output_dir"])
        seg_path = output_dir / "segmentations.dcm"
        if not seg_path.is_file():
            contents = sorted(p.name for p in output_dir.iterdir())
            raise RuntimeError(
                f"expected SEG file not found: {seg_path} (contents: {contents})"
            )
        return {
            "seg_path": str(seg_path),
            "output_dir": str(output_dir),
            "task": prediction["task"],
            "fast": prediction["fast"],
            "roi_subset": prediction["roi_subset"],
            "elapsed_s": prediction["elapsed_s"],
        }
