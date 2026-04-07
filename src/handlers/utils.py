"""Shared utilities for model handlers."""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from PIL import Image


def decode_base64_image(data: str) -> Image.Image:
    """Decode a base64-encoded string into a PIL Image.

    Args:
        data: Base64-encoded image bytes.

    Returns:
        PIL Image in RGB mode.
    """
    image_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def serialize_json_response(data: dict[str, Any]) -> str:
    """Serialize a dictionary to a JSON string.

    Args:
        data: Response dictionary.

    Returns:
        JSON-encoded string.
    """
    return json.dumps(data)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an S3 URI into bucket and prefix.

    Args:
        uri: S3 URI (e.g. ``s3://bucket/path/to/prefix/``).

    Returns:
        Tuple of (bucket, prefix). Trailing slash is stripped from prefix.

    Raises:
        ValueError: If the URI does not match the expected format.
    """
    match = re.match(r"^s3://([^/]+)/(.+?)/?$", uri)
    if not match:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return match.group(1), match.group(2)
