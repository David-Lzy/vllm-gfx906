# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for the gfx906 release baseline asset generator."""

import base64
import struct
import zlib
from dataclasses import replace

from benchmarks.gfx906.production_baseline import SCENARIOS, content_for


def _scenario(name: str):
    return next(scenario for scenario in SCENARIOS if scenario.name == name)


def _decoded_pixels(data_url: str) -> bytes:
    """Return unfiltered RGB scanlines from one generated PNG data URL."""
    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/png;base64"
    payload = base64.b64decode(encoded)
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")

    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        data = payload[data_start:data_end]
        if kind == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            compressed.extend(data)
        offset = data_end + 4

    raw = zlib.decompress(compressed)
    row_size = width * 3 + 1
    assert len(raw) == height * row_size
    assert raw[::row_size] == b"\x00" * height
    return raw


def _first_image_pixels(name: str, request_id: int) -> bytes:
    scenario = replace(_scenario(name), image_size=64)
    content = content_for(scenario, request_id)
    return _decoded_pixels(content[0]["image_url"]["url"])


def test_unique_grid_requests_change_decoded_pixels() -> None:
    """Unique grid inputs must not become cache-equivalent after decoding."""
    samples = [
        _first_image_pixels("grid4096_unique_c1", request_id) for request_id in range(4)
    ]

    assert len({sample for sample in samples}) == len(samples)


def test_reuse_grid_requests_keep_decoded_pixels() -> None:
    """Reuse grid inputs must intentionally retain the same decoded pixels."""
    first = _first_image_pixels("grid4096_reuse_c4", 0)
    later = _first_image_pixels("grid4096_reuse_c4", 3)

    assert first == later
