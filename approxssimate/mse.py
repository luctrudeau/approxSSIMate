"""
ApproxSSIMate MSE utilities.

Provides functions to:
  - Store and load frame-based mean squared error (MSE) measurements
  - Support distortion statistics workflows for SSIM approximation

Copyright (c) 2026, Luc Trudeau and Maria G. Martini

This software is licensed under the BSD 2-Clause License.
See the LICENSE file in the project root for full license information.
"""

import json
import numpy as np
from pathlib import Path

MSE_FILE_VERSION = 1

def write_mse_file(path,
    *,
    name=None,
    reference_path=None,
    distorted_path=None,
    width=None,
    height=None,
    mse_values=None,
):
    payload = {
        "version": MSE_FILE_VERSION,
        "frame_count": len(mse_values),
        "mse": [float(x) for x in mse_values],
    }

    if name is not None:
        payload["name"] = name

    if reference_path is not None:
        payload["reference_path"] = reference_path

    if distorted_path is not None:
        payload["distorted_path"] = distorted_path

    if width is not None:
        payload["width"] = width

    if height is not None:
        payload["height"] = height

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def read_mse_file(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    version = payload.get("version")

    if version != MSE_FILE_VERSION:
        raise ValueError(
            f"Unsupported MSE file version: {version}"
        )

    if "mse" not in payload:
        raise ValueError(
            "Invalid MSE file: missing 'mse' field"
        )

    if len(payload["mse"]) != payload.get("frame_count", len(payload["mse"])):
        raise ValueError(
            "Invalid MSE file: frame_count does not match number of MSE values"
        )

    return payload

def compute_mse(ref_img, dist_img) -> float:
    ref = np.asarray(ref_img, dtype=np.float64)
    dist = np.asarray(dist_img, dtype=np.float64)

    if ref.shape != dist.shape:
        raise ValueError(
            f"Image shapes differ: reference {ref.shape}, distorted {dist.shape}"
        )

    diff = ref - dist
    return float(np.mean(diff * diff, dtype=np.float64))
