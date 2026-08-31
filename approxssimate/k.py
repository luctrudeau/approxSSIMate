"""
ApproxSSIMate reference-statistics utilities.

Provides functions to:
  - Compute source calibration coefficients from reference images
  - Store and load frame-based calibration data
  - Estimate SSIM from calibration coefficients and MSE values

Copyright (c) 2026, Luc Trudeau and Maria G. Martini

This software is licensed under the BSD 2-Clause License.
See the LICENSE file in the project root for full license information.
"""

import json
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter
from skimage.util import crop

K_FILE_VERSION = 1
K_METRIC_NAME = "approxssimate_k"

def write_k_file(path,
    *,
    source_path,
    width,
    height,
    k_values,
    channel="luma",
    method="variance-beta",
    win_size=7,
    data_range=255.0,
    beta=0.5,):
    payload = {
        "version": K_FILE_VERSION,
        "metric": K_METRIC_NAME,
        "method": method,
        "source_path": source_path,
        "width": width,
        "height": height,
        "channel": channel,
        "win_size": win_size,
        "data_range": data_range,
        "beta": beta,
        "frame_count": len(k_values),
        "k": [float(k) for k in k_values],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def read_k_file(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    if payload.get("version") != K_FILE_VERSION:
        raise ValueError(
            f"Unsupported .k file version {payload.get('version')}; "
            f"expected {K_FILE_VERSION}"
        )

    if payload.get("metric") != K_METRIC_NAME:
        raise ValueError("Not an ApproxSSIMate k file")
    
    return payload

def compute_k(ref, win_size=7, data_range=255.0, beta=0.5, eps=1e-6):
    ref = np.asarray(ref, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported.")
    
    NP = win_size * win_size
    cov_norm = NP / (NP - 1)

    ux = uniform_filter(ref, size=win_size)
    uxx = uniform_filter(ref * ref, size=win_size)
    vx = np.maximum(cov_norm * (uxx - ux * ux), 0.0)

    C2 = (0.03 * data_range) ** 2
    B2 = vx + vx + C2

    pad = (win_size - 1) // 2

    weights = (vx + eps) ** beta
    wmean = weights.mean(dtype=np.float64) + 1e-10

    inv = weights / (wmean * B2)

    return crop(inv, pad).mean(dtype=np.float64)

def approx_ssim_from_k_mse(k: ArrayLike, mse: ArrayLike) -> np.ndarray:
    """
    Estimate SSIM from the reference statistic k and MSE.

    Supports scalar and array inputs through NumPy broadcasting.
    Inputs are converted to float64 and the output is clipped to
    the valid SSIM range [0, 1].
    """
    k = np.asarray(k, dtype=np.float64)
    mse = np.asarray(mse, dtype=np.float64)
    return np.clip(1.0 - k * mse, 0.0, 1.0)
