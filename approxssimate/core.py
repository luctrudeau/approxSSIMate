"""
ApproxSSIMate core utilities.

Provides common numerical helpers used by the SSIM approximation
workflows, including global mean squared error (MSE) computation.

Copyright (c) 2026, Luc Trudeau and Maria G. Martini

This software is licensed under the BSD 2-Clause License.
See the LICENSE file in the project root for full license information.
"""

import numpy as np

def compute_mse(ref_img, dist_img) -> float:
    ref = np.asarray(ref_img, dtype=np.float64)
    dist = np.asarray(dist_img, dtype=np.float64)

    if ref.shape != dist.shape:
        raise ValueError(
            f"Image shapes differ: reference {ref.shape}, distorted {dist.shape}"
        )

    diff = ref - dist
    return float(np.mean(diff * diff, dtype=np.float64))
