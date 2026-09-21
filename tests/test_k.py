"""Tests for k-based SSIM approximation."""

import numpy as np

from approxssimate.k import approx_ssim_from_k_mse


def test_approx_ssim_from_k_mse():
    k = [0.01, 0.02]
    mse = [10.0, 20.0]

    np.testing.assert_allclose(
        approx_ssim_from_k_mse(k, mse),
        [0.9, 0.6],
    )

    np.testing.assert_allclose(
        approx_ssim_from_k_mse(k, mse, pooled=True),
        [0.775],
    )
