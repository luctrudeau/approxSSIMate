"""
approxssimate: fast SSIM approximations from global MSE and reference statistics.
"""

__version__ = "0.2.0"

from .k import (
    compute_k,
    approx_ssim_from_k_mse,
)

__all__ = [
    "compute_k",
    "approx_ssim_from_k_mse",
]