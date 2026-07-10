"""
approxssimate: fast SSIM approximations from global MSE and reference statistics.
"""

__version__ = "0.1.0"

from .core import (
    ssim_local_mse,
    ssim_global_mse,
    ssim_global_mse_var,
    ssim_global_mse_std,
)

from .k import (
    compute_k,
    approx_ssim_from_k_mse,
)

__all__ = [
    "ssim_local_mse",
    "ssim_global_mse",
    "ssim_global_mse_var",
    "ssim_global_mse_std",
]