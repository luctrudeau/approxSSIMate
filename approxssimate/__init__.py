"""
approxssimate: fast SSIM approximations from global MSE and reference statistics.
"""

from .core import (
    ssim_local_mse,
    ssim_global_mse,
    ssim_global_mse_var,
    ssim_global_mse_std,
)

__all__ = [
    "ssim_local_mse",
    "ssim_global_mse",
    "ssim_global_mse_var",
    "ssim_global_mse_std",
]