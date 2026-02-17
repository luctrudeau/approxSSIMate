"""
ApproxSSIMate – SSIM approximation from global MSE.

Implements:
  - Local-MSE-based SSIM approximation
  - Variance-weighted global MSE approximation
  - Standard-deviation-weighted global MSE approximation

Related to the forthcoming paper:
"A Simple Relationship Between SSIM and PSNR for DCT-Based Compressed Images and Video:
 Modeling Local Error Statistics"

Copyright (c) 2026, Luc Trudeau and Maria G. Martini

This software is licensed under the BSD 2-Clause License.
See the LICENSE file in the project root for full license information.
"""

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim
from skimage.util import crop

def _check_inputs(ref_img, dist_imgs, win_size, data_range):
    if win_size % 2 == 0 or win_size < 3:
        raise ValueError("win_size must be an odd integer >= 3.")
    if data_range <= 0:
        raise ValueError("data_range must be > 0.")

    ref = np.asarray(ref_img, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported.")

    dists = [np.asarray(d, dtype=np.float64) for d in dist_imgs]
    for d in dists:
        if d.shape != ref.shape:
            raise ValueError("All distorted images must match ref_img shape.")
        if d.ndim != 2:
            raise ValueError("Only 2D grayscale images are supported.")

    return ref, dists

def ssim_reference(ref_img, dist_imgs, win_size=7, data_range=255.0):
    ref, dists = _check_inputs(ref_img, dist_imgs, win_size, data_range=data_range)
    out = []
    for d in dists:
        score = ssim(ref, d, win_size=win_size, data_range=data_range)
        out.append(float(score))
    return np.asarray(out, dtype=np.float64)

def _reference_statistics(ref, win_size, data_range):
    NP = win_size * win_size
    cov_norm = NP / (NP - 1)
    ux = uniform_filter(ref, size=win_size)
    uxx = uniform_filter(ref * ref, size=win_size)
    return np.maximum(cov_norm * (uxx - ux * ux), 0.0)

def ssim_local_mse(ref_img, dist_imgs, win_size=7, data_range=255.0):
    """
    Approximate SSIM using local MSE and reference-image statistics.

    This implementation follows the formulation described in:

        Martini, M. G.,
        "Measuring Objective Image and Video Quality:
        On the Relationship Between SSIM and PSNR for DCT-Based Compressed Images,"
        IEEE Transactions on Instrumentation and Measurement, 2025.

    Under the assumption that local means are preserved by compression,
    SSIM can be approximated as:

        SSIM ~= 1 - MSE_l / (2 * var_x + C2)

    where:
        - MSE_l is the local window-based mean squared error
        - var_x is the local variance of the reference image
        - C2 is the SSIM contrast stabilization constant

    This method requires computing local MSE between the reference and
    distorted images.
    """
    ref, dists = _check_inputs(ref_img, dist_imgs, win_size, data_range=data_range)
    vx = _reference_statistics(ref, win_size, data_range)
    C2 = (0.03 * data_range) ** 2
    B2 = vx + vx + C2
    pad = (win_size - 1) // 2

    diff = np.empty_like(ref)
    out = []
    for d in dists:
        np.subtract(ref, d, out=diff)
        np.square(diff, out=diff)
        uee = uniform_filter(diff, size=win_size)
        A2 = uee
        S = A2 / B2
        out.append(1 - crop(S, pad).mean(dtype=np.float64))
    
    return np.asarray(out, dtype=np.float64)

def ssim_global_mse(ref_img, dist_imgs, win_size=7, data_range=255.0):
    """
    Approximate SSIM using global MSE and reference-image statistics only.

    In this simplified model, the local MSE is replaced by the global
    image-level mean squared error (MSE_G), assuming uniform spatial
    distribution of distortion.

    The approximation becomes:

        SSIM ~= 1 - MSE_G / (2 * var_x + C2)

    where:
        - var_x is the local window-based variance of the reference image
        - C2 is the SSIM contrast stabilization constant

    This approach ignores local distortion structure and serves as a
    baseline when only global MSE is available.
    """
    ref, dists = _check_inputs(ref_img, dist_imgs, win_size, data_range=data_range)
    vx = _reference_statistics(ref, win_size, data_range)
    C2 = (0.03 * data_range) ** 2
    B2 = vx + vx + C2
    pad = (win_size - 1) // 2
    B2_inv = crop(1.0 / B2, pad).mean(dtype=np.float64)

    diff = np.empty_like(ref)
    out = []
    for d in dists:
        np.subtract(ref, d, out=diff)
        np.multiply(diff, diff, out=diff)
        A2 = diff.mean(dtype=np.float64)
        S = 1.0 - A2 * B2_inv
        out.append(S)

    return np.asarray(out, dtype=np.float64)

def ssim_global_mse_var(ref_img, dist_imgs, win_size=7, data_range=255.0, beta=0.46, eps=1e-6):
    """
    Approximate SSIM using variance-weighted redistribution of global MSE.

    In this formulation, the local MSE is estimated from the global MSE
    by redistributing distortion according to a sublinear function of
    the local variance:

        MSE_l ~= MSE_G * ((var_x + eps)**beta / E[(var_x + eps)**beta])

    where:
        - var_x is the local variance of the reference image
        - beta in [0, 1] is a shaping exponent
        - eps is a small constant for numerical stability
        - E[.] denotes spatial averaging

    The resulting SSIM approximation uses only:
        - Global MSE
        - Reference-image statistics

    This model corresponds to the variance-based approach described in
    the forthcoming work:

        "A Simple Relationship Between SSIM and PSNR for DCT-Based
         Compressed Images and Video: Modeling Local Error Statistics"
        Trudeau and Martini (2026).
    """
    ref, dists = _check_inputs(ref_img, dist_imgs, win_size, data_range=data_range)
    vx = _reference_statistics(ref, win_size, data_range)
    C2 = (0.03 * data_range) ** 2
    B2 = vx + vx + C2
    pad = (win_size - 1) // 2
    weights = (vx + eps) ** beta
    wmean = weights.mean(dtype=np.float64) + 1e-10
    inv = weights / (wmean * B2)
    k = crop(inv, pad).mean(dtype=np.float64)

    diff = np.empty_like(ref)
    out = []
    for d in dists:
        np.subtract(ref, d, out=diff)
        np.multiply(diff, diff, out=diff)
        mse = diff.mean(dtype=np.float64)
        out.append(1.0 - k * mse)
        
    return np.asarray(out, dtype=np.float64)

def ssim_global_mse_std(ref_img, dist_imgs, win_size=7, data_range=255.0, beta=0.46, eps=1e-6):
    """
Approximate SSIM using standard-deviation-weighted redistribution
    of global MSE.

    This formulation is derived from the sublinear variance model with
    beta approximately equal to 0.5, leading to a standard-deviation-
    based weighting:

        MSE_l ~= MSE_G * ((std_x + eps) / E[std_x + eps])

    where:
        - std_x is the local standard deviation of the reference image
        - eps is a small constant for numerical stability
        - E[.] denotes spatial averaging

    Empirical results indicate improved robustness under severe
    degradation compared to linear variance weighting.

    This method requires only global MSE and reference-image statistics.
    """
    ref, dists = _check_inputs(ref_img, dist_imgs, win_size, data_range=data_range)
    vx = _reference_statistics(ref, win_size, data_range)
    C2 = (0.03 * data_range) ** 2
    B2 = vx + vx + C2
    pad = (win_size - 1) // 2
    weights = np.sqrt(vx + eps)
    wmean = weights.mean(dtype=np.float64) + 1e-10
    inv = weights / (wmean * B2)
    k = crop(inv, pad).mean(dtype=np.float64)

    diff = np.empty_like(ref)
    out = []
    for d in dists:
        np.subtract(ref, d, out=diff)
        np.multiply(diff, diff, out=diff)
        mse = diff.mean(dtype=np.float64)
        out.append(1.0 - k * mse)
        
    return np.asarray(out, dtype=np.float64)
