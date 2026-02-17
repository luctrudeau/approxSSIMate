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

import argparse
from PIL import Image
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim
from skimage.util import crop
import time

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

# This implementation is based on the SSIM approximation method described in:
#
#   Martini, M. G., “Measuring Objective Image and Video Quality:
#   On the Relationship Between SSIM and PSNR for DCT-Based Compressed Images,”
#   IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–13, 2025.
#   DOI: 10.1109/TIM.2025.3529045
#
# The implementation structure is adapted from the reference SSIM implementation
# in scikit-image (skimage.metrics.structural_similarity):
# https://scikit-image.org/
#
# The code has been modified to estimate SSIM from local MSE and local
# variances instead of computing the full SSIM formulation.
def ssim_local_mse(ref_img, dist_imgs, win_size=7, data_range=255.0):
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

def _load_image(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float64)

def main():
    parser = argparse.ArgumentParser(
        description="ApproxSSIMate: SSIM and SSIM approximations from reference-only statistics."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    def add_common_args(p):
        p.add_argument("ref", help="Reference image (8-bit)")
        p.add_argument("dist", nargs="+", help="Distorted image(s) (8-bit)")
        p.add_argument("--win-size", type=int, default=7, help="SSIM window size (odd integer >= 3)")

    p_ref = subparsers.add_parser("reference", help="Compute reference SSIM (skimage)")
    add_common_args(p_ref)

    p_loc = subparsers.add_parser("local-mse", help="Compute SSIM approximation using local MSE")
    add_common_args(p_loc)

    p_glo = subparsers.add_parser("global-mse", help="Compute SSIM approximation using global MSE")
    add_common_args(p_glo)

    p_glo_v = subparsers.add_parser("global-mse-var", help="Variance-weighted approximation of local MSE from global MSE")
    add_common_args(p_glo_v)

    p_glo_s = subparsers.add_parser("global-mse-std", help="Standard deviation-weighted approximation of local MSE from global MSE")
    add_common_args(p_glo_s)

    args = parser.parse_args()

    ref_img = _load_image(args.ref)
    dist_imgs = [_load_image(p) for p in args.dist]

    if args.cmd == "reference":
        fn = ssim_reference
    elif args.cmd == "local-mse":
        fn = ssim_local_mse
    elif args.cmd == "global-mse":
        fn = ssim_global_mse
    elif args.cmd == "global-mse-var":
        fn = ssim_global_mse_var
    elif args.cmd == "global-mse-std":
        fn = ssim_global_mse_std
    else:
        raise RuntimeError("Unknown command")

    t0 = time.perf_counter()
    scores = fn(ref_img, dist_imgs, win_size=args.win_size)
    t1 = time.perf_counter()

    for path, score in zip(args.dist, scores):
        print(f"{path}\t{score:.6f}")
    elapsed = t1 - t0
    n = len(scores)
    sec_per_img = elapsed / n
    throughput = n / elapsed
    h, w = ref_img.shape
    print(f"\nMethod: {args.cmd}")
    print(f"Processed {n} {h}x{w} image(s) in {elapsed:.3f} seconds")
    print(f"Average time per image: {sec_per_img:.6f} seconds")
    print(f"Throughput: {throughput:.2f} images/second")

if __name__ == "__main__":
    main()