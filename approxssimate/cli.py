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
from .core import ssim_reference, ssim_local_mse, ssim_global_mse, ssim_global_mse_var, ssim_global_mse_std, compute_mse
from .k import compute_k, write_k_file, read_k_file, approx_ssim_from_k_mse
from pathlib import Path
from PIL import Image

import numpy as np
import time

def _load_image(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float64)

def _cmd_k(args):
    t0 = time.perf_counter()

    ref_img = _load_image(args.ref)
    h, w = ref_img.shape

    k = compute_k(
        ref_img,
        win_size=args.win_size,
        data_range=args.data_range,
        beta=args.beta,
    )

    write_k_file(
        args.output,
        source_path=Path(args.ref).name,
        width=w,
        height=h,
        channel=args.channel,
        win_size=args.win_size,
        data_range=args.data_range,
        beta=args.beta,
        k_values=[k],
    )

    t1 = time.perf_counter()

    print(f"Wrote k data to: {args.output}")
    print(f"k: {k:.17g}")
    print(f"Processed 1 {w}x{h} image in {t1 - t0:.3f} seconds")

def _cmd_ssim(args):
    payload = read_k_file(args.k)

    k_values = payload["k"]
    if len(k_values) != 1:
        raise ValueError(
            "The ssim command currently supports only .k files with a single k value."
        )

    k = float(k_values[0])

    if args.mse is not None:
        if args.images:
            raise ValueError("Use either --mse or an image pair, not both.")

        mse = float(args.mse)
    else:
        if len(args.images) != 2:
            raise ValueError("ssim requires either --mse or an image pair: ref dist")

        ref_path, dist_path = args.images

        ref_img = _load_image(ref_path)
        dist_img = _load_image(dist_path)

        t0 = time.perf_counter()
        mse = compute_mse(ref_img, dist_img)
        t1 = time.perf_counter()
        
        h, w = ref_img.shape
        print(f"Processed 1 {w}x{h} image in {t1 - t0:.3f} seconds")
    
    score = approx_ssim_from_k_mse(k, mse)

    print(f"ApproxSSIM: {score:.6f}")
    print(f"k: {k:.17g}")
    print(f"MSE: {mse:.6f}")

def main():
    parser = argparse.ArgumentParser(
        description="ApproxSSIMate: SSIM and SSIM approximations from reference-only statistics."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_k = subparsers.add_parser("k", help="Compute ApproxSSIMate source calibration k")
    p_k.add_argument("ref", help="Reference image (8-bit)")
    p_k.add_argument("-o", "--output", required=True, help="Output .k file")
    p_k.add_argument("--win-size", type=int, default=7, help="SSIM window size (odd integer >= 3)")
    p_k.add_argument("--data-range", type=float, default=255.0, help="Sample data range")
    p_k.add_argument("--beta", type=float, default=0.5, help="Variance exponent")
    p_k.add_argument("--channel", default="luma", help="Channel used to compute k")

    p_ssim = subparsers.add_parser("ssim", help="Estimate SSIM from an ApproxSSIMate .k file")
    p_ssim.add_argument("-k", required=True, help="Input .k calibration file")
    p_ssim.add_argument("--mse", type=float, help="Known global MSE")
    p_ssim.add_argument("images", nargs="*", help="Optional image pair: ref dist")

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

    if args.cmd == "k":
        _cmd_k(args)
        return
    
    if args.cmd == "ssim":
        _cmd_ssim(args)
        return

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