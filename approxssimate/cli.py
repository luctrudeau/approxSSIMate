"""
ApproxSSIMate command-line interface.

Provides commands to:
  - Compute source calibration coefficient k from a reference image
  - Estimate SSIM from k and global MSE
  - Compute global MSE from reference and distorted images when needed

Copyright (c) 2026, Luc Trudeau and Maria G. Martini

This software is licensed under the BSD 2-Clause License.
See the LICENSE file in the project root for full license information.
"""

import argparse
from itertools import zip_longest
from pathlib import Path
import numpy as np
import time

from .io import iter_frames
from .k import compute_k, write_k_file, read_k_file, approx_ssim_from_k_mse
from .mse import compute_mse, write_mse_file, read_mse_file

def _cmd_k(args):
    t0 = time.perf_counter()

    k_values = []
    width = None
    height = None

    for frame in iter_frames(args.ref):
        if width is None:
            height, width = frame.shape

        k = compute_k(
            frame,
            win_size=args.win_size,
            data_range=args.data_range,
            beta=args.beta,
        )

        k_values.append(k)

    if not k_values:
        raise ValueError(f"No frames found in: {args.ref}")

    write_k_file(
        args.output,
        source_path=Path(args.ref).name,
        width=width,
        height=height,
        channel=args.channel,
        win_size=args.win_size,
        data_range=args.data_range,
        beta=args.beta,
        k_values=k_values,
    )

    t1 = time.perf_counter()

    print(f"Wrote k data to: {args.output}")
    print(f"Processed {len(k_values)} {width}x{height} frames(s) in {t1 - t0:.3f} seconds")

def _cmd_mse(args):
    t0 = time.perf_counter()

    mses = []

    width = None
    height = None

    ref_frames = iter_frames(args.ref)
    dist_frames = iter_frames(args.dist)

    for i, (ref_frame, dist_frame) in enumerate(zip_longest(ref_frames, dist_frames)):
        if ref_frame is None:
            raise ValueError(
                f"Distorted input has more frames than reference "
                f"(extra frame at index {i})"
            )

        if dist_frame is None:
            raise ValueError(
                f"Reference input has more frames than distorted "
                f"(missing frame at index {i})"
            )

        if width is None:
            height, width = ref_frame.shape

        mses.append(compute_mse(ref_frame, dist_frame))

    if not mses:
        raise ValueError(
            f"No frames decoded when comparing "
            f"{args.ref} and {args.dist}"
        )

    write_mse_file(
        args.output,
        name=args.name,
        reference_path=Path(args.ref).name,
        distorted_path=Path(args.dist).name,
        width=width,
        height=height,
        mse_values=mses,
    )

    t1 = time.perf_counter()

    print(f"Wrote MSE data to: {args.output}")
    print(f"Frames: {len(mses)}")
    print(f"Size: {width}x{height}")
    print(f"Mean MSE: {np.mean(mses):.6f}")
    print(f"Processed in {t1 - t0:.3f} seconds")

def _cmd_ssim(args):
    t0 = time.perf_counter()

    k_payload = read_k_file(args.k)
    k_values = np.asarray(k_payload["k"], dtype=np.float64)

    results = []

    for mse_path in args.mse:
        mse_payload = read_mse_file(mse_path)
        mse_values = np.asarray(mse_payload["mse"], dtype=np.float64)

        if k_values.size != 1 and k_values.shape != mse_values.shape:
            raise ValueError(
                f"k contains {k_values.size} values, "
                f"but {mse_path} contains {mse_values.size} MSE values."
            )

        scores = approx_ssim_from_k_mse(k_values, mse_values)

        results.append({
            "name": mse_payload.get(
                "name", Path(mse_path).stem,
            ),
            "frames": len(mse_values),
            "mse": np.mean(mse_values),
            "ssim": np.mean(scores),
        })
            
    t1 = time.perf_counter()

    name_width = max(len("Name"), max(len(result["name"]) for result in results))
    print(
        f"{'Name':<{name_width}}"
        f"{'Frames':>8}"
        f"{'Mean MSE':>15}"
        f"{'Mean ApproxSSIM':>18}"
    )

    for result in results:
        print(
            f"{result['name']:<{name_width}}"
            f"{result['frames']:>8}"
            f"{result['mse']:>15.6f}"
            f"{result['ssim']:>18.6f}"
        )
    print(
        f"\nProcessed {len(results)} MSE file(s) "
        f"in {t1 - t0:.3f} seconds"
    )

def main():
    parser = argparse.ArgumentParser(
        description="ApproxSSIMate: fast SSIM approximation from global MSE and reference statistics."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_k = subparsers.add_parser("k", help="Compute ApproxSSIMate source calibration k")
    p_k.add_argument("ref", help="Reference image or video (8bit)")
    p_k.add_argument("-o", "--output", required=True, help="Output .k file")
    p_k.add_argument("--win-size", type=int, default=7, help="SSIM window size (odd integer >= 3)")
    p_k.add_argument("--data-range", type=float, default=255.0, help="Sample data range")
    p_k.add_argument("--beta", type=float, default=0.5, help="Variance exponent")
    p_k.add_argument("--channel", default="luma", help="Channel used to compute k")

    p_mse = subparsers.add_parser(
        "mse", help="Compute frame MSE values between reference and distorted media"
    )
    p_mse.add_argument("ref", help="Reference image or video (8bit)")
    p_mse.add_argument("dist", help="Distorted image or video (8bit)")
    p_mse.add_argument("-o", "--output", required=True, help="Output .mse file")
    p_mse.add_argument("--name", help="Optional name for this distortion point")

    p_ssim = subparsers.add_parser(
        "ssim", help="Estimate SSIM from ApproxSSIMate reference and MSE files")
    p_ssim.add_argument("-k", required=True, help="Input .k reference statistics file")
    p_ssim.add_argument("-m", "--mse", nargs="+", required=True,
                        help="Input .mse distortion statistics file(s)")

    args = parser.parse_args()

    try:
        if args.cmd == "k":
            _cmd_k(args)
            return

        if args.cmd == "mse":
            _cmd_mse(args)
            return
    
        if args.cmd == "ssim":
            _cmd_ssim(args)
            return

    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

if __name__ == "__main__":
    main()
