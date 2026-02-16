import argparse
from PIL import Image
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.util import crop
import time

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
def mse2ssim(ref_img, dist_imgs, win_size=7):
    if win_size % 2 == 0 or win_size < 3:
        raise ValueError("win_size must be an odd integer >= 3.")
    pad = (win_size - 1) // 2
    NP = win_size * win_size
    cov_norm = NP / (NP - 1)
    C2 = (0.03 * 255) ** 2

    ref_img = ref_img.astype(np.float64, copy=False)
    ux = uniform_filter(ref_img, size=win_size)
    uxx = uniform_filter(ref_img * ref_img, size=win_size)
    vx = cov_norm * (uxx - ux * ux)

    out = []
    for dist_img in dist_imgs:
        if dist_img.shape != ref_img.shape:
            raise ValueError("The shapes of the reference and distorted images must be the same.")

        dist_img = dist_img.astype(np.float64, copy=False)
        uy = uniform_filter(dist_img, size=win_size)
        uyy = uniform_filter(dist_img * dist_img, size=win_size)
        uee = uniform_filter((ref_img - dist_img) ** 2, size=win_size)
        vy = cov_norm * (uyy - uy * uy)

        A2 = uee
        B2 = vx + vy + C2
        S = A2 / B2
        out.append(1 - crop(S, pad).mean(dtype=np.float64))
    
    return np.asarray(out, dtype=np.float64)

def _load_image(path):
    return np.array(Image.open(path).convert("L"))

def main():
    parser = argparse.ArgumentParser(
        description="Approximate SSIM using local MSE."
    )
    parser.add_argument("ref", help="Reference image")
    parser.add_argument("dist", nargs="+", help="Distorted image(s)")
    args = parser.parse_args()

    ref_img = _load_image(args.ref)
    dist_imgs = [_load_image(p) for p in args.dist]

    t0 = time.perf_counter()
    scores = mse2ssim(ref_img, dist_imgs)
    t1 = time.perf_counter()

    for path, score in zip(args.dist, scores):
        print(f"{path}\t{score:.6f}")
    elapsed = t1 - t0
    n = len(scores)
    sec_per_img = elapsed / n
    throughput = n / elapsed
    shape = ref_img.shape
    print(f"\nApproxSSIMating from local MSE for {n} {shape[0]}x{shape[1]} image(s) in {elapsed:.3f} seconds")
    print(f"Average time per image: {sec_per_img:.3f} seconds")
    print(f"Throughput: {throughput:.2f} images/second")

if __name__ == "__main__":
    main()