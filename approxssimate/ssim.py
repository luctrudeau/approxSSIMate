import argparse
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time

def _load_image(path):
    return np.array(Image.open(path).convert("L"))

def compute_ssim(ref_img, dist_imgs, win_size=7):
    out = []
    for dist_img in dist_imgs:
        if dist_img.shape != ref_img.shape:
            raise ValueError("The shapes of the reference and distorted images must be the same.")

        score = ssim(
            ref_img,
            dist_img,
            win_size=win_size,
        )
        out.append(float(score))
    
    return np.asarray(out, dtype=np.float64)

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
    scores = compute_ssim(ref_img, dist_imgs)
    t1 = time.perf_counter()

    for path, score in zip(args.dist, scores):
        print(f"{path}\t{score:.6f}")
    elapsed = t1 - t0
    n = len(scores)
    sec_per_img = elapsed / n
    throughput = n / elapsed
    shape = ref_img.shape
    print(f"\nComputed SSIM for {n} {shape[0]}x{shape[1]} image(s) in {elapsed:.3f} seconds")
    print(f"Average time per image: {sec_per_img:.3f} seconds")
    print(f"Throughput: {throughput:.2f} images/second")

if __name__ == "__main__":
    main()