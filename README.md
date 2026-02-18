# approxSSIMate

Lightweight models for approximating SSIM from global distortion signals.

`approxSSIMate` provides fast, reference-based models that approximate
SSIM using only:

- Global MSE (or PSNR)
- Local statistics of the reference image

The goal is to enable SSIM-like reasoning without computing full
window-based SSIM over both images.

---

## Why?

Computing SSIM requires local window statistics from both
the reference and distorted image.

In many practical encoding scenarios:

- The reference image is fixed
- Multiple distorted versions are evaluated
- Only global distortion (MSE / PSNR) is available

Examples include:

- Bitrate ladder construction
- Multi-encoding experiments
- Convex-hull selection workflows
- Fast rate–distortion exploration

In such cases, recomputing full SSIM repeatedly can be expensive.

`approxSSIMate` provides fast approximations that reuse
reference-image statistics and operate from global MSE only.

---

## Implemented Models

### 1. Reference SSIM
Full SSIM using scikit-image (baseline comparison).

### 2. Local-MSE SSIM
SSIM approximation using local MSE as proposed in:

Maria G. Martini,
*Measuring Objective Image and Video Quality: On the Relationship Between SSIM and PSNR for DCT-Based Compressed Images*,
IEEE Transactions on Instrumentation and Measurement, 2025.

### 3. Global-MSE SSIM
SSIM estimated using global MSE and reference local variance.

### 4. Variance-Weighted Global MSE
Redistributes global MSE proportionally to local variance.

### 5. Standard-Deviation-Weighted Global MSE
A sublinear (beta ≈ 0.5) variant using standard deviation
for improved robustness under heavy distortion.

The variance and standard-deviation models are described in an upcoming paper.

---

## Installation

```bash
pip install approxssimate