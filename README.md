# approxSSIMate

Lightweight models for approximating SSIM from global distortion signals.

`approxSSIMate` provides fast, reference-based models that approximate
SSIM using only:

- Global MSE (or PSNR)
- Local statistics of the reference image

The goal is to enable SSIM-like reasoning without computing full
window-based SSIM over both images.

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

## Installation

```bash
pip install approxssimate
```

## Command-Line Usage

approxSSIMate can be used directly from the command line after installation.

```bash
approxssimate <mode> reference.png distorted1.png [distorted2.png ...]
```

### Available modes

- reference: Full SSIM computation using scikit-image (baseline reference).
- local-mse: SSIM approximation using local MSE (requires both images).
- global-mse: SSIM approximation using only global MSE and reference statistics.
- global-mse-var: Variance-weighted redistribution of global MSE.
- global-mse-std: Standard-deviation-weighted redistribution of global MSE.

### Example
```bash
approxssimate global-mse-var ref.png dist_35.jpg dist_50.jpg dist_75.jpg
```
This computes SSIM approximations for multiple distorted images using the variance-based model.

### Notes

- Images are converted to grayscale internally.
- All distorted images must match the reference resolution.
- Designed for batch evaluation workflows (e.g., bitrate ladder construction).

## Sponsorship

approxSSIMate is an open research project focused on making perceptual quality evaluation faster and more practical for real-world encoding workflows.

If your organization benefits from faster SSIM estimation, large-scale encoding experiments, bitrate ladder construction, or convex-hull optimization workflows, consider sponsoring the project.

### Roadmap (Funding-Enabled Milestones)

#### Tier 1 — Native C Implementation

Develop a production-ready C implementation of the SSIM approximation models:
	•	Optimized for speed and low memory footprint
	•	Designed for production integration
	•	Architecture compatible with libVMAF’s SSIM implementation
	•	Potential upstream contribution to libVMAF
	•	Also available as a standalone CLI and embeddable library

#### Tier 2 — SIMD Optimizations

Architecture-specific acceleration layers:
	•	AVX2 / AVX-512 (x86 servers)
	•	NEON (ARM-based systems)