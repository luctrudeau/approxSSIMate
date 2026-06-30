# approxSSIMate

Lightweight models for approximating SSIM from global distortion signals.

Presented at QoMEX 2026: [poster PDF](docs/qomex2026-approxssimate-poster.pdf)

`approxSSIMate` provides fast, reference-based models that approximate SSIM using only:

- Global MSE (or PSNR)
- Local statistics of the reference image

The goal is to enable SSIM-like reasoning without computing full window-based SSIM over both images.

## News

  - June 2026: Initial research-preview release, v0.1.0 “Cardiff”, prepared for QoMEX 2026.

## Installation

Install the current research-preview version from source:

```bash
git clone https://github.com/luctrudeau/approxSSIMate.git
cd approxSSIMate
pip install -e .
```

## Quick start

approxSSIMate uses a two-step workflow:
  1. Compute a source-dependent calibration file (.k) from the reference image
```bash
approxssimate k MY_IMAGE.png -o MY_IMAGE.k
```
  2. Use the .k to estimate SSIM from MSE values or distorted images. You can specify either the MSE directly
```bash
approxssimate ssim -k MY_IMAGE.k --mse="12.58,28.17,41.52,53.66" 
```
Or instead you can specify the distorted image(s)
```bash
approxssimate ssim -k MY_IMAGE.k MY_IMAGE.png MY_DISTORTED_IMG_1.png MY_DISTORTED_IMG_2.png MY_DISTORTED_IMG_3.png
```

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

### Notes

- Images are converted to grayscale internally.
- All distorted images must match the reference resolution.
- Designed for batch evaluation workflows (e.g., bitrate ladder construction).

## Sponsorship

approxSSIMate is an open research project focused on making perceptual quality evaluation faster and more practical for real-world encoding workflows.

If your organization benefits from faster SSIM estimation, large-scale encoding experiments, bitrate ladder construction, or convex-hull optimization workflows, consider sponsoring the project.

### Roadmap (Possible sponsored milestones)

#### Tier 1 — Native C Implementation

Develop a production-ready C implementation of the SSIM approximation models:
 - Optimized for speed and low memory footprint
 - Designed for production integration
 - Architecture compatible with libVMAF’s SSIM implementation
 - Potential upstream contribution to libVMAF
 - Also available as a standalone CLI and embeddable library

#### Tier 2 — SIMD Optimizations

Architecture-specific acceleration layers:
 - AVX2 / AVX-512 (x86 servers)
 - NEON (ARM-based systems)

### Support & Collaboration
 - Sponsor via GitHub Sponsors [![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-%23EA4AAA?logo=github)](https://github.com/sponsors/luctrudeau)
 - Reach out directly to discuss collaboration or production integration