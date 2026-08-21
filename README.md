# approxSSIMate

approxSSIMate is a lightweight tool for fast SSIM approximation from global distortion measurements and reference statistics.

The tool models the relationship between SSIM and distortion statistics, enabling fast SSIM estimation without computing local SSIM windows. It is designed for image and video quality analysis workflows where many encoding points need to be evaluated efficiently.

Presented at QoMEX 2026: [poster PDF](docs/qomex2026-approxssimate-poster.pdf)

## Features

- Estimate SSIM from MSE and reference statistics.
- Support for images and videos through a common frame-based interface.
- Generate reusable reference statistics (`.k`) files.
- Generate reusable distortion statistics (`.mse`) files.
- Evaluate multiple distortion points for quality ladder analysis.
- Lightweight alternative for workflows requiring fast SSIM approximation.

## News

  - June 2026: Initial research-preview release, v0.1.0 “Cardiff”, prepared for QoMEX 2026.

## Installation

Install the current research-preview version from source:

```bash
git clone https://github.com/luctrudeau/approxSSIMate.git
cd approxSSIMate
pip install -e .
```

## Workflow

approxSSIMate uses a 3-step workflow:

### 1. Compute reference statistics (`.k`) file

```bash
approxssimate k reference.png -o reference.k
```

or for video

```bash
approxssimate k reference.mp4 -o reference.k
```

### 2. Compute Mean Square Error (`.mse`) file

```bash
approxssimate mse reference.png distorted.png -o distorted.mse
```

or

```bash
approxssimate mse reference.mp4 distorted.mp4 -o distorted.mse
```

### 3. Approximate SSIM (approxSSIMate)

```bash
approxssimate ssim -k reference.k -m distorted.mse
```

Multiple distortion points can be evaluated:

```bash
approxssimate ssim -k reference.k -m quality_95.mse quality_75.mse quality_55.mse
```

## Why?

Computing SSIM requires local window statistics from both
the reference and distorted content.

In many practical encoding scenarios:

- The reference content is fixed
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

- Image and video inputs are evaluated using luma samples.
- Reference and distorted inputs must have matching resolution and frame count.
- Designed for batch evaluation workflows (e.g., bitrate ladder construction).

### Python API

```python
from approxssimate import compute_k, approx_ssim_from_k_mse

k = compute_k(reference)
mse = np.mean((ref - dist) ** 2)
score = approx_ssim_from_k_mse(k, mse)
```

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