# approxSSIMate

### Fast SSIM approximation from source calibration and MSE

[Paper / preprint] · [Poster]

Lightweight models for approximating SSIM from global distortion signals.

![approxSSIMate calibration workflow](docs/assets/architecture.jpg)

`approxSSIMate` provides fast, reference-based models that approximate
SSIM using only:

## News

  **June 2026:** Initial research-preview release, **v0.1.0 “Cardiff”**, prepared for QoMEX 2026.

## Installation

### From PyPI

```bash
pip install approxssimate
```

### From source

```bash
git clone https://github.com/luctrudeau/approxSSIMate.git
cd approxSSIMate
pip install -e .
```

### Verify the installation

```bash
approxssimate --help
```

You should see the available command-line modes, including:

```text
approxssimate k
approxssimate ssim
```

## Quick start

`approxSSIMate` uses a two-step workflow: first compute a source-dependent calibration file from the reference image, then reuse it to estimate SSIM from MSE values or distorted images.

### 1. Compute the source calibration

```bash
approxssimate k kodim20.png -o kodim20.k
```

This creates a small JSON `.k` file containing the source-dependent calibration scalar.

### 2. Estimate SSIM from MSE

```bash
approxssimate ssim --k kodim20.k --mse 12.58 28.17 41.52 53.66
```

Comma-separated MSE values are also supported:

```bash
approxssimate ssim --k kodim20.k --mse=12.58,28.17,41.52,53.66
```

Example output:

```text
k: 0.0023887065372418942
source: kodim20.png
size: 768x512
beta: 0.5

MSE             ApproxSSIM
12.582530       0.969944
28.166527       0.932718
41.519559       0.900822
53.657033       0.871829

Processed 4 MSE value(s) in 0.001 seconds
```

### 3. Estimate SSIM from distorted images

```bash
approxssimate ssim --k kodim20.k kodim20.png kodim20-6.png kodim20-12.png kodim20-18.png
```

The first image is the reference. All remaining images are treated as distorted versions of the same source.

## Citation

If you use `approxSSIMate` in your research, please cite the associated QoMEX 2026 paper:

```bibtex
@inproceedings{trudeau2026estimating,
  title = {Estimating SSIM from MSE for DCT-Based Compressed Images via Modeling Local Error Statistics},
  author = {Trudeau, Luc and Martini, Maria G.},
  booktitle = {17th International Conference on Quality of Multimedia Experience (QoMEX 2026)},
  year = {2026}
}

## Sponsorship

approxSSIMate is an open research project focused on making perceptual quality evaluation faster and more practical for real-world encoding workflows.

If your organization benefits from faster SSIM estimation, large-scale encoding experiments, bitrate ladder construction, or convex-hull optimization workflows, consider sponsoring the project.

### Roadmap (Funding-Enabled Milestones)

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