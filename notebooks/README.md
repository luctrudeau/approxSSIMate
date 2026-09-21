# ApproxSSIMate Notebooks

This directory contains reproducible experiments, examples, and supporting material for ApproxSSIMate.

The notebooks complement the papers and command-line examples in the main repository by showing the complete workflow: preparing data, computing reference statistics, measuring distortion, estimating SSIM, and reproducing published results.

## Foundations

| Notebook | Description |
| --- | --- |
| [`01_ssim_psnr_foundations.ipynb`](01_ssim_psnr_foundations.ipynb) | Early exploratory notebook reproducing Maria G. Martini's observations on the relationship between SSIM and PSNR/MSE. These experiments helped motivate the development of ApproxSSIMate. |

## Published experiments

| Notebook | Description |
| --- | --- |
| [`qomex2026_results.ipynb`](qomex2026_results.ipynb) | Reproduces the Kodak image experiments from the QoMEX 2026 paper *Estimating SSIM from MSE for DCT-Based Compressed Images via Modeling Local Error Statistics*. |
| [`ibc2026_results.ipynb`](ibc2026_results.ipynb) | Reproduces the video coding experiments from the IBC 2026 paper *Scalable SSIM Estimation from PSNR for Per-Title and Context-Adaptive Encoding Workflows*. |
| [`mmsp2026_live_results.ipynb`](mmsp2026_live_results.ipynb) | Reproduces the LIVE image-quality experiments from the MMSP 2026 paper *ApproxSSIMate: Fast SSIM Estimation from MSE for Image and Video Coding*. |
| [`mmsp2026_vtm_results.ipynb`](mmsp2026_vtm_results.ipynb) | Reproduces the VTM 24 / Objective-1-fast video coding experiments from the MMSP 2026 paper. |

## Running the notebooks

Install ApproxSSIMate and its development dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .