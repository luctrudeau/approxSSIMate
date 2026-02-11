# approxSSIMate

Lightweight models for approximating SSIM from simpler distortion metrics.

approxSSIMate provides small, fast models that estimate SSIM using cheaper
distortion signals such as PSNR or local MSE. The goal is to enable
SSIM-like reasoning without the computational cost of full SSIM evaluation.

## Features

- PSNR → SSIM approximation
- MSE → SSIM approximation
- Lightweight Python implementation
- Research-friendly and reproducible
