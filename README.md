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

## Background

The SSIM approximation implemented in approxSSIMate builds on:

Maria G. Martini,
*Measuring Objective Image and Video Quality: On the Relationship Between SSIM and PSNR for DCT-Based Compressed Images*,
IEEE Transactions on Instrumentation and Measurement, 2025.
DOI: 10.1109/TIM.2025.3529045