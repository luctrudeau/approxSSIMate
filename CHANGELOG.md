# Changelog

## v0.2.0 - Unreleased

### Breaking changes

- Removed the legacy SSIM computation and approximation APIs:
  - `ssim_reference`
  - `ssim_local_mse`
  - `ssim_global_mse`
  - `ssim_global_mse_var`
  - `ssim_global_mse_std`
- The public API is now focused on SSIM estimation from reference-image statistics using:
  - `compute_k`
  - `approx_ssim_from_k_mse`
- Users relying on the removed functions will need to migrate to the new `k`-based workflow.


## v0.1.0 “Cardiff” - 2026-06-28

Initial research-preview release prepared for QoMEX 2026.

### Added

- Source-side calibration workflow with `.k` files.
- `approxssimate k` command to compute source-dependent calibration scalars.
- `approxssimate ssim` command to estimate SSIM from MSE values or distorted images.
- Support for multiple MSE values and multiple distorted images.
- JSON `.k` calibration file format.
- Research comparison modes for SSIM approximation variants.
- QoMEX 2026 poster and citation metadata.