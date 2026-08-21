# Changelog

## v0.2.0 - Unreleased

### Breaking changes

- Removed the legacy SSIM and SSIM approximation APIs:
  - `ssim_reference()`
  - `ssim_local_mse()`
  - `ssim_global_mse()`
  - `ssim_global_mse_var()`
  - `ssim_global_mse_std()`
- The public API is now focused on SSIM estimation from reference statistics using:
  - `compute_k()`
  - `approx_ssim_from_k_mse()`
- Users relying on the removed functions will need to migrate to the new `k`-based workflow.

### Added

- Added support for videos (using PyAV).
- Added the `.mse` file format for storing distortion statistics.
- Added `approxssimate mse` command to generate `.mse` files.
- Added support for processing multiple `.mse` files in a single `ssim` command, enabling quality ladder analysis.
- Added `write_mse_file()` and `read_mse_file()` to the public API.

### Changed

- `approxssimate ssim` command now estimates SSIM from `.k` and `.mse` files.
- `approx_ssim_from_k_mse()` now supports arrays as parameters
- Updated the CLI workflow to separate:
  - media decoding
  - distortion measurement
  - SSIM approximation


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