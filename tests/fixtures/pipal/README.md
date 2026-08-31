# PIPAL Test Fixtures

These test fixtures are derived from the [PIPAL dataset](https://github.com/HaomingCai/PIPAL-dataset).

## Reference image

The reference fixture is derived from:

- Original filename: `A0013.bmp`
- Original dimensions: 288 × 288 pixels
- Fixture dimensions: 77 × 77 pixels
- Fixture format: PNG

## Distorted images

Five distorted versions of `A0013.bmp` were selected from the dataset. Each image was resized from 288 × 288 to 77 × 77 pixels and saved in PNG format.

## Modifications

The reference and distorted images were:

1. Resized from 288 × 288 to 77 × 77 pixels.
2. Converted from BMP to lossless PNG to reduce file size without introducing additional compression artifacts.

No cropping or other intentional image transformations were applied.

## Expected MSE values

The expected mean squared error values for the distorted images are stored in `expected_mse.csv`.

Each row associates a distorted fixture with its expected luma MSE:

```csv
image,mse
dist_01.png,94.92
```

The expected values are generated independently with FFmpeg. Because FFmpeg reports MSE to two decimal places, tests should compare these values with an appropriate numerical tolerance.

### Luma conversion

The expected MSE values are calculated from 8-bit, full-range BT.601 luma samples. The fixture-generation script applies the following FFmpeg filter to both the reference and distorted images:

```text
scale=in_range=full:out_range=full:out_color_matrix=bt601,
format=yuv444p,
extractplanes=y
```

The filter performs these operations:

- `in_range=full` treats the PNG RGB samples as full-range values.
- `out_range=full` produces full-range luma values from 0 through 255.
- `out_color_matrix=bt601` uses the BT.601 RGB-to-luma conversion matrix.
- `format=yuv444p` converts the image to an 8-bit planar YUV representation.
- `extractplanes=y` retains only the Y′, or luma, plane used for MSE.

Using an explicit matrix and range makes the fixture-generation process reproducible and avoids relying on FFmpeg’s automatically selected color-conversion defaults.

## Regenerating expected MSE values

The `generate_expected_mse.sh` script regenerates `expected_mse.csv` from `ref.png` and every file matching `dist_*.png` in this directory.

FFmpeg must be installed and available on `PATH`.

Run the script from this directory:

```bash
./generate_expected_mse.sh
```

The script can also be run from the repository root:

```bash
tests/fixtures/pipal/generate_expected_mse.sh
```

Regenerate `expected_mse.csv` whenever the reference image, distorted images, luma-conversion policy, or fixture-generation process changes. The generated values should not be edited manually.

## Purpose

This small subset is used to test the approxSSIMate image-processing workflow while keeping the repository and continuous-integration workload lightweight.

The fixtures and expected values exercise:

- PNG decoding
- RGB-to-luma conversion
- MSE calculation
- `.mse` file generation
- CLI behavior
- Agreement with an independent FFmpeg calculation

## Source and attribution

PIPAL dataset:

- Repository: [HaomingCai/PIPAL-dataset](https://github.com/HaomingCai/PIPAL-dataset)
- Reference image: `A0013.bmp`

Refer to the upstream PIPAL repository for dataset licensing, usage conditions, and citation information.