"""Tests for the k-based SSIM command."""

import subprocess
import tempfile
from pathlib import Path
import sys

import numpy as np

from approxssimate.k import write_k_file
from approxssimate.mse import write_mse_file


def test_cli_ssim_pooled():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        k_path = tmp_path / "test.k"
        mse_path = tmp_path / "test.mse"

        write_k_file(
            k_path,
            source_path="reference.y4m",
            width=1920,
            height=1080,
            k_values=np.array([0.01, 0.02]),
        )

        write_mse_file(
            mse_path,
            reference_path="reference.y4m",
            distorted_path="distorted.y4m",
            width=1920,
            height=1080,
            mse_values=np.array([10.0, 20.0]),
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "approxssimate",
                "ssim",
                "-k", str(k_path),
                "--mse", str(mse_path),
                "--pooled",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert "0.775" in result.stdout
