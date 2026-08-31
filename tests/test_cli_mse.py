"""Tests for the image MSE command."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pipal"
REFERENCE = FIXTURE_DIR / "ref.png"
EXPECTED_MSE_FILE = FIXTURE_DIR / "expected_mse.csv"


def load_expected_mses():
    with EXPECTED_MSE_FILE.open(newline="", encoding="utf-8") as file:
        return [
            (row["image"], float(row["mse"]))
            for row in csv.DictReader(file)
        ]


EXPECTED_MSES = load_expected_mses()


@pytest.mark.parametrize(
    ("image_name", "expected_mse"),
    EXPECTED_MSES,
    ids=[image_name for image_name, _ in EXPECTED_MSES],
)
def test_mse_command_matches_ffmpeg(tmp_path, image_name, expected_mse):
    distorted = FIXTURE_DIR / image_name
    mse_file = tmp_path / f"{Path(image_name).stem}.mse"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "approxssimate",
            "mse",
            str(REFERENCE),
            str(distorted),
            "--output",
            str(mse_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with mse_file.open(encoding="utf-8") as file:
        result = json.load(file)

    assert result["frame_count"] == 1
    assert result["width"] == 77
    assert result["height"] == 77
    assert result["reference_path"] == "ref.png"
    assert result["distorted_path"] == image_name
    assert result["mse"][0] == pytest.approx(expected_mse, abs=0.0051)
    