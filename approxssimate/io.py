"""
ApproxSSIMate input/output utilities.

Provides helpers to:
  - Load still images as grayscale NumPy arrays
  - Stream video frames one at a time using PyAV
  - Present images and videos through a common frame iterator

Copyright (c) 2026, Luc Trudeau and Maria G. Martini

This software is licensed under the BSD 2-Clause License.
See the LICENSE file in the project root for full license information.
"""

from pathlib import Path

import av
import numpy as np
from PIL import Image, UnidentifiedImageError


def _load_image(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float64)

def iter_frames(path):
    """
    Iterate over grayscale frames from an image or video file.

    Images yield one frame. Videos yield one frame at a time.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )


    # Try image first
    try:
        yield _load_image(path)
        return
    except UnidentifiedImageError:
        pass

    # Try video
    try:
        container = av.open(path)
    except av.FFmpegError as e:
        raise ValueError(f"Unable to read media file: {path}") from e

    try:
        for frame in container.decode(video=0):
            if frame.format.name.endswith(("10le", "10be", "12le", "12be", "16le", "16be")):
                raise ValueError(
                    f"Unsupported video pixel format: {frame.format.name}. "
                    "Only 8-bit inputs are currently supported."
                )

            y_plane = frame.planes[0]
            y = np.frombuffer(y_plane, dtype=np.uint8).reshape(frame.height, y_plane.line_size)[:, :frame.width]
            yield y
    finally:
        container.close()
