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


def iter_frames(path):
    """
    Decode an image or video as a sequence of 8-bit luma frames.

    Still images yield one frame. Videos yield each decoded frame.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        container = av.open(path)
    except av.FFmpegError as error:
        raise ValueError(f"Unable to read media file: {path}") from error

    try:
        if not container.streams.video:
            raise ValueError(f"No video or image stream found in: {path}")

        stream = container.streams.video[0]
        decoded_frames = 0

        for frame in container.decode(stream):
            if any(
                component.bits > 8
                for component in frame.format.components
            ):
                raise ValueError(
                    f"Unsupported pixel format: {frame.format.name}. "
                    "Only 8-bit inputs are currently supported."
                )

            yield frame.to_ndarray(format="gray").astype(
                np.float64,
                copy=False,
            )
            decoded_frames += 1

        if decoded_frames == 0:
            raise ValueError(f"No frames decoded from: {path}")

    except av.FFmpegError as error:
        raise ValueError(f"Unable to decode media file: {path}") from error
    finally:
        container.close()
