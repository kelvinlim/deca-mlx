"""Image loading and 224x224 crop matching official DECA ``TestData``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import estimate_transform, warp


def _bbox2point(left, right, top, bottom, kind: str = "bbox"):
    if kind == "kpt68":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    else:
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    return old_size, center


def load_image(
    path: str | Path,
    iscrop: bool = False,
    crop_size: int = 224,
    scale: float = 1.25,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict:
    """Return an official-style sample dict with an NCHW float32 image in ``[0, 1]``.

    ``IMG_0392_inputs.jpg`` is already a cropped DECA input, so the demo default is
    ``iscrop=False`` (the same similarity warp official DECA uses in that mode).
    """
    image = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    h, w = image.shape[:2]
    if iscrop:
        if bbox is None:
            left, top, right, bottom = 0, 0, w - 1, h - 1
            kind = "bbox"
        else:
            left, top, right, bottom = bbox
            kind = "bbox"
        old_size, center = _bbox2point(left, right, top, bottom, kind)
        size = int(old_size * scale)
        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ]
        )
    else:
        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]], dtype=np.float64)
    dst_pts = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]], dtype=np.float64)
    tform = estimate_transform("similarity", src_pts, dst_pts)
    image = image / 255.0
    dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
    dst_nchw = dst_image.transpose(2, 0, 1).astype(np.float32)
    return {
        "image": dst_nchw,
        "imagename": Path(path).stem,
        "tform": tform.params.astype(np.float32),
        "original_image": image.transpose(2, 0, 1).astype(np.float32),
    }
