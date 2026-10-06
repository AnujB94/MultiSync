from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def build_transform(
    rotate: int,
    flip_horizontal: bool,
    flip_vertical: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    if rotate not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported rotate value: {rotate}. Use 0/90/180/270.")

    def _apply(frame: np.ndarray) -> np.ndarray:
        img = frame
        if rotate == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotate == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if flip_horizontal:
            img = cv2.flip(img, 1)
        if flip_vertical:
            img = cv2.flip(img, 0)
        return img

    return _apply
