from __future__ import annotations

import numpy as np

from dualpose.capture.transforms import build_transform


def test_flip_horizontal() -> None:
    img = np.array(
        [
            [[0, 0, 0], [1, 1, 1]],
            [[2, 2, 2], [3, 3, 3]],
        ],
        dtype=np.uint8,
    )
    t = build_transform(rotate=0, flip_horizontal=True, flip_vertical=False)
    out = t(img)
    assert out[0, 0, 0] == 1
    assert out[0, 1, 0] == 0


def test_rotate_180() -> None:
    img = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    img = np.repeat(img, 3, axis=2)
    t = build_transform(rotate=180, flip_horizontal=False, flip_vertical=False)
    out = t(img)
    assert out[0, 0, 0] == 4
    assert out[1, 1, 0] == 1
