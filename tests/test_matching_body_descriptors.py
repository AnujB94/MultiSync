from __future__ import annotations

import numpy as np

from dualpose.matching.body_descriptors import (
    extract_body_descriptor,
    hist_similarity,
    proportion_similarity,
)


def _kpts() -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        k[i] = np.array([100 + i * 3, 80 + i * 2, 1.0], dtype=np.float32)
    return k


def test_body_descriptor_extracts_upper_and_lower_histograms() -> None:
    img = np.zeros((240, 160, 3), dtype=np.uint8)
    img[:120, :, :] = (20, 40, 200)    # upper red-ish
    img[120:, :, :] = (20, 200, 40)    # lower green-ish
    bbox = np.array([20, 20, 140, 220], dtype=np.float32)

    d = extract_body_descriptor(img, bbox, _kpts())
    assert d.upper_hist is not None
    assert d.lower_hist is not None
    assert d.proportions is not None


def test_hist_and_proportion_similarity_ranges() -> None:
    img1 = np.zeros((200, 120, 3), dtype=np.uint8)
    img2 = np.zeros((200, 120, 3), dtype=np.uint8)
    img1[:100] = (40, 20, 200)
    img1[100:] = (40, 200, 20)
    img2[:100] = (45, 25, 195)
    img2[100:] = (45, 195, 25)

    bbox = np.array([10, 10, 110, 190], dtype=np.float32)
    d1 = extract_body_descriptor(img1, bbox, _kpts())
    d2 = extract_body_descriptor(img2, bbox, _kpts())

    hs = hist_similarity(d1.upper_hist, d2.upper_hist)
    assert hs is not None and 0.0 <= hs <= 1.0
    ps = proportion_similarity(d1.proportions, d2.proportions)
    assert ps is not None and 0.0 <= ps <= 1.0
