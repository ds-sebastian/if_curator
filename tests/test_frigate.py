import hashlib

import cv2
import numpy as np
import pytest

from if_curator import frigate
from if_curator.frigate import align, confidence, iou, preprocess, trimmed_mean


def test_trimmed_mean_matches_scipy_definition():
    values = np.random.default_rng(0).normal(size=(20, 512))
    cut = int(0.15 * 20)  # scipy trims int(proportion * n) from each end of every coordinate
    expected = np.sort(values, axis=0)[cut:-cut].mean(axis=0)
    assert np.allclose(trimmed_mean(values), expected)
    assert np.allclose(trimmed_mean(values[:6]), values[:6].mean(axis=0))  # too few to trim


def test_confidence_curve():
    assert confidence(0.3) == pytest.approx(0.5)
    assert round(confidence(0.41), 2) == 0.9
    assert confidence(0.0) < 0.01 < 0.99 < confidence(0.6)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_preprocess_keeps_bgr_order_and_pads():
    face = np.zeros((224, 112, 3), np.uint8)
    face[..., 0] = 255  # blue in BGR
    tensor = preprocess(face)
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor[0, 0, 56, 56] == pytest.approx(1.0) and tensor[0, 2, 56, 56] == pytest.approx(-1.0)
    assert tensor[0, 0, 56, 0] == pytest.approx(-1.0)  # padding left and right of the narrow face


def test_align_puts_eyes_where_frigate_does():
    """Eyes end up level, 30% of the width apart, centered at 35% of the height."""
    image = np.zeros((100, 100, 3), np.uint8)
    landmarks = np.zeros((68, 2))
    landmarks[36:42], landmarks[42:48] = (30, 40), (70, 30)  # image-left eye, image-right eye; tilted
    for x, y in ((30, 40), (70, 30)):
        cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
    aligned = align(image, landmarks.reshape(1, 68, 2))[..., 0].astype(float)
    ys, xs = np.nonzero(aligned > 100)
    left, right = xs < 50, xs >= 50
    assert np.average(xs[left], weights=aligned[ys, xs][left]) == pytest.approx(35, abs=1.5)
    assert np.average(xs[right], weights=aligned[ys, xs][right]) == pytest.approx(65, abs=1.5)
    assert np.average(ys, weights=aligned[ys, xs]) == pytest.approx(35, abs=1.5)


def test_fetch_model_verifies_checksum(tmp_path, monkeypatch):
    class Download:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield self.data

    monkeypatch.setattr(frigate.requests, "get", lambda *a, **k: Download(b"good"))
    monkeypatch.setitem(frigate.MODEL_HASHES, "model.onnx", hashlib.sha256(b"good").hexdigest())
    assert frigate.fetch_model(tmp_path, "model.onnx").read_bytes() == b"good"

    monkeypatch.setattr(frigate.requests, "get", lambda *a, **k: Download(b"evil"))
    with pytest.raises(RuntimeError, match="checksum"):
        frigate.fetch_model(tmp_path / "other", "model.onnx")
    assert not any((tmp_path / "other").iterdir())
