import hashlib
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PIL import Image
from scipy.stats import trim_mean

from if_curator import frigate
from if_curator.config import Config


@pytest.mark.parametrize("n", [1, 2, 5, 6, 7, 13, 14, 30])
def test_raw_class_mean_matches_stable_upstream(n):
    vectors = np.random.default_rng(n).normal(size=(n, 512)).astype(np.float32)
    vectors[0] *= 5
    actual = frigate.class_mean(vectors)
    np.testing.assert_array_equal(actual, trim_mean(vectors, 0.15, axis=0))
    assert not np.allclose(actual, trim_mean([frigate.unit(v) for v in vectors], 0.15, axis=0))


def test_frigate_ndarray_preprocessing_preserves_bgr_and_padding():
    image = np.full((80, 160, 3), (220, 100, 10), np.uint8)
    tensor = frigate.preprocess(image)
    assert tensor.shape == (1, 3, 112, 112) and tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 56, 56], np.array([220, 100, 10]) / 127.5 - 1, atol=1e-7)
    assert (tensor[:, :, :28] == -1).all()
    assert (tensor[:, :, 84:] == -1).all()


def test_resize_rounds_short_dimension_to_multiple_of_four():
    image = np.full((117, 200, 3), 150, np.uint8)
    tensor = frigate.preprocess(image)
    # 65.52 -> 64, leaving 24 rows of padding at either edge.
    assert (tensor[:, :, :24] == -1).all()
    assert (tensor[:, :, 24:88] > 0).all()
    assert (tensor[:, :, 88:] == -1).all()


@pytest.mark.parametrize("layout", [(68, 2), (1, 68, 2), (68, 1, 2)])
def test_alignment_matches_upstream_eye_geometry(layout):
    image = np.random.default_rng(0).integers(0, 255, (200, 180, 3), dtype=np.uint8)
    landmarks = np.zeros((68, 2))
    landmarks[42:48] = [130.9, 80.9]
    landmarks[36:42] = [60.9, 100.9]
    matrix = cv2.getRotationMatrix2D((95, 90), np.degrees(np.arctan2(20, -70)) - 180, 54 / np.sqrt(5300))
    matrix[0, 2] += 90 - 95
    matrix[1, 2] += 70 - 90
    expected = cv2.warpAffine(image, matrix, (180, 200), flags=cv2.INTER_CUBIC)
    np.testing.assert_array_equal(frigate.align(image, landmarks.reshape(layout)), expected)


def test_confidence_is_sigmoid_not_cosine_and_includes_blur(monkeypatch):
    assert frigate.confidence(0.3) == 0.5
    assert frigate.confidence(0.5) == 0.98
    assert frigate.confidence(0.5, 0.06) == 0.92
    assert frigate.confidence(-1, 0.06) == 0
    assert frigate.blur_reduction(np.ones((112, 112, 3), np.uint8)) == 0.06
    monkeypatch.setattr(Config, "FRIGATE_BLUR_CONFIDENCE_FILTER", False)
    assert frigate.blur_reduction(np.ones((112, 112, 3), np.uint8)) == 0


def test_model_checksum_is_enforced_without_loading(tmp_path, monkeypatch):
    (tmp_path / "test").write_bytes(b"bad")
    monkeypatch.setitem(frigate.MODEL_HASHES, "test", hashlib.sha256(b"good").hexdigest())
    with pytest.raises(ValueError, match="checksum"):
        frigate.ensure_model(tmp_path, "test")


def test_inference_uses_landmarks_and_keeps_raw_output():
    model = object.__new__(frigate.FrigateModel)
    landmarks = np.zeros((68, 2), dtype=np.float32)
    landmarks[42:48], landmarks[36:42] = [80, 45], [30, 45]
    model.landmarks = Mock()
    model.landmarks.fit.return_value = True, np.array([[landmarks]])
    model.session = Mock()
    model.session.get_inputs.return_value = [Mock(name="input")]
    model.session.get_inputs.return_value[0].name = "data"
    expected = np.arange(512, dtype=np.float32) + 1
    model.session.run.return_value = [np.array([expected])]
    pixels = np.asarray(Image.new("RGB", (112, 112), (200, 70, 40)))
    np.testing.assert_array_equal(model.get(pixels), expected)
    assert model.session.run.call_args.args[1]["data"].shape == (1, 3, 112, 112)
