from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from if_curator.config import Config
from if_curator.faces import FaceCandidate


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    for name in Config.setting_names():
        monkeypatch.setattr(Config, name, getattr(type(Config), name))
    monkeypatch.setattr(Config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Config, "OUTPUT_DIR", str(tmp_path / "exports"))


@pytest.fixture
def image():
    rng = np.random.default_rng(2)
    pixels = rng.integers(40, 220, (240, 320, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.minimum(pixels[:, :, 0].astype(int) + 30, 255)
    return Image.fromarray(pixels.astype(np.uint8))


@pytest.fixture
def metadata():
    return dict(
        id="face-a",
        imageWidth=320,
        imageHeight=240,
        boundingBoxX1=60,
        boundingBoxY1=40,
        boundingBoxX2=200,
        boundingBoxY2=200,
    )


@pytest.fixture
def candidate():
    return FaceCandidate("asset-a", "person-a", "face-a")


@pytest.fixture
def fake_app(monkeypatch):
    from if_curator import faces

    def detect(app, image, expected):
        from cv2 import COLOR_RGB2BGR, cvtColor

        return cvtColor(np.asarray(image), COLOR_RGB2BGR), SimpleNamespace(
            det_score=0.95,
            bbox=np.array(expected),
            kps=np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], dtype=np.float32),
        )

    embedding = np.arange(512, dtype=np.float32) + 1
    recognition = SimpleNamespace(get=lambda bgr, target: embedding.copy())
    app = SimpleNamespace(models={"recognition": recognition})
    monkeypatch.setattr(faces, "detect_target", detect)
    monkeypatch.setattr(faces, "get_insightface_app", lambda: app)
    monkeypatch.setattr(faces, "model_fingerprint", lambda app: "test-fingerprint")
    return app
