import numpy as np
import pytest

from if_curator import config


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Run every test in an empty directory, away from real settings."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONNECTION_FILE", tmp_path / ".immich_config.json")


def textured(width, height, seed=0):
    """A sharp, colorful, mid-brightness image that passes every quality check."""
    rng = np.random.default_rng(seed)
    image = rng.integers(60, 200, (height, width, 3), dtype=np.uint8)
    image[:, :, 0] //= 2
    return image


def photo(index, person="p1", box=(100, 100, 300, 300), frame=(800, 600), **extra):
    face = dict(zip(("boundingBoxX1", "boundingBoxY1", "boundingBoxX2", "boundingBoxY2"), box))
    face |= {"id": f"face-{index}", "imageWidth": frame[0], "imageHeight": frame[1]}
    asset = {"id": f"asset-{index:03d}", "checksum": f"sum-{index}", "fileCreatedAt": f"2025-01-{1 + index % 28:02d}"}
    return asset | {"people": [{"id": person, "name": "P", "faces": [face]}]} | extra


class FakeImmich:
    """Serves `photo()` assets; every image is textured and matches its face box frame."""

    def __init__(self, assets, broken=()):
        self.assets, self.broken, self.downloads = assets, set(broken), []

    def photos(self, person_id, years):
        return [a for a in self.assets if any(p["id"] == person_id for p in a["people"])]

    def target_faces(self, asset, person_id):
        return [f for p in asset["people"] if p["id"] == person_id for f in p["faces"]]

    def image(self, asset_id, original=False):
        self.downloads.append((asset_id, original))
        if (asset_id, original) in self.broken:
            raise OSError("cannot decode")
        face = next(a for a in self.assets if a["id"] == asset_id)["people"][0]["faces"][0]
        scale = 2 if original else 1
        return textured(face["imageWidth"] * scale, face["imageHeight"] * scale, seed=len(self.downloads))


class FakeFaces:
    """Finds the target exactly and embeds each face as its person's direction plus noise."""

    device = "CPU"

    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.embedded = 0

    def detect(self, image, target):
        return tuple(int(v) for v in target)

    def embed(self, faces):
        self.embedded += len(faces)
        rng = np.random.default_rng(self.embedded)
        return [np.eye(512)[0] * 10 + rng.normal(0, 1, 512) for _ in faces]
