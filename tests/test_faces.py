import numpy as np
import pytest
from conftest import FakeFaces, FakeImmich, photo, textured

from if_curator import faces
from if_curator.config import Settings
from if_curator.faces import analyze_faces, crop_region, locate, measure, scale_box, verdict
from if_curator.selection import Job


def test_scale_box_rejects_other_aspect_ratios():
    assert scale_box((10, 20, 30, 40), (100, 50), (400, 200)) == (40, 80, 120, 160)
    with pytest.raises(ValueError):
        scale_box((10, 20, 30, 40), (100, 50), (200, 200))


def test_crop_region_clips_and_offsets_target():
    region, target = crop_region(np.zeros((100, 200, 3)), (10, 20, 50, 60), margin=0.5)
    assert region.shape == (80, 70, 3) and target == (10, 20, 50, 60)
    region, target = crop_region(np.zeros((100, 200, 3)), (100, 40, 140, 60), margin=0.5)
    assert region.shape == (40, 80, 3) and target == (20, 10, 60, 30)


@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda f: np.full_like(f, 128), "blurry"),
        (lambda f: f // 10, "too_dark"),
        (lambda f: np.clip(f.astype(int) + 200, 0, 255).astype(np.uint8), "too_bright"),
        (lambda f: np.repeat(f[..., :1], 3, axis=2), "grayscale"),
        (lambda f: f, None),
    ],
)
def test_quality_checks(change, reason):
    face = change(textured(150, 180))
    assert verdict({"detected": True, **measure(face)}, Settings()) == reason


def test_sharpness_is_measured_at_arcface_scale():
    """The same face must not look sharper or blurrier just because the photo is bigger."""
    face = textured(112, 112)
    large = np.kron(face, np.ones((4, 4, 1), np.uint8))
    assert measure(large)["sharpness"] == pytest.approx(measure(face)["sharpness"], rel=0.05)


def test_locate():
    immich = FakeImmich([])
    assert locate(immich, photo(1, box=(-5, 10, 150, 160)), "p1", 80).box == (0, 10, 150, 160)
    assert locate(immich, photo(1, box=(0, 0, 50, 200)), "p1", 80).reason == "too_small"
    assert locate(immich, photo(1, isEdited=True), "p1", 80).reason == "edited"
    assert locate(immich, photo(1), "someone else", 80).reason == "no_face_box"
    twice = photo(1)
    twice["people"][0]["faces"] *= 2
    assert locate(immich, twice, "p1", 80).reason == "several_faces"


def settings(tmp_path, **overrides):
    return Settings(CACHE_DIR=str(tmp_path / "cache"), **overrides)


def test_analysis_embeds_usable_faces_and_caches(tmp_path):
    immich = FakeImmich([photo(i) for i in range(5)] + [photo(9, box=(0, 0, 20, 20))], broken={("asset-004", False)})
    model = FakeFaces()
    job = Job({"id": "p1", "name": "P"}, 3)
    analyze_faces(immich, model, job, settings(tmp_path))
    reasons = {c.asset_id: c.reason for c in job.candidates}
    assert reasons == {
        **{f"asset-00{i}": None for i in range(4)},
        "asset-004": "download_failed",
        "asset-009": "too_small",
    }
    assert all(c.embedding.shape == (512,) for c in job.eligible) and model.embedded == 4

    immich.downloads.clear()
    again = Job({"id": "p1", "name": "P"}, 3)
    analyze_faces(immich, FakeFaces(), again, settings(tmp_path))
    assert immich.downloads == [("asset-004", False)]  # only the failed download is retried
    assert [c.reason for c in again.candidates] == [c.reason for c in job.candidates]

    stricter = Job({"id": "p1", "name": "P"}, 3)
    analyze_faces(immich, FakeFaces(), stricter, settings(tmp_path, BLUR_THRESHOLD=1e9))
    assert {c.reason for c in stricter.candidates} == {"blurry", "download_failed", "too_small"}


def test_undetected_faces_are_rejected(tmp_path):
    model = FakeFaces()
    model.detect = lambda image, target: None
    job = Job({"id": "p1", "name": "P"}, 3)
    analyze_faces(FakeImmich([photo(1)]), model, job, settings(tmp_path))
    assert job.candidates[0].reason == "no_face_detected" and model.embedded == 0


def test_large_libraries_are_sampled_through_time(tmp_path, monkeypatch):
    monkeypatch.setattr(faces, "MAX_CANDIDATES", 3)
    job = Job({"id": "p1", "name": "P"}, 3)
    analyze_faces(FakeImmich([photo(i) for i in range(9)]), FakeFaces(), job, settings(tmp_path))
    assert [c.asset_id for c in job.eligible] == ["asset-000", "asset-004", "asset-008"]
    assert sum(c.reason == "sample_limit" for c in job.candidates) == 6
