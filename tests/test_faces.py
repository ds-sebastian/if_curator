import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from if_curator import faces, immich_api
from if_curator.config import Config
from if_curator.faces import FaceCandidate, FacePipelineError
from if_curator.quality import assess_quality, check_grayscale


def test_target_metadata_not_first(metadata):
    asset = {
        "id": "group",
        "people": [{"id": "other", "faces": [dict(metadata, id="other-face")]}, {"id": "target", "faces": [metadata]}],
    }
    assert immich_api.resolve_face_metadata(asset, "target") is metadata


@pytest.mark.parametrize(
    "responses", [[], [{"person": {"id": "other"}}], [{"person": {"id": "target"}}, {"person": {"id": "target"}}]]
)
def test_missing_ambiguous_metadata(monkeypatch, responses):
    response = Mock()
    response.json.return_value = responses
    monkeypatch.setattr(immich_api.requests, "get", lambda *a, **kw: response)
    with pytest.raises(ValueError):
        immich_api.resolve_face_metadata({"id": "asset"}, "target")


def test_metadata_fallback_without_optional_fields(monkeypatch, metadata):
    response = Mock()
    response.json.return_value = [dict(metadata, person={"id": "target"})]
    monkeypatch.setattr(immich_api.requests, "get", lambda *a, **kw: response)
    assert immich_api.resolve_face_metadata({"id": "asset"}, "target")["id"] == metadata["id"]


def test_repeated_target_in_asset_rejected(metadata):
    with pytest.raises(ValueError, match="ambiguous"):
        immich_api.resolve_face_metadata({"people": [{"id": "p", "faces": [metadata, metadata]}]}, "p")


@pytest.mark.parametrize(
    "boxes, succeeds",
    [
        ([[0, 0, 30, 30, 0.99], [60, 40, 200, 200, 0.8]], True),
        ([[0, 0, 30, 30, 0.99]], False),
        ([[60, 40, 200, 200, 0.8], [61, 41, 201, 201, 0.9]], False),
    ],
)
def test_detection_matches_expected_not_largest(image, boxes, succeeds):
    landmarks = np.ones((len(boxes), 5, 2), dtype=np.float32)
    app = SimpleNamespace(det_model=SimpleNamespace(detect=lambda *a, **kw: (np.array(boxes), landmarks)))
    if succeeds:
        _, target = faces.detect_target(app, image, (60, 40, 200, 200))
        assert target.det_score == 0.8
    else:
        with pytest.raises(ValueError, match="ambiguous"):
            faces.detect_target(app, image, (60, 40, 200, 200))


def test_scaled_coordinates(metadata):
    assert faces.scaled_bbox(metadata, (160, 120)) == (30, 20, 100, 100)


@pytest.mark.parametrize(
    "change", [{"imageWidth": 0}, {"boundingBoxX1": -1}, {"boundingBoxX2": 900}, {"boundingBoxY1": float("nan")}]
)
def test_invalid_geometry(metadata, change):
    with pytest.raises(ValueError):
        faces.scaled_bbox(dict(metadata, **change), (320, 240))


def test_unverified_aspect_ratio(metadata):
    with pytest.raises(ValueError, match="coordinate_mismatch"):
        faces.scaled_bbox(metadata, (240, 320))


def test_balanced_color_not_grayscale():
    rgb = np.tile(np.array([[220, 80, 80], [80, 220, 80], [80, 80, 220]], dtype=np.uint8), (30, 10, 1))
    assert np.ptp(rgb.mean(axis=(0, 1))) == 0
    assert check_grayscale(rgb)[0]
    assert not check_grayscale(np.full((30, 30, 3), 100, dtype=np.uint8))[0]


def test_prepared_bytes_and_target_quality(tmp_path, image, metadata, candidate, fake_app):
    source = tmp_path / "source.png"
    image.save(source)
    faces._prepare_one(candidate, metadata, source, tmp_path, fake_app, "fp")
    assert not candidate.reasons
    assert candidate.embedding is not None
    assert candidate.effective_dimensions == (140, 160)
    assert candidate.image_hash == hashlib.sha256(candidate.prepared_path.read_bytes()).hexdigest()
    with Image.open(candidate.prepared_path) as encoded:
        assert encoded.size == (182, 208)


def test_blurry_dark_face_not_rescued_by_background(tmp_path, image, metadata, candidate, fake_app):
    image.paste((15, 8, 5), (60, 40, 200, 200))
    source = tmp_path / "source.png"
    image.save(source)
    faces._prepare_one(candidate, metadata, source, tmp_path, fake_app, "fp")
    assert any("Blurry" in reason for reason in candidate.reasons)
    assert any("Underexposed" in reason for reason in candidate.reasons)


@pytest.mark.parametrize("align", [True, False])
def test_minimum_size_before_margin_or_alignment(tmp_path, image, metadata, candidate, fake_app, monkeypatch, align):
    monkeypatch.setattr(Config, "ENABLE_FACE_ALIGNMENT", align)
    monkeypatch.setattr(Config, "FACE_MARGIN", 1.0)
    source = tmp_path / "source.png"
    image.resize((160, 120)).save(source)
    with pytest.raises(ValueError, match="face_too_small"):
        faces._prepare_one(candidate, metadata, source, tmp_path, fake_app, "fp")


def test_zero_confidence_is_rejected(image):
    result = assess_quality(image, confidence=0)
    assert not result.passed
    assert result.measurements["detection_confidence"] == 0


def test_cache_scoped_by_face_person_bytes_and_model(tmp_path, image, metadata, fake_app, monkeypatch):
    monkeypatch.setattr(Config, "ENABLE_CACHE", True)
    source = tmp_path / "source.png"
    image.save(source)
    embedder = Mock(return_value=np.ones(512))
    fake_app.models["recognition"].get = embedder
    for person, face, fp in [
        ("p", "f", "m"),
        ("p", "f", "m"),
        ("other", "f", "m"),
        ("p", "other", "m"),
        ("p", "f", "new-model"),
    ]:
        candidate = FaceCandidate("a", person, face)
        faces._prepare_one(candidate, metadata, source, tmp_path, fake_app, fp)
    assert embedder.call_count == 4
    image.paste((200, 50, 50), (70, 50, 80, 60))
    image.save(source)
    faces._prepare_one(FaceCandidate("a", "p", "f"), metadata, source, tmp_path, fake_app, "m")
    assert embedder.call_count == 5


def test_empty_and_unavailable_model(tmp_path, monkeypatch):
    assert faces.prepare_face_candidates([], "p", tmp_path) == ([], None)
    monkeypatch.setattr(faces, "get_insightface_app", lambda: None)
    with pytest.raises(FacePipelineError, match="unavailable"):
        faces.prepare_face_candidates([{"id": "a"}], "p", tmp_path)


def test_prepare_rejects_edits_and_reports_provenance(tmp_path, image, metadata, fake_app, monkeypatch):
    monkeypatch.setattr(faces, "resolve_face_metadata", lambda *a: metadata)
    monkeypatch.setattr(faces, "fetch_image_source", lambda *a: (image.copy(), "preview"))
    records, fp = faces.prepare_face_candidates([{"id": "good"}, {"id": "edited", "isEdited": True}], "p", tmp_path)
    assert fp == "test-fingerprint:frigate-test"
    good = next(c for c in records if c.asset_id == "good")
    assert good.source == "preview" and good.embedding is not None
    bad = next(c for c in records if c.asset_id == "edited")
    assert "unverified_edited_coordinates" in bad.reasons[0]
    assert not (tmp_path / "sources").exists()


def test_runtime_model_failure_does_not_fallback(tmp_path, image, metadata, fake_app, monkeypatch):
    monkeypatch.setattr(faces, "resolve_face_metadata", lambda *a: metadata)
    monkeypatch.setattr(faces, "fetch_image_source", lambda *a: (image.copy(), "original"))
    fake_app.models["recognition"].get = Mock(side_effect=RuntimeError("inference failed"))
    with pytest.raises(FacePipelineError):
        faces.prepare_face_candidates([{"id": "a"}], "p", tmp_path)


def test_aligned_output_uses_local_landmarks_and_encoded_quality(
    tmp_path, image, metadata, candidate, fake_app, monkeypatch
):
    import sys

    # Isolate the crop transform from optional model dependencies in offline tests.
    alignment_module = SimpleNamespace(
        estimate_norm=lambda kps, image_size: np.array([[0.5, 0, 0], [0, 0.5, 0]], dtype=np.float32)
    )
    monkeypatch.setitem(sys.modules, "insightface.utils.face_align", alignment_module)
    monkeypatch.setattr(Config, "ENABLE_FACE_ALIGNMENT", True)
    detector = Mock(wraps=faces.detect_target)
    monkeypatch.setattr(faces, "detect_target", detector)
    embedder = Mock(return_value=np.ones(512))
    fake_app.models["recognition"].get = embedder
    source = tmp_path / "source.png"
    image.save(source)
    faces._prepare_one(candidate, metadata, source, tmp_path, fake_app, "fp")
    assert detector.call_count == 1
    np.testing.assert_allclose(embedder.call_args.args[1].kps[0], [20, 25])
    assert candidate.effective_dimensions == (140, 160)
    with Image.open(candidate.prepared_path) as output:
        assert output.size == (112, 112)
    assert candidate.measurements["detection_confidence"] == 0.95


def test_detector_runtime_failure_is_not_a_rejection(image):
    app = SimpleNamespace(det_model=SimpleNamespace(detect=Mock(side_effect=ValueError("bad model input"))))
    with pytest.raises(FacePipelineError, match="detector failed"):
        faces.detect_target(app, image, (60, 40, 200, 200))


def test_embedding_value_error_is_not_a_quality_rejection(tmp_path, image, metadata, fake_app, monkeypatch):
    monkeypatch.setattr(faces, "resolve_face_metadata", lambda *a: metadata)
    monkeypatch.setattr(faces, "fetch_image_source", lambda *a: (image.copy(), "original"))
    fake_app.models["recognition"].get = Mock(side_effect=ValueError("bad model input"))
    with pytest.raises(FacePipelineError):
        faces.prepare_face_candidates([{"id": "a"}], "p", tmp_path)


def test_fingerprint_changes_with_weights(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"first-version")
    app = SimpleNamespace(models={"recognition": SimpleNamespace(model_file=model)})
    first = faces.model_fingerprint(app)
    model.write_bytes(b"second-version")
    assert first != faces.model_fingerprint(app)


def test_candidate_cap_audited_without_downloads(tmp_path, fake_app, monkeypatch):
    calls = []

    def no_metadata(asset, person_id):
        calls.append(asset["id"])
        raise ValueError("missing_target")

    monkeypatch.setattr(faces, "resolve_face_metadata", no_metadata)
    records, _ = faces.prepare_face_candidates([{"id": f"{i:04}"} for i in range(3002)], "p", tmp_path)
    assert len(calls) == 3000
    assert len(records) == 3002
    assert sum(c.reasons == ["candidate_pool_cap"] for c in records) == 2


def test_embedding_receives_decoded_export_pixels(tmp_path, image, metadata, candidate, fake_app):
    import cv2

    source = tmp_path / "source.png"
    image.save(source)
    embedding_call = Mock(return_value=np.ones(512))
    fake_app.models["recognition"].get = embedding_call
    faces._prepare_one(candidate, metadata, source, tmp_path, fake_app, "fp")
    with Image.open(candidate.prepared_path) as final:
        expected = cv2.cvtColor(np.asarray(final.convert("RGB")), cv2.COLOR_RGB2BGR)
    np.testing.assert_array_equal(embedding_call.call_args.args[0], expected)
