import hashlib
import json
from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image

from if_curator import config, immich_api
from if_curator.cli import execute_jobs
from if_curator.config import Config
from if_curator.faces import FaceCandidate
from if_curator.runs import RunWorkspace, person_directory


def job_with_faces(tmp_path, count=2):
    candidates = []
    for i in range(count):
        path = tmp_path / f"prepared{i}.jpg"
        Image.new("RGB", (120, 120), (120, 60, 80)).save(path)
        candidate = FaceCandidate(
            str(i),
            "p",
            str(i),
            prepared_path=path,
            image_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            selected=True,
        )
        candidates.append(candidate)
    return dict(
        person={"id": "p", "name": "../Same / Name"},
        config={"mode": "face"},
        candidates=candidates,
        selected_faces=candidates,
        assets=[],
        limit=count,
    )


def test_exports_identical_bytes_manifest_and_isolated_reruns(tmp_path):
    job = job_with_faces(tmp_path)
    one = execute_jobs([job])
    manifest = json.loads((one / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    for candidate in job["candidates"]:
        assert (one / candidate.output_path).read_bytes() == candidate.prepared_path.read_bytes()
    assert "API_KEY" not in manifest["configuration"]
    assert manifest["jobs"][0]["candidates"][0]["sha256"] == job["candidates"][0].image_hash
    two = execute_jobs([job_with_faces(tmp_path, 1)])
    assert one != two
    assert len(list(one.rglob("*.jpg"))) == 2
    assert len(list(two.rglob("*.jpg"))) == 1


def test_changed_artifact_cannot_publish(tmp_path):
    job = job_with_faces(tmp_path, 1)
    job["candidates"][0].prepared_path.write_bytes(b"modified")
    workspace = RunWorkspace(tmp_path / "runs")
    with pytest.raises(ValueError, match="changed"):
        execute_jobs([job], workspace)
    assert workspace.path.name.endswith(".incomplete")
    assert not workspace.destination.exists()
    assert json.loads((workspace.path / "manifest.json").read_text())["status"] == "failed"


def test_interrupted_export_remains_incomplete(tmp_path, monkeypatch):
    job = job_with_faces(tmp_path)
    workspace = RunWorkspace(tmp_path / "runs")
    monkeypatch.setattr(workspace, "export_faces", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        execute_jobs([job], workspace)
    assert workspace.path.exists() and not workspace.destination.exists()
    assert json.loads((workspace.path / "manifest.json").read_text())["status"] != "complete"


def test_person_directory_sanitized_and_unique():
    first = person_directory("../", "a")
    assert "/" not in first and first != person_directory("../", "b")
    assert person_directory("Same", "a") != person_directory("Same", "b")


def load_config(monkeypatch, tmp_path, **environment):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "not-existing.json")
    for name in Config.setting_names():
        monkeypatch.delenv(name, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, str(value))
    instance = object.__new__(type(Config))
    instance._load()
    return instance


def test_configuration_defaults_and_all_overrides(monkeypatch, tmp_path):
    instance = load_config(monkeypatch, tmp_path)
    assert instance.MIN_FACE_WIDTH == 100
    assert not instance.ENABLE_FACE_ALIGNMENT
    assert instance.FACE_MAX_IMAGES == 30
    overrides = {}
    for name in Config.setting_names():
        default = getattr(type(Config), name)
        if isinstance(default, bool):
            overrides[name] = str(not default).lower()
        elif isinstance(default, int):
            overrides[name] = 20
        elif isinstance(default, float):
            overrides[name] = 0.25
        else:
            overrides[name] = "custom-cache"
    instance = load_config(monkeypatch, tmp_path, **overrides)
    for name in overrides:
        expected = overrides[name]
        if isinstance(getattr(instance, name), bool):
            expected = expected == "true"
        assert getattr(instance, name) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"MIN_FACE_WIDTH": 0},
        {"FACE_MAX_IMAGES": -1},
        {"MIN_CONFIDENCE": 2},
        {"BLUR_THRESHOLD": "nan"},
        {"ENABLE_CACHE": "perhaps"},
        {"FACE_MARGIN": "bad"},
    ],
)
def test_invalid_configuration(monkeypatch, tmp_path, overrides):
    with pytest.raises(ValueError):
        load_config(monkeypatch, tmp_path, **overrides)


def response(data):
    result = Mock()
    result.content = data
    return result


def test_original_decode_failure_falls_back(monkeypatch, image):
    encoded = BytesIO()
    image.save(encoded, format="JPEG")
    get = Mock(side_effect=[response(b"unsupported"), response(encoded.getvalue())])
    monkeypatch.setattr(immich_api.requests, "get", get)
    decoded, source = immich_api.fetch_image_source("a")
    assert source == "preview" and decoded.size == image.size
    assert get.call_count == 2


def test_exif_orientation_and_rgb(monkeypatch):
    encoded = BytesIO()
    photo = Image.new("L", (100, 200), 100)
    exif = Image.Exif()
    exif[274] = 6
    photo.save(encoded, format="JPEG", exif=exif)
    monkeypatch.setattr(immich_api.requests, "get", lambda *a, **kw: response(encoded.getvalue()))
    decoded, source = immich_api.fetch_image_source("a")
    assert source == "original"
    assert decoded.size == (200, 100) and decoded.mode == "RGB"
    assert decoded.getexif().get(274, 1) == 1


def test_legacy_face_selection_requires_identity_and_staging():
    from if_curator.diversity import select_diverse_assets

    with pytest.raises(ValueError, match="person_id"):
        select_diverse_assets([{"id": "a"}], 30, "Person")


def test_object_selection_small_pool_unchanged():
    from if_curator.diversity import select_diverse_assets

    assets = [{"id": "a"}]
    assert select_diverse_assets(assets, 30, "Dog", entity_type="object") == assets


def test_face_strategy_presets_and_custom_time(monkeypatch):
    from if_curator import cli

    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **kw: "1")
    assert cli._get_strategy_choice(True, "face") == (30, "smart")
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **kw: "2")
    assert cli._get_strategy_choice(True, "face") == (5, "smart")
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **kw: "3")
    monkeypatch.setattr(cli.IntPrompt, "ask", Mock(side_effect=[-1, 7]))
    monkeypatch.setattr(cli.Confirm, "ask", lambda *a, **kw: False)
    assert cli._get_strategy_choice(True, "face") == (7, "time")


def test_failed_model_initialization_is_not_cached(monkeypatch):
    import sys
    from types import SimpleNamespace

    from if_curator import embeddings

    monkeypatch.setattr(embeddings, "_insightface_app", None)
    monkeypatch.setenv("FORCE_CPU", "true")
    instance = SimpleNamespace(prepare=Mock(side_effect=RuntimeError("prepare failed")))
    construct = Mock(return_value=instance)
    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    )
    monkeypatch.setitem(sys.modules, "insightface.app", SimpleNamespace(FaceAnalysis=construct))
    assert embeddings.get_insightface_app() is None
    assert embeddings.get_insightface_app() is None
    assert construct.call_count == 2
    assert construct.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
