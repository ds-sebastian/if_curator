import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from test_centroid_selection import candidate, job, vector

from if_curator import camera, cli
from if_curator.camera import CameraSample, embed_samples, load_manifest
from if_curator.frigate import unit
from if_curator.runs import RunWorkspace
from if_curator.selection import evaluate, objective, select_jobs


def sample(i, angle=0, person="p", split="validation"):
    return CameraSample(str(i), person, split, str(i), Path("unused"), str(i), embedding=vector(angle))


def manifest(tmp_path, items):
    for i, item in enumerate(items):
        item.setdefault("id", str(i))
        item.setdefault("person_id", "p")
        item.setdefault("capture_group", str(i))
        item.setdefault("path", f"{i}.png")
        p = tmp_path / item["path"]
        if not p.exists():
            Image.new("RGB", (112, 112), (20 * i, 90, 130)).save(p)
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"schema_version": 1, "samples": items}))
    return path


@pytest.mark.parametrize(
    "field,value", [("capture_group", "event"), ("path", "same.png"), ("asset_id", "immich-asset")]
)
def test_cross_split_leakage_rejected(tmp_path, field, value):
    path = manifest(tmp_path, [{"split": split, field: value} for split in ["reference", "test"]])
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(path)


def test_labels_are_required_and_unknowns_explicit(tmp_path):
    path = manifest(tmp_path, [{"split": "test", "person_id": None}])
    assert load_manifest(path)[0].person_id is None
    document = json.loads(path.read_text())
    del document["samples"][0]["person_id"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="requires"):
        load_manifest(path)


def test_no_test_inference_during_reference_validation_loading(tmp_path, monkeypatch):
    path = manifest(tmp_path, [{"split": split} for split in ["reference", "validation", "test"]])
    samples = load_manifest(path)
    get = Mock(return_value=vector())
    monkeypatch.setattr(camera, "get_frigate_model", lambda: SimpleNamespace(get=get))
    embed_samples(samples, "reference")
    embed_samples(samples, "validation")
    assert get.call_count == 2
    assert samples[2].embedding is None
    # Uniform, blurry and grayscale crops are deliberately retained in evaluation.
    assert samples[1].embedding is not None


def test_changed_samples_rejected(tmp_path, monkeypatch):
    samples = load_manifest(manifest(tmp_path, [{"split": "test"}]))
    samples[0].path.write_bytes(b"changed")
    monkeypatch.setattr(camera, "get_frigate_model", Mock())
    with pytest.raises(ValueError, match="changed"):
        embed_samples(samples, "test")


def test_failed_crops_stay_in_metric_denominators(tmp_path, monkeypatch):
    samples = load_manifest(manifest(tmp_path, [{"split": "test"}, {"split": "test"}]))
    samples[0].embedding = vector()
    monkeypatch.setattr(
        camera, "get_frigate_model", lambda: SimpleNamespace(get=Mock(side_effect=ValueError("invalid_embedding")))
    )
    embed_samples(samples, "test")
    report = evaluate({"p": vector()}, samples)
    assert report["known_count"] == 2 and report["inference_failures"] == 1
    assert report["correct_recognition_rate"] == 0.5
    assert report["unknown_false_accept_rate"] is None


def test_validation_improves_held_out_objective_and_test_cannot_change_selection():
    def run(test_angle):
        j = job([candidate(0, -0.3), candidate(1), candidate(2, 0.3)], 2)
        samples = [sample(10, 0.3), sample(20, test_angle, split="test")]
        result = select_jobs([j], samples)
        return j, result

    j, result = run(-2)
    other, _ = run(2)
    assert [c.asset_id for c in j["selected_faces"]] == [c.asset_id for c in other["selected_faces"]]
    assert float(unit(result["centers"]["p"]) @ vector(0.3)) > float(
        unit(result["baseline_centers"]["p"]) @ vector(0.3)
    )
    assert j["selection_report"]["final_joint_objective"] < j["selection_report"]["initial_joint_objective"]


def test_reference_camera_overrides_immich_proxy():
    j = job([candidate(0, -0.3), candidate(1), candidate(2, 0.3)])
    select_jobs([j], [sample(10, 0.3, split="reference")])
    assert j["selection_report"]["reference_source"] == "camera_reference"
    assert [c.asset_id for c in j["selected_faces"]] == ["2"]


def test_unknown_false_accept_penalty_and_actual_metrics():
    unknown = sample(10, 0, person=None)
    reference = {"p": vector()}
    assert objective({"p": vector()}, reference, [unknown]) > objective({"p": vector()}, reference, [])
    report = evaluate({"p": vector()}, [unknown])
    assert report["unknown_false_accept_rate"] == 1
    assert report["correct_recognition_rate"] is None


def test_enrollment_test_overlap_rejected():
    c, s = candidate(0), sample(5, split="test")
    c.image_hash = s.sha256
    with pytest.raises(ValueError, match="overlaps"):
        select_jobs([job([c])], [s])


def test_finalize_reads_test_only_after_selection(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(cli, "embed_samples", lambda samples, split: events.append(split))
    real_select = cli.select_jobs

    def select(*args):
        events.append("select")
        return real_select(*args)

    monkeypatch.setattr(cli, "select_jobs", select)
    monkeypatch.setattr(cli, "get_frigate_model", lambda: SimpleNamespace(identity={"profile": "test"}))
    j = job([candidate(0)])
    j["person"]["name"] = "Person"
    workspace = RunWorkspace(tmp_path)
    cli.finalize_face_selection([j], workspace, [sample(3, split="test")])
    assert events == ["reference", "validation", "select", "test"]
    assert workspace.manifest["evaluation"]["test_used_for_selection"] is False
    assert workspace.manifest["jobs"][0]["selection_report"]["leave_one_out"]


def test_unknown_labeled_identity_does_not_silently_become_impostor(tmp_path):
    with pytest.raises(ValueError, match="must be queued"):
        cli.finalize_face_selection([job([candidate(0)])], RunWorkspace(tmp_path), [sample(3, person="other")])


def test_image_decode_failure_is_a_reported_attempt(tmp_path, monkeypatch):
    samples = load_manifest(manifest(tmp_path, [{"split": "test"}]))
    s = samples[0]
    s.path.write_bytes(b"unsupported")
    s.sha256 = hashlib.sha256(s.path.read_bytes()).hexdigest()
    monkeypatch.setattr(camera, "get_frigate_model", Mock())
    embed_samples(samples, "test")
    assert s.error == "decode_failed"
    assert evaluate({"p": vector()}, samples)["correct_recognition_rate"] == 0
