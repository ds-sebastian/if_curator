import json

import cv2
import numpy as np
import pytest
from conftest import FakeFaces, FakeImmich, photo

from if_curator.config import Settings
from if_curator.export import export, folder_names
from if_curator.selection import Candidate, Job


def selected_job(name, person_id, assets, object_class=None):
    job = Job({"id": person_id, "name": name}, 5, object_class)
    for asset in assets:
        face = asset["people"][0]["faces"][0]
        box = (face["boundingBoxX1"], face["boundingBoxY1"], face["boundingBoxX2"], face["boundingBoxY2"])
        frame = (face["imageWidth"], face["imageHeight"])
        job.candidates.append(Candidate(asset["id"], "2025-01-01", face_id=face["id"], box=box, frame=frame))
    job.candidates.append(Candidate("rejected", reason="blurry"))
    job.selected = job.candidates[:-1]
    return job


def test_folder_names_are_labels():
    jobs = [Job({"id": "aaaaaaaa1", "name": "Ann/Lee"}, 1), Job({"id": "b", "name": "Bo"}, 1)]
    jobs.append(Job({"id": "cccccccc2", "name": "Ann/Lee"}, 1))
    assert folder_names(jobs) == ["Ann_Lee (aaaaaaaa)", "Bo", "Ann_Lee (cccccccc)"]


def test_exports_frigate_face_crops_with_manifest(tmp_path):
    assets = [photo(i) for i in range(3)]
    immich = FakeImmich(assets, broken={("asset-001", True)})
    jobs = [selected_job("Ann", "p1", assets), selected_job("Rex", "p2", assets[:1], object_class="dog")]
    run = export(jobs, immich, FakeFaces(), Settings(OUTPUT_DIR=str(tmp_path / "out")))

    assert sorted(p.name for p in run.parent.iterdir()) == [run.name]  # no staging left behind
    face = cv2.imread(str(run / "Ann" / "000.webp"))
    assert face.shape == (400, 400, 3)  # the detected box, from the 2x original
    manifest = json.loads((run / "manifest.json").read_text())
    ann, rex = manifest["people"]
    assert [i["source"] for i in ann["images"]] == ["original", "preview", "original"]
    assert ann["rejections"] == {"blurry": 1} and ann["exported"] == 3 and "API_KEY" not in manifest["settings"]
    assert rex["images"][0]["file"] == "Rex/000.jpg" and (run / "Rex" / "000.jpg").exists()


def test_failed_export_leaves_nothing(tmp_path):
    class Broken(FakeFaces):
        def detect(self, image, target):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        export([selected_job("Ann", "p1", [photo(1)])], FakeImmich([photo(1)]), Broken(), Settings(OUTPUT_DIR="out"))
    assert list((tmp_path / "out").iterdir()) == []


def test_runs_never_overwrite(tmp_path, monkeypatch):
    job = selected_job("Ann", "p1", [photo(1)])
    settings = Settings(OUTPUT_DIR="out", USE_FULL_RESOLUTION=False)
    runs = {export([job], FakeImmich([photo(1)]), FakeFaces(), settings) for _ in range(3)}
    assert len(runs) == 3
    assert all(np.asarray(cv2.imread(str(r / "Ann" / "000.webp"))).shape == (200, 200, 3) for r in runs)
