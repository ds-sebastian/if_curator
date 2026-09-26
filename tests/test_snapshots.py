import json

import cv2
import numpy as np
from conftest import FakeFaces, FakeImmich, textured

from if_curator.config import Settings
from if_curator.export import export
from if_curator.selection import Candidate, Job, unit
from if_curator.snapshots import analyze_snapshots, label_snapshots, parse_name


def direction(*weights):
    vector = np.zeros(512)
    vector[: len(weights)] = weights
    return vector


def person(name, center, count=10):
    job = Job({"id": name, "name": name}, count)
    job.center = unit(center)
    return job


def snapshot(event, embedding, timestamp="1758844800.5"):
    return Candidate(f"{event}-{timestamp}-unknown-0.62.webp", embedding=np.asarray(embedding, float))


def test_parse_name():
    assert parse_name("1758844800.1-ab12cd-1758844801.25-unknown-0.62.webp") == (
        "1758844800.1-ab12cd",
        "2025-09-26T00:00:01+00:00",
    )
    assert parse_name("odd.webp") == ("odd.webp", "")


def test_labels_only_clear_matches_and_one_per_event():
    ann, bo = person("ann", direction(1, 0)), person("bo", direction(0, 1))
    shots = [
        snapshot("1.0-a", direction(1, 0.2)),  # clearly Ann
        snapshot("1.0-a", direction(1, 0.1), timestamp="1758844801.0"),  # same event, even clearer
        snapshot("2.0-b", direction(1, 0.95)),  # between the two
        snapshot("3.0-c", direction(0, 0, 1)),  # a stranger
        snapshot("4.0-d", direction(0.1, 1)),  # Bo
    ]
    label_snapshots(shots, [ann, bo], recognition_threshold=0.9)
    assert [c.reason for c in shots] == ["same_event", None, "ambiguous", "no_match", None]
    assert ann.snapshots == [shots[1]] and bo.snapshots == [shots[4]]
    assert shots[1].measures["cosine"] > 0.99


def test_labeled_snapshots_are_spread_and_capped():
    ann = person("ann", direction(1, 0), count=2)
    shots = [snapshot(f"{i}.0-x", direction(1, 0.3 * (i - 2))) for i in range(5)]
    label_snapshots(shots, [ann], recognition_threshold=0.9)
    assert len(ann.snapshots) == 2 and {c.asset_id[:3] for c in ann.snapshots} >= {"2.0"}


class FakeFrigate:
    def __init__(self, images):
        self.images = images

    def snapshots(self):
        return list(self.images)

    def snapshot(self, name):
        if self.images[name] is None:
            import requests

            raise requests.ConnectionError
        return cv2.imencode(".webp", self.images[name])[1].tobytes()


def test_snapshots_are_quality_checked_and_embedded():
    frigate = FakeFrigate({"good.webp": textured(90, 110), "dark.webp": textured(90, 110) // 8, "gone.webp": None})
    shots = analyze_snapshots(frigate, FakeFaces(), Settings())
    assert {c.asset_id: c.reason for c in shots} == {
        "good.webp": None,
        "dark.webp": "too_dark",
        "gone.webp": "download_failed",
    }
    assert shots[0].embedding.shape == (512,) and shots[0].data


def test_labeled_snapshots_are_exported_as_is(tmp_path):
    job = Job({"id": "p1", "name": "Ann"}, 3)
    data = cv2.imencode(".webp", textured(60, 70))[1].tobytes()
    job.snapshots = [Candidate("1.0-a-2.0-unknown-0.5.webp", taken="t", data=data, measures={"cosine": 0.6})]
    run = export([job], FakeImmich([]), FakeFaces(), Settings(OUTPUT_DIR=str(tmp_path / "out")))
    assert (run / "Ann" / "000.webp").read_bytes() == data
    person = json.loads((run / "manifest.json").read_text())["people"][0]
    assert person["frigate_snapshots"] == 1 and person["images"][0]["source"] == "frigate"
    assert person["images"][0]["frigate_file"] == "1.0-a-2.0-unknown-0.5.webp"
