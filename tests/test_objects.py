import numpy as np
from conftest import FakeImmich, photo

from if_curator.config import Settings
from if_curator.objects import analyze_objects
from if_curator.selection import Job


class FakeObjects:
    def class_id(self, name):
        return 16

    def detect(self, image, class_id):
        return [(10, 10, 200, 150), (300, 300, 330, 330)]  # the second is too small to use

    def embed(self, crops):
        return [np.full(4, crop.shape[0], float) for crop in crops]


def test_object_crops_become_candidates():
    immich = FakeImmich([photo(1), photo(2)], broken={("asset-002", False)})
    job = Job({"id": "p1", "name": "Rex"}, 5, object_class="dog")
    analyze_objects(immich, FakeObjects(), job, Settings())
    (crop,) = job.eligible
    assert crop.asset_id == "asset-001" and crop.box == (10, 10, 200, 150) and crop.frame == (800, 600)
    assert crop.embedding.tolist() == [140.0] * 4
    assert [c.reason for c in job.candidates] == [None, "download_failed"]
