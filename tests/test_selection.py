import numpy as np
import pytest

from if_curator.config import Config
from if_curator.faces import FaceCandidate, select_face_candidates


def candidate(index, vector=None, confidence=0.9):
    if vector is None:
        vector = np.eye(512)[index]
    return FaceCandidate(
        str(index),
        "p",
        str(index),
        embedding=vector,
        effective_dimensions=(120, 140),
        measurements={"detection_confidence": confidence},
        created_at=f"2020-01-{index + 1:02}",
    )


@pytest.mark.parametrize("mode", ["time", "smart"])
def test_tiny_pool_does_not_bypass_gates(mode):
    good, bad, invalid = candidate(0), candidate(1), candidate(2, np.zeros(512))
    bad.reasons = ["Blurry"]
    assert select_face_candidates([good, bad, invalid], 30, mode) == [good]
    assert "invalid_embedding" in invalid.reasons
    assert select_face_candidates([], 30, mode) == []


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "auto"])
def test_invalid_limits(limit):
    with pytest.raises(ValueError):
        select_face_candidates([], limit)


def test_duplicates_keep_higher_quality():
    low, high = candidate(0, confidence=0.8), candidate(1, np.eye(512)[0], confidence=0.99)
    assert select_face_candidates([low, high], 30) == [high]
    assert low.reasons == ["near_duplicate"]


def test_deterministic_and_ceiling():
    def run(order):
        return [c.asset_id for c in select_face_candidates([candidate(i) for i in order], 3)]

    assert run(range(8)) == run(reversed(range(8)))
    assert len(run(range(8))) == 3


def test_outliers_and_secondary_cluster(monkeypatch):
    monkeypatch.setattr(Config, "FACE_DUPLICATE_DISTANCE", 0.001)
    rng = np.random.default_rng(3)
    records = []
    for i in range(30):
        center = np.eye(512)[0 if i < 20 else 1]
        records.append(candidate(i, center + rng.normal(0, 0.018, 512)))
    outlier = candidate(31, np.eye(512)[20])
    records.append(outlier)
    selected = select_face_candidates(records, 8)
    assert "isolated_outlier" in outlier.reasons
    assert any(int(c.asset_id) >= 20 for c in selected)
    assert any(int(c.asset_id) < 20 for c in selected)


def test_zero_dispersion_skips_outlier_gate():
    records = [candidate(i) for i in range(10)]
    assert len(select_face_candidates(records, 30)) == 10
    assert not any(c.reasons for c in records)


def test_invalid_embeddings():
    vectors = [np.full(512, np.nan), np.ones(511), np.ones((1, 512)), np.zeros(512)]
    records = [candidate(i, vector) for i, vector in enumerate(vectors)]
    assert select_face_candidates(records, 30) == []
    assert all(c.reasons == ["invalid_embedding"] for c in records)
