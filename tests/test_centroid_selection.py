import numpy as np
import pytest

from if_curator.config import Config
from if_curator.faces import FaceCandidate
from if_curator.frigate import class_mean, unit
from if_curator.selection import approved_pool, objective, robust_reference, select_jobs


def vector(angle=0, magnitude=1):
    result = np.zeros(512, np.float32)
    result[:2] = np.cos(angle), np.sin(angle)
    return result * magnitude


def candidate(i, angle=0, person="p"):
    return FaceCandidate(
        str(i),
        person,
        embedding=vector(angle),
        created_at=f"2026-01-{i + 1:02d}",
        measurements={"detection_confidence": 0.9, "blur_variance": 150},
        effective_dimensions=(120, 120),
    )


def job(candidates, limit=30, person="p", mode="smart"):
    return {
        "person": {"id": person},
        "candidates": candidates,
        "requested_limit": limit,
        "selection_mode": mode,
        "config": {"mode": "face"},
    }


def test_independent_similar_faces_can_improve_centroid():
    # Cosine distance .0199 is below the previous .05 dedupe gate.
    records = [candidate(0, -0.1), candidate(1, 0.1)]
    j = job(records)
    select_jobs([j])
    assert j["limit"] == 2
    assert not any(c.reasons for c in records)
    history = j["selection_report"]["objective_history"]
    assert history[-1] < history[0]
    assert j["selection_report"]["centroid_reference_cosine"] == pytest.approx(1)


def test_stops_early_and_records_leave_one_out():
    j = job([candidate(i) for i in range(20)])
    select_jobs([j])
    assert j["limit"] == 1
    assert j["selection_report"]["leave_one_out"][0]["confidence"] is None
    j = job([candidate(0, -0.1), candidate(1, 0.1)])
    select_jobs([j])
    assert j["selection_report"]["leave_one_out"][0]["cosine"] == pytest.approx(np.cos(0.2))


def test_duplicates_and_bursts_use_capture_evidence():
    records = [candidate(i, i / 100) for i in range(5)]
    records[0].image_hash = records[1].image_hash = "same-bytes"
    records[2].capture_group = records[3].capture_group = "burst"
    pool = approved_pool(records, "smart")
    assert [c.asset_id for c in pool] == ["0", "2", "4"]
    assert records[1].reasons == records[3].reasons == ["duplicate_capture"]


def test_pixel_similarity_only_deduplicates_nearby_captures():
    records = [candidate(i) for i in range(3)]
    records[0].created_at = "2026-01-01T12:00:00Z"
    records[1].created_at = "2026-01-01T12:00:01Z"
    for c in records:
        c.pixel_signature = np.ones(768)
    assert [c.asset_id for c in approved_pool(records, "smart")] == ["0", "2"]


def test_reselection_deterministic_and_keeps_raw_embeddings():
    records = [candidate(i, angle) for i, angle in enumerate([-0.3, 0.1, 0.2, 0.3])]
    records[0].embedding *= 3
    original = records[0].embedding.copy()
    j = job(records, 3)
    first = select_jobs([j])
    ids = [c.asset_id for c in j["selected_faces"]]
    second = select_jobs([j])
    assert [c.asset_id for c in j["selected_faces"]] == ids
    np.testing.assert_array_equal(first["centers"]["p"], second["centers"]["p"])
    np.testing.assert_array_equal(records[0].embedding, original)


def test_cross_identity_ambiguity_penalized():
    reference = {"p": vector(0), "q": vector(0.4)}
    close = {"p": vector(0.2), "q": vector(0.21)}
    separated = {"p": vector(0), "q": vector(0.4)}
    assert objective(separated, reference, []) < objective(close, reference, [])


def test_robust_reference_balances_capture_groups():
    reference = robust_reference([vector(0)] * 10 + [vector(0.2)], ["burst"] * 10 + ["other"])
    np.testing.assert_allclose(reference, unit(vector(0.1)), atol=1e-6)


def test_time_mode_enforces_quality_and_ceiling():
    records = [candidate(i, i / 10) for i in range(8)]
    records[0].reasons = ["blurry"]
    j = job(records, 3, mode="time")
    select_jobs([j])
    assert [c.asset_id for c in j["selected_faces"]] == ["1", "4", "7"]


def test_multi_identity_order_independent():
    def run(reverse):
        jobs = [
            job([candidate(i, a, p) for i, a in enumerate(angles)], 3, p)
            for p, angles in [("p", [-0.2, 0.1, 0.3]), ("q", [1, 1.2, 1.4])]
        ]
        result = select_jobs(jobs[::-1] if reverse else jobs)
        return {p: center.tolist() for p, center in result["centers"].items()}

    assert run(False) == run(True)


def test_centroid_degenerate_combinations_rejected():
    with pytest.raises(ValueError):
        class_mean([vector(0), -vector(0)])


def test_unknown_score_configuration_must_be_below_recognition(monkeypatch):
    monkeypatch.setattr(Config, "FRIGATE_UNKNOWN_SCORE", 0.95)
    with pytest.raises(ValueError, match="must not exceed"):
        Config.validate_settings()
