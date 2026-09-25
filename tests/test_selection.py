import numpy as np

from if_curator.selection import Candidate, Job, farthest_points, geometric_median, select, unit


def direction(*weights):
    vector = np.zeros(512)
    vector[: len(weights)] = weights
    return vector


def job(name, vectors, count=10, object_class=None):
    candidates = [Candidate(f"{name}-{i}", embedding=np.asarray(v, float)) for i, v in enumerate(vectors)]
    return Job({"id": name, "name": name}, count, object_class, candidates)


def test_farthest_points_start_typical_then_spread():
    units = unit([direction(1, 0), direction(1, 0.1), direction(1, -0.1), direction(1, 1), direction(1, -1)])
    chosen = farthest_points(units, 3)
    assert chosen[0] == 0 and set(chosen[1:]) == {3, 4}


def test_farthest_points_skip_duplicates_and_never_repeat():
    units = unit([direction(1, 0), direction(1, 0.01), direction(0, 1)])
    chosen = farthest_points(units, 10)
    assert len(chosen) == 2 and 2 in chosen  # 0 and 1 are near-duplicates
    assert sorted(farthest_points(units, 10, duplicate=np.inf)) == [0, 1, 2]


def test_geometric_median_ignores_a_minority():
    rng = np.random.default_rng(0)
    inliers = [direction(1) + rng.normal(0, 0.01, 512) for _ in range(7)]
    units = unit(np.vstack(inliers + [direction(0, 1), direction(0, 0, 1), direction(0, 1, 1)]))
    assert geometric_median(units) @ unit(direction(1)) > 0.99


def test_select_drops_wrong_faces_and_scores_held_out():
    rng = np.random.default_rng(1)
    alice = [direction(1, 0) + rng.normal(0, 0.03, 512) for _ in range(12)]
    bob = [direction(0.5, 1) + rng.normal(0, 0.03, 512) for _ in range(12)]
    stranger = direction(0, 0, 1)
    look_alike = direction(0.4, 1)  # tagged as Alice but closer to Bob
    jobs = [job("alice", alice + [stranger, look_alike], count=3), job("bob", bob, count=3)]
    select(jobs, recognition_threshold=0.9)
    reasons = {c.asset_id: c.reason for c in jobs[0].candidates if c.reason}
    assert reasons == {"alice-12": "unlike_person", "alice-13": "resembles bob"}
    assert all(len(j.selected) == 3 and j.recognized == 1.0 for j in jobs)
    assert not {id(c) for c in jobs[0].selected} & {id(c) for c in jobs[1].selected}


def test_objects_are_spread_but_not_identity_checked():
    objects = job("rex", [direction(1), direction(0, 1), direction(0, 0, 1)], count=2, object_class="dog")
    select([objects], recognition_threshold=0.9)
    assert len(objects.selected) == 2 and objects.recognized is None
    assert all(c.reason is None for c in objects.candidates)
