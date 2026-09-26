"""Choose the images that give Frigate the most robust center for each person.

Frigate recognizes a face by its cosine similarity to the trimmed mean of a person's
library. A library drawn from the dense middle of someone's photos (posed, frontal, well
lit) produces a center that carries those conditions with it, and camera faces rarely
share them. Farthest-point sampling instead spreads the library over every pose, light
and age in the photos, so those nuisances cancel in the mean and the center is closer to
identity alone. Spreading out also picks outliers first, so faces that don't match the
person are removed before sampling.
"""

from dataclasses import dataclass, field

import numpy as np

from .frigate import confidence, trimmed_mean

# Frigate's confidence midpoint. Faces less similar than this to the person's overall
# center are more likely someone else, or too degraded to be useful.
IDENTITY_FLOOR = 0.3
# Two faces this similar show the same look; exporting both only adds weight to it.
DUPLICATE_SIMILARITY = 0.9


@dataclass
class Candidate:
    asset_id: str
    taken: str = ""
    checksum: str = ""
    face_id: str | None = None
    box: tuple[float, float, float, float] | None = None  # In `frame` pixels: Immich's preview.
    frame: tuple[int, int] | None = None
    measures: dict = field(default_factory=dict)
    embedding: np.ndarray | None = field(default=None, repr=False)
    reason: str | None = None  # Why it can't be used; None means eligible.
    data: bytes | None = field(default=None, repr=False)  # A Frigate snapshot's image, exported as is.


@dataclass
class Job:
    person: dict
    count: int
    object_class: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    selected: list[Candidate] = field(default_factory=list)
    recognized: float | None = None
    center: np.ndarray | None = field(default=None, repr=False)  # Robust center of the usable faces.
    snapshots: list[Candidate] = field(default_factory=list)  # Frigate snapshots labeled as this person.

    @property
    def name(self) -> str:
        return self.person["name"]

    @property
    def eligible(self) -> list[Candidate]:
        return [c for c in self.candidates if c.reason is None]


def unit(vectors) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def geometric_median(units: np.ndarray) -> np.ndarray:
    """A center that a minority of wrong faces can't drag away (Weiszfeld's algorithm)."""
    center = units.mean(axis=0)
    for _ in range(100):
        weights = 1 / np.maximum(np.linalg.norm(units - center, axis=1), 1e-9)
        updated = np.average(units, axis=0, weights=weights)
        if np.linalg.norm(updated - center) < 1e-7:
            break
        center = updated
    return unit(center)


def farthest_points(units: np.ndarray, count: int, duplicate: float = DUPLICATE_SIMILARITY) -> list[int]:
    """Start at the most typical face, then repeatedly add the face least like any chosen."""
    similarity = units @ units.T
    chosen = [int(np.argmax(similarity.sum(axis=1)))]
    closest = similarity[chosen[0]].copy()
    closest[chosen] = np.inf
    while len(chosen) < count:
        index = int(np.argmin(closest))
        if closest[index] >= duplicate:
            break
        chosen.append(index)
        closest = np.maximum(closest, similarity[index])
        closest[index] = np.inf
    return chosen


def select(jobs: list[Job], recognition_threshold: float) -> None:
    faces = [job for job in jobs if job.object_class is None]
    for job in faces:
        job.center = geometric_median(unit([c.embedding for c in job.eligible])) if job.eligible else None
    for job in faces:
        rivals = [(other.name, other.center) for other in faces if other is not job and other.center is not None]
        for candidate in job.eligible:
            vector = unit(candidate.embedding)
            own = vector @ job.center
            if own < IDENTITY_FLOOR:
                candidate.reason = "unlike_person"
                continue
            for name, center in rivals:
                if vector @ center > own:
                    candidate.reason = f"resembles {name}"
                    break
    for job in jobs:
        pool = job.eligible
        if pool:
            duplicate = DUPLICATE_SIMILARITY if job.object_class is None else np.inf
            job.selected = [pool[i] for i in farthest_points(unit([c.embedding for c in pool]), job.count, duplicate)]
    _score(faces, recognition_threshold)


def _score(faces: list[Job], threshold: float) -> None:
    """Share of each person's unselected eligible faces that Frigate would recognize."""
    enrolled = [job for job in faces if job.selected]
    if not enrolled:
        return
    centers = unit([trimmed_mean([c.embedding for c in job.selected]) for job in enrolled])
    for index, job in enumerate(enrolled):
        chosen = {id(c) for c in job.selected}
        held_out = [c for c in job.eligible if id(c) not in chosen]
        if not held_out:
            continue
        scores = unit([c.embedding for c in held_out]) @ centers.T
        best = scores.argmax(axis=1)
        hits = [b == index and round(confidence(s[b]), 2) >= threshold for b, s in zip(best, scores)]
        job.recognized = float(np.mean(hits))
