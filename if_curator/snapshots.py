"""Label Frigate's unlabeled face snapshots with the Immich person they match.

Frigate saves recent face attempts from its cameras, cropped and stored exactly as library
faces are. They are the camera views a phone library lacks, so labeling the right ones is
the most direct way to move a person's center toward how cameras see them. A wrong label
does the opposite, so a snapshot is labeled only when Frigate would confidently recognize
it against the person's whole Immich library, and clearly prefer that person over every
other queued one.
"""

import datetime
from urllib.parse import quote

import cv2
import numpy as np
import requests

from .faces import BATCH, measure, verdict
from .frigate import cosine_for
from .immich import bounded_map
from .selection import DUPLICATE_SIMILARITY, Candidate, Job, farthest_points, unit

MATCH_MARGIN = 0.1  # Cosine by which the best person must beat the runner-up.


class Frigate:
    def __init__(self, url: str, user: str = "", password: str = ""):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        if user:
            response = self.session.post(f"{self.url}/api/login", json={"user": user, "password": password}, timeout=30)
            response.raise_for_status()

    def snapshots(self) -> list[str]:
        response = self.session.get(f"{self.url}/api/faces", timeout=30)
        response.raise_for_status()
        return response.json().get("train", [])

    def snapshot(self, name: str) -> bytes:
        response = self.session.get(f"{self.url}/clips/faces/train/{quote(name)}", timeout=30)
        response.raise_for_status()
        return response.content


def parse_name(name: str) -> tuple[str, str]:
    """Frigate names snapshots `{event id}-{timestamp}-{label}-{score}.webp`; event ids contain one dash."""
    parts = name.rsplit(".", 1)[0].split("-")
    if len(parts) != 5:
        return name, ""
    try:
        taken = datetime.datetime.fromtimestamp(float(parts[2]), datetime.UTC).isoformat(timespec="seconds")
    except (ValueError, OverflowError):
        taken = ""
    return f"{parts[0]}-{parts[1]}", taken


def analyze_snapshots(frigate: Frigate, model, settings, progress=lambda **_: None) -> list[Candidate]:
    """Every snapshot, checked for quality like a library face and embedded the way Frigate does."""
    names = frigate.snapshots()

    def fetch(name: str):
        try:
            return name, frigate.snapshot(name)
        except requests.RequestException:
            return name, None

    snapshots, batch = [], []

    def embed_batch():
        for (candidate, _), vector in zip(batch, model.embed([face for _, face in batch])):
            candidate.embedding = vector
            candidate.reason = None if vector is not None else "no_landmarks"
        batch.clear()

    for done, (name, data) in enumerate(bounded_map(fetch, names), 1):
        candidate = Candidate(name, taken=parse_name(name)[1], data=data)
        snapshots.append(candidate)
        face = None if data is None else cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if face is None:
            candidate.reason = "download_failed"
        else:
            candidate.measures = {"detected": True, **measure(face)}
            candidate.reason = verdict(candidate.measures, settings) or "pending"
            if candidate.reason == "pending":
                batch.append((candidate, face))
                if len(batch) == BATCH:
                    embed_batch()
        progress(completed=done, total=len(names))
    embed_batch()
    return snapshots


def label_snapshots(snapshots: list[Candidate], jobs: list[Job], recognition_threshold: float) -> None:
    """Give each queued person the snapshots that are clearly them, one per event, spread out."""
    people = [job for job in jobs if job.object_class is None and job.center is not None]
    usable = [c for c in snapshots if c.reason is None]
    if not people or not usable:
        return
    scores = unit([c.embedding for c in usable]) @ unit([job.center for job in people]).T
    ranked = np.sort(scores, axis=1)
    runner_up = ranked[:, -2] if len(people) > 1 else np.full(len(usable), -1.0)
    floor = cosine_for(recognition_threshold)
    best_per_event: dict[tuple[int, str], tuple[float, Candidate]] = {}
    for candidate, row, second in zip(usable, scores, runner_up):
        person = int(np.argmax(row))
        candidate.measures["cosine"] = round(float(row[person]), 3)
        if row[person] < floor:
            candidate.reason = "no_match"
        elif row[person] - second < MATCH_MARGIN:
            candidate.reason = "ambiguous"
        else:
            key = (person, parse_name(candidate.asset_id)[0])
            previous = best_per_event.get(key)
            if previous and previous[0] >= row[person]:
                candidate.reason = "same_event"
            else:
                if previous:
                    previous[1].reason = "same_event"
                best_per_event[key] = (float(row[person]), candidate)
    for index, job in enumerate(people):
        matched = [c for (person, _), (_, c) in best_per_event.items() if person == index]
        if matched:
            chosen = farthest_points(unit([c.embedding for c in matched]), job.count, DUPLICATE_SIMILARITY)
            job.snapshots = [matched[i] for i in chosen]
