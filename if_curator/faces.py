"""Find each photo's face the way Frigate would, check its quality, and embed it."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .immich import BOX_KEYS, DOWNLOAD_ERRORS, Immich, bounded_map
from .selection import Candidate, Job

PIPELINE = "frigate-0.17-yunet-v1"  # Change to invalidate cached analyses.
MAX_CANDIDATES = 1000
CONTEXT = 0.5  # Detect within this margin around Immich's box, as Frigate does in an uploaded photo.
BATCH = 8


def scale_box(box, frame, size):
    """Map a box from Immich's preview pixels onto an image of `size` (width, height)."""
    sx, sy = size[0] / frame[0], size[1] / frame[1]
    if abs(sx / sy - 1) > 0.01:
        raise ValueError("face box doesn't fit the image")
    return box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy


def crop_region(image: np.ndarray, box, margin: float):
    """Crop a box plus `margin` of its size on each side. Returns the crop and the box inside it."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = max(0, int(x1 - mx)), max(0, int(y1 - my))
    right, bottom = min(image.shape[1], int(np.ceil(x2 + mx))), min(image.shape[0], int(np.ceil(y2 + my)))
    return image[top:bottom, left:right], (x1 - left, y1 - top, x2 - left, y2 - top)


def fetch_region(immich: Immich, candidate: Candidate, margin: float, original: bool = False):
    image = immich.image(candidate.asset_id, original)
    return crop_region(image, scale_box(candidate.box, candidate.frame, image.shape[1::-1]), margin)


def measure(face: np.ndarray) -> dict:
    """Quality as ArcFace will see it, after scaling the face to 112 x 112."""
    small = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(gray.mean()), 1),
        "color": round(float(np.ptp(small, axis=2).mean()), 1),
    }


def verdict(measures: dict, settings) -> str | None:
    if not measures["detected"]:
        return "no_face_detected"
    if measures["brightness"] < 30:
        return "too_dark"
    if measures["brightness"] > 225:
        return "too_bright"
    if measures["sharpness"] < settings.BLUR_THRESHOLD:
        return "blurry"
    if settings.REJECT_GRAYSCALE and measures["color"] < 5:
        return "grayscale"
    if measures.get("aligned") is False:
        return "no_landmarks"
    return None


def locate(immich: Immich, asset: dict, person_id: str, min_size: int) -> Candidate:
    """The person's face box in a photo, from Immich's metadata."""
    candidate = Candidate(asset["id"], taken=asset.get("fileCreatedAt") or "", checksum=asset.get("checksum") or "")
    if asset.get("isEdited"):
        candidate.reason = "edited"  # Immich's boxes may not match the unedited original.
        return candidate
    try:
        faces = immich.target_faces(asset, person_id)
    except DOWNLOAD_ERRORS:
        candidate.reason = "download_failed"
        return candidate
    if len(faces) != 1:
        candidate.reason = "several_faces" if faces else "no_face_box"
        return candidate
    try:
        x1, y1, x2, y2, width, height = (float(faces[0][key]) for key in BOX_KEYS)
    except (KeyError, TypeError, ValueError):
        candidate.reason = "no_face_box"
        return candidate
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
    candidate.face_id, candidate.box, candidate.frame = faces[0].get("id"), (x1, y1, x2, y2), (int(width), int(height))
    if min(x2 - x1, y2 - y1) < min_size:
        candidate.reason = "too_small"
    return candidate


class FaceCache:
    """Analysis results keyed by photo, face box and pipeline, so reruns skip downloads."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory) / "faces"

    def _path(self, c: Candidate) -> Path:
        key = json.dumps([PIPELINE, c.asset_id, c.checksum, c.face_id, c.box, c.frame])
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.npz"

    def load(self, c: Candidate) -> tuple[dict, np.ndarray | None] | None:
        try:
            with np.load(self._path(c)) as data:
                embedding = data["embedding"]
                return json.loads(str(data["measures"])), embedding if embedding.size else None
        except (OSError, ValueError, KeyError):
            return None

    def save(self, c: Candidate, measures: dict, embedding: np.ndarray | None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(c)
        partial = path.with_suffix(".part.npz")
        vector = np.empty(0, np.float32) if embedding is None else embedding
        np.savez(partial, measures=np.array(json.dumps(measures)), embedding=vector)
        partial.replace(path)


def _apply(c: Candidate, measures: dict, embedding: np.ndarray | None, settings) -> bool:
    """Record an analysis on the candidate. False if it still needs an embedding."""
    c.measures = measures
    c.reason = verdict(measures, settings)
    if c.reason is None and embedding is None:
        return False
    c.embedding = embedding
    return True


def analyze_faces(immich: Immich, model, job: Job, settings, progress=lambda **_: None) -> None:
    person_id = job.person["id"]
    photos = immich.photos(person_id, settings.YEARS_FILTER)
    job.candidates = list(bounded_map(lambda a: locate(immich, a, person_id, settings.MIN_FACE_SIZE), photos))

    usable = sorted((c for c in job.candidates if c.reason is None), key=lambda c: (c.taken, c.asset_id))
    if len(usable) > MAX_CANDIDATES:  # Sample evenly through time.
        keep = set(np.linspace(0, len(usable) - 1, MAX_CANDIDATES).round().astype(int))
        for i, c in enumerate(usable):
            if i not in keep:
                c.reason = "sample_limit"
        usable = [c for c in usable if c.reason is None]

    cache = FaceCache(settings.CACHE_DIR)
    todo = [c for c in usable if not ((hit := cache.load(c)) and _apply(c, *hit, settings))]
    done = len(usable) - len(todo)
    progress(completed=done, total=len(usable))

    def fetch(c: Candidate):
        try:
            return c, *fetch_region(immich, c, CONTEXT)
        except ValueError:
            c.reason = "coordinate_mismatch"
        except DOWNLOAD_ERRORS:
            c.reason = "download_failed"
        return c, None, None

    batch = []

    def embed_batch():
        for (c, measures, _), vector in zip(batch, model.embed([face for _, _, face in batch])):
            measures["aligned"] = vector is not None
            cache.save(c, measures, vector)
            _apply(c, measures, vector, settings)
        batch.clear()

    for c, region, target in bounded_map(fetch, todo):
        if region is not None:
            box = model.detect(region, target)
            measures = {"detected": box is not None}
            if box is not None:
                face = region[box[1] : box[3], box[0] : box[2]]
                measures.update(measure(face))
            if verdict(measures, settings):
                cache.save(c, measures, None)
                _apply(c, measures, None, settings)
            else:
                batch.append((c, measures, face))
                if len(batch) == BATCH:
                    embed_batch()
        done += 1
        progress(completed=done, total=len(usable))
    embed_batch()
