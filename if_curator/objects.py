"""Crops of one object class from a person's photos, for Frigate object classification."""

import os
from pathlib import Path

import numpy as np

from .faces import MAX_CANDIDATES
from .immich import DOWNLOAD_ERRORS, Immich, bounded_map
from .selection import Candidate, Job

MODEL = "yolo11m.pt"
MIN_CONFIDENCE = 0.5
MIN_SIZE = 64


class ObjectModel:
    def __init__(self, cache_dir: str, force_cpu: bool = False):
        os.environ.setdefault("YOLO_VERBOSE", "false")
        try:
            import torch
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("Object mode needs the objects extra: uv run --extra objects if-curator") from None
        path = Path(cache_dir) / "yolo" / MODEL
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(path))
        self.torch_device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.device = "GPU" if self.torch_device == "cuda" else "CPU"

    def class_id(self, name: str) -> int:
        ids = {label: i for i, label in self.model.names.items()}
        if name not in ids:
            raise LookupError(f"YOLO doesn't know “{name}”. Try one of: {', '.join(sorted(ids))}")
        return ids[name]

    def detect(self, bgr: np.ndarray, class_id: int) -> list[tuple[float, float, float, float]]:
        result = self.model(bgr, classes=[class_id], conf=MIN_CONFIDENCE, device=self.torch_device, verbose=False)[0]
        return [tuple(box) for box in result.boxes.xyxy.tolist()]

    def embed(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        return [v.cpu().numpy() for v in self.model.embed(crops, device=self.torch_device, verbose=False)]


def analyze_objects(immich: Immich, model: ObjectModel, job: Job, settings, progress=lambda **_: None) -> None:
    class_id = model.class_id(job.object_class)
    photos = sorted(immich.photos(job.person["id"], settings.YEARS_FILTER), key=lambda a: a.get("fileCreatedAt", ""))
    if len(photos) > MAX_CANDIDATES:
        photos = [photos[i] for i in np.unique(np.linspace(0, len(photos) - 1, MAX_CANDIDATES).round().astype(int))]

    def fetch(asset):
        try:
            return asset, immich.image(asset["id"])
        except DOWNLOAD_ERRORS:
            return asset, None

    job.candidates, pending = [], []

    def embed_pending():
        for (candidate, _), vector in zip(pending, model.embed([crop for _, crop in pending])):
            candidate.embedding = vector
        pending.clear()

    for done, (asset, image) in enumerate(bounded_map(fetch, photos), 1):
        photo = Candidate(asset["id"], taken=asset.get("fileCreatedAt") or "", checksum=asset.get("checksum") or "")
        boxes = [] if image is None else model.detect(image, class_id)
        boxes = [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) >= MIN_SIZE]
        if not boxes:
            photo.reason = "download_failed" if image is None else "no_object"
            job.candidates.append(photo)
        for box in boxes:
            crop = Candidate(photo.asset_id, photo.taken, photo.checksum, box=box, frame=image.shape[1::-1])
            job.candidates.append(crop)
            pending.append((crop, image[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]))
        if len(pending) >= 32:
            embed_pending()
        progress(completed=done, total=len(photos))
    if pending:
        embed_pending()
