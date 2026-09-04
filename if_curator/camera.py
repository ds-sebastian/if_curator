"""Local, explicitly split camera face crops. Test pixels are never read by selection."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .frigate import blur_reduction, get_frigate_model


@dataclass
class CameraSample:
    sample_id: str
    person_id: str | None
    split: str
    capture_group: str
    path: Path
    sha256: str
    asset_id: str | None = None
    embedding: np.ndarray | None = None
    reduction: float = 0.0
    error: str | None = None

    def record(self):
        return {
            "id": self.sample_id,
            "person_id": self.person_id,
            "split": self.split,
            "capture_group": self.capture_group,
            "path": str(self.path),
            "sha256": self.sha256,
            "asset_id": self.asset_id,
            "error": self.error,
            "blur_reduction": self.reduction,
        }


def load_manifest(path):
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1 or not isinstance(document.get("samples"), list):
        raise ValueError("Camera manifest requires schema_version 1 and a samples list")
    samples, ids, groups, hashes, assets = [], set(), {}, {}, {}
    for item in document["samples"]:
        if not isinstance(item, dict) or not {"id", "person_id", "split", "capture_group", "path"} <= item.keys():
            raise ValueError("Camera sample requires id, person_id, split, capture_group, path")
        sid, person, split, group = item["id"], item["person_id"], item["split"], item["capture_group"]
        if not all(isinstance(v, str) and v.strip() for v in (sid, group, item["path"])) or sid in ids:
            raise ValueError("Camera IDs must be unique; paths and capture groups must be nonempty")
        if (
            split not in {"reference", "validation", "test"}
            or (person is not None and (not isinstance(person, str) or not person.strip()))
            or (split == "reference" and person is None)
        ):
            raise ValueError("Invalid camera split or person_id; null means unknown and cannot be a reference")
        source = (path.parent / item["path"]).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        asset = item.get("asset_id")
        for key, index in ((group, groups), (digest, hashes), (asset, assets)):
            if key is not None:
                if key in index and index[key] != split:
                    raise ValueError("Camera split leakage: capture group, file or asset crosses splits")
                index[key] = split
        if any(s.sha256 == digest and s.person_id != person for s in samples):
            raise ValueError("Camera labels conflict for the same image")
        ids.add(sid)
        samples.append(CameraSample(sid, person, split, group, source, digest, asset))
    return sorted(samples, key=lambda s: s.sample_id)


def embed_samples(samples, split):
    """Failures remain in metrics. Difficult evaluation crops never face enrollment quality gates."""
    wanted = [s for s in samples if s.split == split and s.embedding is None and s.error is None]
    if not wanted:
        return
    model = get_frigate_model()  # Model initialization failure aborts, not a measured recognition failure.
    for sample in wanted:
        data = sample.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sample.sha256:
            raise ValueError("Camera sample changed since manifest loading")
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            sample.error = "decode_failed"
            continue
        sample.reduction = blur_reduction(bgr)
        try:
            sample.embedding = model.get(bgr)
        except (ValueError, cv2.error) as exc:
            sample.error = f"inference_failed:{type(exc).__name__}"
