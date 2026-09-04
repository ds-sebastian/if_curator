"""Target-bound, file-backed face preparation and representative selection.

InsightFace distances are curation heuristics, not Frigate confidence scores.
"""

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageOps

from .cache import EmbeddingCache
from .config import Config
from .embeddings import get_insightface_app
from .immich_api import fetch_image_source, resolve_face_metadata
from .quality import assess_quality

PREPROCESSING_VERSION = "target-face-jpeg-v1"


class FacePipelineError(RuntimeError):
    """A face run cannot be safely completed (no silent fallback)."""


@dataclass
class FaceCandidate:
    asset_id: str
    person_id: str
    face_id: str | None = None
    created_at: str = ""
    metadata_dimensions: tuple[int, int] | None = None
    source: str | None = None
    source_dimensions: tuple[int, int] | None = None
    effective_dimensions: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    prepared_path: Path | None = None
    image_hash: str | None = None
    measurements: dict[str, float] = field(default_factory=dict)
    embedding: np.ndarray | None = field(default=None, repr=False)
    reasons: list[str] = field(default_factory=list)
    selected: bool = False
    selection_reason: str | None = None
    output_path: str | None = None

    def record(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "person_id": self.person_id,
            "face_id": self.face_id,
            "created_at": self.created_at,
            "metadata_dimensions": self.metadata_dimensions,
            "source": self.source,
            "source_dimensions": self.source_dimensions,
            "effective_dimensions": self.effective_dimensions,
            "bbox": self.bbox,
            "sha256": self.image_hash,
            "measurements": self.measurements,
            "rejection_reasons": self.reasons,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
            "output_path": self.output_path,
        }


def model_fingerprint(app) -> str:
    """Hash actual model files, including detector and optional alignment models."""
    digest = hashlib.sha256()
    models = getattr(app, "models", {})
    if "recognition" not in models:
        raise FacePipelineError("InsightFace recognition model is unavailable")
    for name, model in sorted(models.items()):
        path = Path(model.model_file)
        digest.update(name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return "buffalo_l:" + digest.hexdigest()


def intersection_over_union(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def detect_target(app, image: Image.Image, expected_bbox):
    """Detect without embedding arbitrary neighboring faces."""
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    try:
        boxes, landmarks = app.det_model.detect(bgr, max_num=0)
    except Exception as exc:
        raise FacePipelineError("Local face detector failed") from exc
    matches = [i for i, box in enumerate(boxes) if intersection_over_union(box[:4], expected_bbox) >= 0.5]
    if len(matches) != 1:
        raise ValueError("missing_or_ambiguous_local_face")
    index = matches[0]
    if landmarks is None:
        raise ValueError("missing_local_landmarks")
    face = SimpleNamespace(bbox=boxes[index, :4], kps=landmarks[index], det_score=float(boxes[index, 4]))
    if not np.isfinite(face.det_score) or not np.isfinite(face.kps).all() or face.kps.shape != (5, 2):
        raise ValueError("invalid_local_face")
    return bgr, face


def scaled_bbox(metadata: dict, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    mw, mh = float(metadata["imageWidth"]), float(metadata["imageHeight"])
    w, h = image_size
    coords = np.array(
        [
            metadata[key]
            for key in (
                "boundingBoxX1",
                "boundingBoxY1",
                "boundingBoxX2",
                "boundingBoxY2",
            )
        ],
        dtype=float,
    )
    if not np.isfinite([mw, mh, *coords]).all() or min(mw, mh) <= 0:
        raise ValueError("invalid_face_geometry")
    # Both representations must have the same oriented aspect ratio. Allow rounding.
    if abs(w / h - mw / mh) > max(2 / h, 2 / mh):
        raise ValueError("coordinate_mismatch")
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 <= mw and 0 <= y1 < y2 <= mh):
        raise ValueError("incomplete_or_invalid_face_box")
    return float(x1 * w / mw), float(y1 * h / mh), float(x2 * w / mw), float(y2 * h / mh)


def _quality_order(candidate: FaceCandidate) -> tuple:
    return (
        -candidate.measurements.get("detection_confidence", 0),
        -candidate.measurements.get("blur_variance", 0),
        -min(candidate.effective_dimensions or (0, 0)),
        candidate.asset_id,
        candidate.face_id or "",
    )


def _normalize_embedding(embedding) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size != 512 or not np.isfinite(vector).all():
        raise ValueError("invalid_embedding")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("invalid_embedding")
    return vector / norm


def _prepare_one(candidate: FaceCandidate, metadata: dict, source_path: Path, directory: Path, app, fingerprint):
    with Image.open(source_path) as loaded:
        image = ImageOps.exif_transpose(loaded).convert("RGB")
    candidate.source_dimensions = image.size
    bbox = scaled_bbox(metadata, image.size)
    candidate.bbox = bbox
    x1, y1, x2, y2 = bbox
    fw, fh = x2 - x1, y2 - y1
    candidate.effective_dimensions = fw, fh
    if min(fw, fh) < Config.MIN_FACE_WIDTH:
        raise ValueError("face_too_small")
    margin = Config.FACE_MARGIN
    left, top = max(0, math.floor(x1 - fw * margin)), max(0, math.floor(y1 - fh * margin))
    right = min(image.width, math.ceil(x2 + fw * margin))
    bottom = min(image.height, math.ceil(y2 + fh * margin))
    crop = image.crop((left, top, right, bottom))
    expected = (x1 - left, y1 - top, x2 - left, y2 - top)
    aligned_target = None
    if Config.ENABLE_FACE_ALIGNMENT:
        from insightface.utils.face_align import estimate_norm

        _, detected = detect_target(app, crop, expected)
        matrix = np.asarray(estimate_norm(detected.kps, image_size=112), dtype=np.float32)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise ValueError("invalid_alignment_transform")
        pixels = cv2.warpAffine(np.asarray(crop), matrix, (112, 112), borderValue=0)
        corners = np.array(
            [
                [expected[0], expected[1]],
                [expected[2], expected[1]],
                [expected[2], expected[3]],
                [expected[0], expected[3]],
            ]
        )
        transformed = np.c_[corners, np.ones(4)] @ matrix.T
        expected = (*np.maximum(0, transformed.min(axis=0)), *np.minimum(112, transformed.max(axis=0)))
        # Preserve the verified target through the affine transform. Redetecting
        # a tightly aligned 112px image can lose a valid face or change association.
        projected_landmarks = np.c_[detected.kps, np.ones(5)] @ matrix.T
        if not ((projected_landmarks >= 0).all() and (projected_landmarks < 112).all()):
            raise ValueError("incomplete_aligned_face")
        aligned_target = SimpleNamespace(
            bbox=np.asarray(expected),
            kps=projected_landmarks.astype(np.float32),
            det_score=detected.det_score,
        )
        crop = Image.fromarray(pixels)
    encoded = BytesIO()
    crop.save(encoded, format="JPEG", quality=95, subsampling=0)
    data = encoded.getvalue()
    candidate.image_hash = hashlib.sha256(data).hexdigest()
    identity = hashlib.sha256(f"{candidate.person_id}:{candidate.asset_id}:{candidate.face_id}".encode()).hexdigest()
    candidate.prepared_path = directory / f"{identity}.jpg"
    candidate.prepared_path.write_bytes(data)
    with Image.open(BytesIO(data)) as decoded:
        final = decoded.convert("RGB")
    if aligned_target is None:
        bgr, target = detect_target(app, final, expected)
    else:
        bgr = cv2.cvtColor(np.asarray(final), cv2.COLOR_RGB2BGR)
        target = aligned_target
    roi = final.crop((math.floor(expected[0]), math.floor(expected[1]), math.ceil(expected[2]), math.ceil(expected[3])))
    quality = assess_quality(
        roi,
        confidence=float(target.det_score),
        blur_threshold=Config.BLUR_THRESHOLD,
        min_confidence=Config.MIN_CONFIDENCE,
        reject_grayscale=Config.REJECT_GRAYSCALE,
    )
    candidate.measurements.update(quality.measurements)
    if not quality.passed:
        candidate.reasons.extend(quality.reasons)
        return
    key = f"{candidate.person_id}:{candidate.face_id}:{candidate.asset_id}:{candidate.image_hash}"
    version = f"{fingerprint}:{PREPROCESSING_VERSION}"
    cache = EmbeddingCache(Config.CACHE_DIR) if Config.ENABLE_CACHE else None
    emb = cache.get(key, version) if cache else None
    if emb is not None:
        try:
            emb = _normalize_embedding(emb)
        except ValueError:
            emb = None
    if emb is None:
        try:
            raw_embedding = app.models["recognition"].get(bgr, target)
        except Exception as exc:
            raise FacePipelineError("Local face embedding model failed") from exc
        emb = _normalize_embedding(raw_embedding)
        if cache:
            cache.put(key, emb, version)
    candidate.embedding = emb


def prepare_face_candidates(assets: list[dict], person_id: str, directory: Path, progress_callback=None):
    """Stage bounded downloads; run the shared model serially. Return records and model identity."""
    if not person_id:
        raise ValueError("person_id is required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_dir = directory / "sources"
    source_dir.mkdir(exist_ok=True)
    assets = sorted({a["id"]: a for a in assets}.values(), key=lambda a: (a.get("fileCreatedAt", ""), a["id"]))
    sampled = set(np.linspace(0, len(assets) - 1, min(3000, len(assets)), dtype=int)) if assets else set()
    records = [FaceCandidate(a["id"], person_id, created_at=a.get("fileCreatedAt", "")) for a in assets]
    work = []
    for i, (asset, record) in enumerate(zip(assets, records)):
        if i not in sampled:
            record.reasons.append("candidate_pool_cap")
        else:
            work.append((asset, record))
    if not work:
        source_dir.rmdir()
        return records, None
    app = get_insightface_app()
    if app is None:
        raise FacePipelineError("InsightFace unavailable; face quality validation requires the local detector")
    fingerprint = model_fingerprint(app)

    def download(item):
        asset, record = item
        try:
            metadata = resolve_face_metadata(asset, person_id)
            record.face_id = metadata.get("id")
            record.metadata_dimensions = (metadata["imageWidth"], metadata["imageHeight"])
            # Edits may transform boxes without transforming the original. Conservatively
            # reject until that representation's coordinate relationship can be verified.
            if asset.get("isEdited") or asset.get("edits") or asset.get("sidecarEdits"):
                raise ValueError("unverified_edited_coordinates")
            image, source = fetch_image_source(asset["id"], Config.USE_FULL_RESOLUTION)
            record.source = source
            path = source_dir / (hashlib.sha256(asset["id"].encode()).hexdigest() + ".png")
            try:
                image.save(path, format="PNG")
            finally:
                image.close()
            return record, metadata, path
        except Exception as exc:
            record.reasons.append(f"source_error:{type(exc).__name__}:{exc}")
            return record, None, None

    # executor.map would enqueue the whole library; submit windows of eight instead.
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(work), 8):
            futures = [pool.submit(download, item) for item in work[start : start + 8]]
            for future in futures:
                candidate, metadata, path = future.result()
                if path is not None:
                    try:
                        _prepare_one(candidate, metadata, path, directory, app, fingerprint)
                    except ValueError as exc:
                        candidate.reasons.append(str(exc))
                    except Exception as exc:
                        raise FacePipelineError(f"Face processing failed for {candidate.asset_id}") from exc
                    finally:
                        path.unlink(missing_ok=True)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(work))
    source_dir.rmdir()
    return records, fingerprint


def select_face_candidates(candidates: list[FaceCandidate], limit: int, mode: str = "smart") -> list[FaceCandidate]:
    """Quality gates apply even to tiny pools and explicit time selection."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("Face limit must be a positive integer")
    if mode not in {"smart", "time"}:
        raise ValueError("Unknown face selection mode")
    valid = []
    for candidate in candidates:
        if candidate.reasons:
            continue
        try:
            candidate.embedding = _normalize_embedding(candidate.embedding)
        except (ValueError, TypeError):
            candidate.reasons.append("invalid_embedding")
            continue
        valid.append(candidate)
    if not valid:
        return []
    valid.sort(key=_quality_order)
    vectors = np.stack([c.embedding for c in valid])
    distances = np.clip(1 - vectors @ vectors.T, 0, 2)
    np.fill_diagonal(distances, 0)
    if mode == "time":
        ordered = sorted(valid, key=lambda c: (c.created_at, c.asset_id, c.face_id or ""))
        selected = [ordered[i] for i in np.linspace(0, len(ordered) - 1, min(limit, len(ordered)), dtype=int)]
    else:
        keep = []
        for i, candidate in enumerate(valid):
            if keep and np.any(distances[i, keep] < Config.FACE_DUPLICATE_DISTANCE):
                candidate.reasons.append("near_duplicate")
            else:
                keep.append(i)
        valid = [valid[i] for i in keep]
        distances = distances[np.ix_(keep, keep)]
        if len(valid) >= 10:
            neighbors = distances.copy()
            np.fill_diagonal(neighbors, np.inf)
            isolation = np.sort(neighbors, axis=1)[:, :5].mean(axis=1)
            median = float(np.median(isolation))
            mad = float(np.median(np.abs(isolation - median)))
            threshold = median + Config.FACE_OUTLIER_MAD * 1.4826 * mad
            keep = []
            for i, candidate in enumerate(valid):
                candidate.measurements["neighbor_distance"] = float(isolation[i])
                if mad > 1e-8 and isolation[i] > threshold:
                    candidate.reasons.append("isolated_outlier")
                else:
                    keep.append(i)
            valid = [valid[i] for i in keep]
            distances = distances[np.ix_(keep, keep)]
        if not valid:
            return []
        from .diversity import _kmedoids

        indices, _ = _kmedoids(distances, min(limit, len(valid)))
        selected = [valid[i] for i in indices]
    chosen = {id(c) for c in selected}
    for candidate in valid:
        if id(candidate) in chosen:
            candidate.selected = True
            candidate.selection_reason = "quality_filtered_time_spread" if mode == "time" else "representative_medoid"
        else:
            candidate.reasons.append("not_selected_budget")
    return sorted(selected, key=_quality_order)
