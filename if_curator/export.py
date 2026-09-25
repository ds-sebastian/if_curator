"""Write each person's selected images to a new run folder."""

import json
import re
import shutil
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2

from . import __version__
from .faces import CONTEXT, fetch_region
from .immich import DOWNLOAD_ERRORS, Immich, bounded_map
from .selection import Candidate, Job


def folder_names(jobs: list[Job]) -> list[str]:
    """Frigate uses the folder name as the label, so keep the person's name."""
    names = [re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", job.name).strip(" .") or "person" for job in jobs]
    return [f"{name} ({job.person['id'][:8]})" if names.count(name) > 1 else name for name, job in zip(names, jobs)]


def _regions(immich: Immich, candidate: Candidate, sources: list[str], margin: float) -> list[tuple]:
    regions = []
    for source in sources:
        try:
            regions.append((source, *fetch_region(immich, candidate, margin, original=source == "original")))
        except (ValueError, *DOWNLOAD_ERRORS):
            pass
    return regions


def _encode(job: Job, faces, regions: list[tuple]) -> tuple[bytes, str, str] | None:
    """The best available image: a Frigate face crop (WebP, as Frigate stores it) or an object crop."""
    for source, region, target in regions:
        if job.object_class:
            return cv2.imencode(".jpg", region, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), ".jpg", source
        box = faces.detect(region, target)
        if box is not None:
            face = region[box[1] : box[3], box[0] : box[2]]
            return cv2.imencode(".webp", face, [cv2.IMWRITE_WEBP_QUALITY, 100])[1].tobytes(), ".webp", source
    return None


def _write(job: Job, folder: Path, immich: Immich, faces, sources: list[str], progress) -> list[dict]:
    fetch = partial(_regions, immich, sources=sources, margin=0.0 if job.object_class else CONTEXT)
    images = []
    for candidate, regions in zip(job.selected, bounded_map(fetch, job.selected, workers=4)):
        encoded = _encode(job, faces, regions)
        if encoded:
            data, extension, source = encoded
            path = folder / f"{len(images):03d}{extension}"
            path.write_bytes(data)
            images.append(
                {
                    "file": f"{folder.name}/{path.name}",
                    "asset_id": candidate.asset_id,
                    "face_id": candidate.face_id,
                    "taken": candidate.taken,
                    "source": source,
                    **candidate.measures,
                }
            )
        progress()
    return images


def export(jobs: list[Job], immich: Immich, faces, settings, progress=lambda: None) -> Path:
    """Write into a hidden folder and rename it when complete, so a run is never half there."""
    root = Path(settings.OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    final, n = root / stamp, 2
    while final.exists():
        final, n = root / f"{stamp}_{n}", n + 1
    staging = root / f".{final.name}.partial"
    staging.mkdir()
    sources = ["original", "preview"] if settings.USE_FULL_RESOLUTION else ["preview"]
    people = []
    try:
        for job, name in zip(jobs, folder_names(jobs)):
            (staging / name).mkdir()
            images = _write(job, staging / name, immich, faces, sources, progress)
            reasons = Counter(c.reason for c in job.candidates if c.reason)
            people.append(
                {
                    "name": job.name,
                    "id": job.person["id"],
                    "folder": name,
                    "object_class": job.object_class,
                    "photos": len({c.asset_id for c in job.candidates}),
                    "usable": len(job.eligible),
                    "exported": len(images),
                    "recognized": job.recognized,
                    "rejections": dict(reasons.most_common()),
                    "images": images,
                }
            )
        manifest = {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "version": __version__,
            "settings": settings.public(),
            "people": people,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final
