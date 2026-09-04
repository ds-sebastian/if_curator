"""Isolated run directories and atomic publication of evaluated face artifacts."""

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import Config
from .faces import DETECTOR_INPUT_SIZE, PREPROCESSING_VERSION


def person_directory(name: str, person_id: str, mode: str = "face") -> str:
    slug = re.sub(r"[^\w-]+", "_", name, flags=re.UNICODE).strip("_.")[:64] or "person"
    identity = hashlib.sha256(f"{person_id}:{mode}".encode()).hexdigest()[:12]
    return f"{slug}-{identity}"


class RunWorkspace:
    def __init__(self, output_dir: str | Path):
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
        self.path = root / f".{self.run_id}.incomplete"
        self.destination = root / self.run_id
        self.path.mkdir()
        self.manifest = {
            "schema_version": 2,
            "run_id": self.run_id,
            "status": "preparing",
            "configuration": Config.snapshot(),
            "preprocessing_version": PREPROCESSING_VERSION,
            "face_detector_input_size": DETECTOR_INPUT_SIZE,
            "embedding_backend": "Frigate 0.17.2 large ArcFace; InsightFace target detection",
            "jobs": [],
        }
        self.write_manifest()

    def preparation_directory(self, person_id: str) -> Path:
        path = self.path / ".prepared" / hashlib.sha256(person_id.encode()).hexdigest()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_manifest(self) -> None:
        pending = self.path / "manifest.json.tmp"
        pending.write_text(json.dumps(self.manifest, indent=2, allow_nan=False) + "\n")
        pending.replace(self.path / "manifest.json")

    def record_jobs(self, jobs: list[dict]) -> None:
        self.manifest["jobs"] = [
            {
                "person_id": job["person"]["id"],
                "person_name": job["person"]["name"],
                "mode": job["config"]["mode"],
                "model_fingerprint": job.get("model_fingerprint"),
                "selection_mode": job.get("selection_mode"),
                "selection_report": job.get("selection_report"),
                "requested_limit": job.get("requested_limit", job["limit"]),
                "years_filter": job.get("years_filter"),
                "selected_count": job["limit"],
                "candidates": [candidate.record() for candidate in job.get("candidates", [])],
                "object_outputs": job.get("object_outputs", []),
            }
            for job in jobs
        ]
        self.write_manifest()

    def export_faces(self, job: dict) -> None:
        dirname = person_directory(job["person"]["name"], job["person"]["id"])
        destination = self.path / dirname
        destination.mkdir(exist_ok=False)
        for count, candidate in enumerate(job["selected_faces"]):
            if candidate.reasons or not candidate.selected or candidate.person_id != job["person"]["id"]:
                raise ValueError("Attempt to export an unapproved face")
            data = candidate.prepared_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != candidate.image_hash:
                raise ValueError("Prepared image changed after evaluation")
            relative = f"{dirname}/{count:03d}.jpg"
            target = self.path / relative
            with target.open("xb") as output:
                output.write(data)
            candidate.output_path = relative

    def publish(self, jobs: list[dict]) -> Path:
        self.record_jobs(jobs)
        prepared = self.path / ".prepared"
        if prepared.exists():
            shutil.rmtree(prepared)
        self.manifest["status"] = "complete"
        self.write_manifest()
        self.path.rename(self.destination)
        self.path = self.destination
        return self.destination

    def fail(self, status: str = "failed") -> None:
        self.manifest["status"] = status
        self.write_manifest()
