"""Frigate 0.17.2 large recognizer compatibility (see THIRD_PARTY_NOTICES.md).

Keep raw embeddings: coordinate trimming and per-vector normalization do not commute.
The upstream ndarray path deliberately preserves BGR channel order, even inside PIL.
"""

import hashlib
import logging
import math
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import requests
from PIL import Image
from PIL import __version__ as pillow_version

from .config import Config
from .onnx_runtime import preload_cuda

logger = logging.getLogger(__name__)

PROFILE = "frigate-0.17.2-large-v1"
MODEL_URL = "https://github.com/NickM-27/facenet-onnx/releases/download/v1.0/"
MODEL_HASHES = {
    "arcface.onnx": "ec639a0429b4819130d1405a2d3b38beaa4cc4a6c5bd9cf48b94fdf65461de83",
    "landmarkdet.yaml": "70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b",
}


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (512,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-8:
        raise ValueError("invalid_embedding")
    return vector / np.linalg.norm(vector)


def class_mean(embeddings):
    """Exactly the stable release's coordinate-wise stats.trim_mean(embs, .15)."""
    values = np.asarray(embeddings)
    if values.ndim != 2 or not len(values):
        raise ValueError("empty_class")
    for vector in values:
        unit(vector)
    cut = int(len(values) * 0.15)
    center = np.partition(values, (cut, len(values) - cut - 1), axis=0)[cut : len(values) - cut].mean(axis=0)
    unit(center)
    return center


def confidence(cosine, blur_reduction=0.0):
    return max(0.0, round(1 / (1 + math.exp(-20 * (float(cosine) - 0.3))) - blur_reduction, 2))


def blur_reduction(bgr):
    if not Config.FRIGATE_BLUR_CONFIDENCE_FILTER:
        return 0.0
    variance = cv2.Laplacian(bgr, cv2.CV_64F).var()
    for threshold, reduction in ((120, 0.06), (160, 0.04), (200, 0.02), (250, 0.01)):
        if variance < threshold:
            return reduction
    return 0.0


def align(bgr, landmarks):
    landmarks = np.asarray(landmarks)
    if landmarks.shape not in {(1, 68, 2), (68, 1, 2), (68, 2)}:
        raise ValueError("invalid_frigate_landmarks")
    landmarks = landmarks.reshape(68, 2)
    left = landmarks[42:48].mean(axis=0).astype(int)
    right = landmarks[36:42].mean(axis=0).astype(int)
    dx, dy = right - left
    distance = np.hypot(dx, dy)
    if not np.isfinite(landmarks).all() or distance <= 0:
        raise ValueError("invalid_frigate_landmarks")
    height, width = bgr.shape[:2]
    midpoint = tuple(int(v) for v in (left + right) // 2)
    matrix = cv2.getRotationMatrix2D(midpoint, np.degrees(np.arctan2(dy, dx)) - 180, 0.3 * width / distance)
    matrix[0, 2] += width * 0.5 - midpoint[0]
    matrix[1, 2] += height * 0.35 - midpoint[1]
    return cv2.warpAffine(bgr, matrix, (width, height), flags=cv2.INTER_CUBIC)


def preprocess(bgr):
    # Frigate BaseEmbedding._process_image wraps ndarray without BGR->RGB conversion.
    image = Image.fromarray(bgr)
    width, height = image.size
    if (width, height) != (112, 112):
        size = (
            (112, int(height / width * 112 // 4 * 4)) if width > height else (int(width / height * 112 // 4 * 4), 112)
        )
        if min(size) == 0:
            raise ValueError("invalid_frigate_aspect_ratio")
        image = image.resize(size)
    pixels = np.asarray(image, dtype=np.float32)
    height, width = pixels.shape[:2]
    frame = np.zeros((112, 112, 3), dtype=np.float32)
    y, x = (112 - height) // 2, (112 - width) // 2
    frame[y : y + height, x : x + width] = pixels
    return np.transpose(frame / 127.5 - 1, (2, 0, 1))[None]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ensure_model(directory, name):
    path = directory / name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        pending = directory / f".{name}.{uuid4().hex}.part"
        try:
            with requests.get(MODEL_URL + name, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with pending.open("xb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if file_hash(pending) != MODEL_HASHES[name]:
                raise ValueError(f"Frigate model checksum mismatch: {name}")
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    if file_hash(path) != MODEL_HASHES[name]:
        raise ValueError(f"Frigate model checksum mismatch: {name}")
    return path


class FrigateModel:
    def __init__(self, directory, force_cpu=False):
        import onnxruntime as ort

        directory = Path(directory).expanduser()
        arcface = ensure_model(directory, "arcface.onnx")
        landmarks = ensure_model(directory, "landmarkdet.yaml")
        if not hasattr(cv2, "face"):
            raise RuntimeError("Frigate alignment needs opencv-contrib-python-headless; run uv sync --locked")
        self.landmarks = cv2.face.createFacemarkLBF()
        self.landmarks.loadModel(str(landmarks))
        providers = ["CPUExecutionProvider"]
        if not force_cpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        preload_cuda(ort, providers)
        self.session = ort.InferenceSession(str(arcface), providers=providers)
        active = self.session.get_providers()
        logger.info("Frigate ArcFace active providers: %s", active)
        if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in active:
            logger.warning("Frigate ArcFace could not activate CUDA; using %s", active)
        self.fingerprint = (
            PROFILE
            + ":"
            + hashlib.sha256(
                (
                    file_hash(arcface)
                    + file_hash(landmarks)
                    + cv2.__version__
                    + ort.__version__
                    + pillow_version
                    + str(self.session.get_providers())
                ).encode()
            ).hexdigest()
        )
        self.identity = {
            "profile": PROFILE,
            "sha256": dict(MODEL_HASHES),
            "providers": self.session.get_providers(),
            "opencv": cv2.__version__,
            "onnxruntime": ort.__version__,
            "pillow": pillow_version,
            "fingerprint": self.fingerprint,
        }

    def get(self, bgr, target=None):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        success, landmarks = self.landmarks.fit(gray, np.array([(0, 0, gray.shape[1], gray.shape[0])]))
        if not success or not len(landmarks):
            raise ValueError("frigate_alignment_failed")
        tensor = preprocess(align(bgr, landmarks[0]))
        vector = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0][0]
        unit(vector)
        return vector  # Raw, matching Frigate's enrollment representation.


@lru_cache(maxsize=2)
def _load(directory, force_cpu, version):
    if version != "0.17.2":
        raise ValueError("Unsupported Frigate profile")
    return FrigateModel(directory, force_cpu)


def get_frigate_model():
    return _load(Config.FRIGATE_MODEL_DIR, Config.FORCE_CPU, Config.FRIGATE_VERSION)
