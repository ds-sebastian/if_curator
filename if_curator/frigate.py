"""Frigate 0.17 face recognition, reproduced exactly (see THIRD_PARTY_NOTICES.md).

Frigate stores an uploaded face as the crop of YuNet's detection box. To recognize a face
it aligns the crop with LBF landmarks, embeds it with ArcFace, and compares it by cosine
similarity to each person's 15% trimmed mean of raw library embeddings.
"""

import ctypes.util
import hashlib
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

MODEL_URL = "https://github.com/NickM-27/facenet-onnx/releases/download/v1.0/"
MODEL_HASHES = {
    "facedet.onnx": "321aa5a6afabf7ecc46a3d06bfab2b579dc96eb5c3be7edd365fa04502ad9294",
    "landmarkdet.yaml": "70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b",
    "arcface.onnx": "ec639a0429b4819130d1405a2d3b38beaa4cc4a6c5bd9cf48b94fdf65461de83",
}
DETECTION_THRESHOLD = 0.5  # Frigate's threshold for faces added to the library
MAX_DETECTION_HEIGHT = 1080


def confidence(cosine: float) -> float:
    """Frigate's ArcFace similarity_to_confidence."""
    return 1 / (1 + math.exp(-20 * (cosine - 0.3)))


def cosine_for(score: float) -> float:
    """The cosine at which Frigate reports `score`: the inverse of confidence()."""
    return 0.3 + math.log(score / (1 - score)) / 20


def trimmed_mean(embeddings) -> np.ndarray:
    """scipy.stats.trim_mean(embeddings, 0.15): the center Frigate builds for each person."""
    values = np.sort(np.asarray(embeddings, dtype=np.float64), axis=0)
    cut = int(0.15 * len(values))
    return values[cut : len(values) - cut].mean(axis=0)


def iou(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def align(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """FaceRecognizer.align_face: eyes level, 30% of the width apart, at 35% height."""
    points = np.asarray(landmarks).reshape(-1, 2)
    left = points[42:48].mean(axis=0).astype(int)
    right = points[36:42].mean(axis=0).astype(int)
    dx, dy = right - left
    height, width = bgr.shape[:2]
    center = (int((left[0] + right[0]) // 2), int((left[1] + right[1]) // 2))
    scale = 0.3 * width / max(math.hypot(dx, dy), 1e-6)
    matrix = cv2.getRotationMatrix2D(center, float(np.degrees(np.arctan2(dy, dx)) - 180), scale)
    matrix[0, 2] += width * 0.5 - center[0]
    matrix[1, 2] += height * 0.35 - center[1]
    return cv2.warpAffine(bgr, matrix, (width, height), flags=cv2.INTER_CUBIC)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """ArcfaceEmbedding._preprocess_inputs. Frigate passes BGR pixels through PIL unchanged."""
    image = Image.fromarray(bgr)
    width, height = image.size
    if (width, height) != (112, 112):
        if width > height:
            image = image.resize((112, max(4, int(height / width * 112 // 4 * 4))))
        else:
            image = image.resize((max(4, int(width / height * 112 // 4 * 4)), 112))
    pixels = np.asarray(image, dtype=np.float32)
    frame = np.zeros((112, 112, 3), dtype=np.float32)
    y, x = (112 - pixels.shape[0]) // 2, (112 - pixels.shape[1]) // 2
    frame[y : y + pixels.shape[0], x : x + pixels.shape[1]] = pixels
    return np.transpose(frame / 127.5 - 1, (2, 0, 1))[None]


def fetch_model(directory: Path, name: str) -> Path:
    path = directory / name
    if path.exists():
        return path
    directory.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    with requests.get(MODEL_URL + name, stream=True, timeout=(15, 120)) as response, partial.open("wb") as file:
        response.raise_for_status()
        for chunk in response.iter_content(1 << 20):
            digest.update(chunk)
            file.write(chunk)
    if digest.hexdigest() != MODEL_HASHES[name]:
        partial.unlink()
        raise RuntimeError(f"Downloaded {name} does not match its expected checksum")
    partial.replace(path)
    return path


@contextmanager
def _quiet_stdout():
    """LBF prints from C++ while loading; keep the terminal clean."""
    sys.stdout.flush()
    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


class FrigateFaces:
    """Frigate's detector, landmarker and embedder. Not thread-safe."""

    def __init__(self, model_dir: str | Path, force_cpu: bool = False):
        import onnxruntime as ort

        paths = {name: fetch_model(Path(model_dir), name) for name in MODEL_HASHES}
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # OpenCV 5 warns about DNN targets.
        self.detector = cv2.FaceDetectorYN.create(str(paths["facedet.onnx"]), "", (320, 320), DETECTION_THRESHOLD, 0.3)
        if not hasattr(cv2, "face"):
            raise RuntimeError("Frigate's landmarks need opencv-contrib-python-headless; run `uv sync --locked`")
        self.landmarks = cv2.face.createFacemarkLBF()
        with _quiet_stdout():
            self.landmarks.loadModel(str(paths["landmarkdet.yaml"]))
        providers = ["CPUExecutionProvider"]
        nvidia_driver = ctypes.util.find_library("nvcuda" if sys.platform == "win32" else "cuda")
        if not force_cpu and nvidia_driver and "CUDAExecutionProvider" in ort.get_available_providers():
            ort.preload_dlls()  # CUDA libraries from the gpu extra, if installed
            providers.insert(0, "CUDAExecutionProvider")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(paths["arcface.onnx"]), options, providers=providers)
        self.input = self.session.get_inputs()[0].name
        self.device = "GPU" if "CUDAExecutionProvider" in self.session.get_providers() else "CPU"

    def detect(self, bgr: np.ndarray, target) -> tuple[int, int, int, int] | None:
        """The YuNet box Frigate would crop for the target face, or None if it finds none."""
        scale = min(1.0, MAX_DETECTION_HEIGHT / bgr.shape[0])
        if scale < 1:
            bgr = cv2.resize(bgr, (int(bgr.shape[1] * scale), MAX_DETECTION_HEIGHT))
        self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self.detector.detect(bgr)
        best, best_overlap = None, 0.5
        for face in [] if faces is None else faces:
            x, y, w, h = face[:4] / scale
            box = (int(max(x, 0)), int(max(y, 0)), int(x + w), int(y + h))
            overlap = iou(box, target)
            if overlap >= best_overlap:
                best, best_overlap = box, overlap
        return best

    def embed(self, faces: list[np.ndarray]) -> list[np.ndarray | None]:
        """Raw ArcFace embeddings of tight face crops; None where landmarks can't be fit."""
        tensors, index = [], []
        for i, face in enumerate(faces):
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            found, landmarks = self.landmarks.fit(gray, np.array([(0, 0, gray.shape[1], gray.shape[0])]))
            if found and len(landmarks) and np.asarray(landmarks[0]).size == 136:
                tensors.append(preprocess(align(face, landmarks[0])))
                index.append(i)
        vectors: list[np.ndarray | None] = [None] * len(faces)
        if tensors:
            for i, vector in zip(index, self.session.run(None, {self.input: np.concatenate(tensors)})[0]):
                vectors[i] = vector.astype(np.float32)
        return vectors
