"""
Unified embedding interface for faces and objects.

- Faces: InsightFace (ArcFace/Buffalo_L)
- Objects: SigLIP (Vision Transformer via transformers)
"""

import contextlib
import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_insightface_app = None
_siglip_model = None
_siglip_processor = None


def _is_force_cpu() -> bool:
    """Check if CPU mode is forced via environment variable."""
    return os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes")


# =============================================================================
# InsightFace (Faces)
# =============================================================================


def get_insightface_app():
    """Singleton for InsightFace app with automatic GPU/CPU fallback."""
    global _insightface_app
    if _insightface_app is not None:
        return _insightface_app

    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        # Get providers, excluding TensorRT to avoid noisy errors
        providers = [p for p in ort.get_available_providers() if p != "TensorrtExecutionProvider"]
        logger.info(f"Available ONNX providers: {providers}")

        # Determine device: 0 for GPU, -1 for CPU
        gpu_providers = {
            "CUDAExecutionProvider", "ROCmExecutionProvider",
            "MPSExecutionProvider", "CoreMLExecutionProvider",
        }
        ctx_id = -1 if _is_force_cpu() else (0 if gpu_providers & set(providers) else -1)

        device_str = "GPU" if ctx_id >= 0 else "CPU"
        logger.info(f"Loading InsightFace Buffalo_L on {device_str} (ctx_id={ctx_id})...")

        # Suppress C-level output during model loading
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            _insightface_app = FaceAnalysis(name="buffalo_l", root="~/.insightface", providers=providers)
            _insightface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        return _insightface_app

    except ImportError:
        logger.error("InsightFace not installed!")
        return None
    except Exception as e:
        logger.error(f"Failed to load InsightFace: {e}")
        # Retry on CPU if GPU failed
        if ctx_id == 0:
            logger.warning("Retrying InsightFace on CPU...")
            try:
                from insightface.app import FaceAnalysis
                _insightface_app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
                _insightface_app.prepare(ctx_id=-1, det_size=(640, 640))
                return _insightface_app
            except Exception as ex:
                logger.error(f"CPU fallback failed: {ex}")
        return None


def get_face_embedding(img_pil: Image.Image) -> np.ndarray | None:
    """Get embedding of the largest face in a PIL image."""
    app = get_insightface_app()
    if not app:
        return None

    try:
        # InsightFace expects BGR cv2 image
        img_bgr = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        faces = app.get(img_bgr)

        if not faces:
            return None

        # Return embedding of largest face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.embedding
    except Exception as e:
        logger.error(f"Error getting face embedding: {e}")
        return None


# =============================================================================
# SigLIP (Objects)
# =============================================================================


def get_siglip_model():
    """Singleton for SigLIP model and processor with GPU auto-detection."""
    global _siglip_model, _siglip_processor
    if _siglip_model is not None:
        return _siglip_model, _siglip_processor

    try:
        import warnings

        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        model_name = "google/siglip-base-patch16-224"
        logger.info(f"Loading SigLIP model ({model_name})...")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*use_fast.*")
            _siglip_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            _siglip_model = SiglipVisionModel.from_pretrained(model_name)

        _siglip_model.eval()

        # Move to GPU if available
        if not _is_force_cpu():
            if torch.cuda.is_available():
                _siglip_model = _siglip_model.cuda()
                logger.info("SigLIP running on CUDA GPU")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                _siglip_model = _siglip_model.to("mps")
                logger.info("SigLIP running on Apple MPS")
            else:
                logger.info("SigLIP running on CPU")
        else:
            logger.info("FORCE_CPU set. SigLIP running on CPU")

        return _siglip_model, _siglip_processor

    except ImportError as e:
        logger.error(f"transformers/torch not installed: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load SigLIP: {e}")
        return None, None


def get_object_embedding(img_pil: Image.Image) -> np.ndarray | None:
    """Get 768-dim SigLIP embedding for an image."""
    model, processor = get_siglip_model()
    if model is None:
        return None

    try:
        import torch

        inputs = processor(images=img_pil, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.pooler_output.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Error getting object embedding: {e}")
        return None


# =============================================================================
# Unified Interface
# =============================================================================


def get_embedding(img_pil: Image.Image, entity_type: str = "face") -> np.ndarray | None:
    """Get embedding for an image based on entity type ('face' or 'object')."""
    return get_face_embedding(img_pil) if entity_type == "face" else get_object_embedding(img_pil)


def is_embedding_available(entity_type: str = "face") -> bool:
    """Check if embedding model is available for the given entity type."""
    if entity_type == "face":
        return get_insightface_app() is not None
    model, _ = get_siglip_model()
    return model is not None
