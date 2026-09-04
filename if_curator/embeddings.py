"""
Unified embedding interface for faces and objects.

- Faces: local InsightFace app, consumed by the target-bound face pipeline
- Objects: SigLIP (Vision Transformer via transformers)
- Caching: Disk-based cache avoids recomputation on reruns
"""

import contextlib
import logging
import os

import numpy as np
from PIL import Image

from .cache import get_cache
from .onnx_runtime import preload_cuda

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

    ctx_id = -1
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        # Get providers, excluding TensorRT to avoid noisy errors
        providers = [p for p in ort.get_available_providers() if p != "TensorrtExecutionProvider"]
        if _is_force_cpu():
            providers = ["CPUExecutionProvider"]
        logger.info(f"Requested ONNX providers: {providers}")
        preload_cuda(ort, providers)

        # Determine device: 0 for GPU, -1 for CPU
        gpu_providers = {
            "CUDAExecutionProvider",
            "ROCmExecutionProvider",
            "MPSExecutionProvider",
            "CoreMLExecutionProvider",
        }
        ctx_id = -1 if _is_force_cpu() else (0 if gpu_providers & set(providers) else -1)

        device_str = "GPU" if ctx_id >= 0 else "CPU"
        logger.info(f"Loading InsightFace Buffalo_L on {device_str} (ctx_id={ctx_id})...")

        # Suppress C-level output during model loading
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            app = FaceAnalysis(name="buffalo_l", root="~/.insightface", providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            _insightface_app = app

        active = app.det_model.session.get_providers()
        logger.info("InsightFace detector active providers: %s", active)
        if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in active:
            logger.warning("InsightFace could not activate CUDA; using %s", active)
        return _insightface_app

    except ImportError as exc:
        logger.error(f"InsightFace dependency unavailable: {exc}")
        return None
    except Exception as e:
        logger.error(f"Failed to load InsightFace: {e}")
        # Retry on CPU if GPU failed
        if ctx_id == 0:
            logger.warning("Retrying InsightFace on CPU...")
            try:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(name="buffalo_l", root="~/.insightface", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                _insightface_app = app
                return app
            except Exception as ex:
                logger.error(f"CPU fallback failed: {ex}")
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


def get_object_embeddings_batch(images: list[Image.Image]) -> list[np.ndarray | None]:
    """Get SigLIP embeddings for a batch of images (GPU-efficient)."""
    model, processor = get_siglip_model()
    if model is None:
        return [None] * len(images)

    try:
        import torch

        inputs = processor(images=images, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.pooler_output.cpu().numpy()
            return [embeddings[i] for i in range(len(embeddings))]
    except Exception as e:
        logger.error(f"Error in batch embedding: {e}")
        # Fall back to individual computation
        return [get_object_embedding(img) for img in images]


# =============================================================================
# Unified Interface with Caching
# =============================================================================


def get_embedding(img_pil: Image.Image, entity_type: str = "object", asset_id: str | None = None) -> np.ndarray | None:
    """Object embedding cache. Face embeddings use the target-bound face pipeline."""
    from .config import Config

    if entity_type != "object":
        raise ValueError("Face embeddings require a prepared, target-bound candidate")
    cache = get_cache(Config.CACHE_DIR) if Config.ENABLE_CACHE and asset_id is not None else None
    if cache:
        cached = cache.get(asset_id, "siglip")
        if cached is not None:
            return cached
    embedding = get_object_embedding(img_pil)
    if embedding is not None and cache:
        cache.put(asset_id, embedding, "siglip")
    return embedding


def is_embedding_available(entity_type: str = "face") -> bool:
    """Check if embedding model is available for the given entity type."""
    if entity_type == "face":
        return get_insightface_app() is not None
    model, _ = get_siglip_model()
    return model is not None
