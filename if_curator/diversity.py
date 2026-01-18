"""
Diversity selection for training data curation.

Uses Farthest Point Sampling (FPS) algorithm with embeddings:
- Faces: InsightFace embeddings
- Objects: SigLIP embeddings
"""

import logging
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from .config import Config, get_headers
from .embeddings import get_embedding, is_embedding_available

logger = logging.getLogger(__name__)


def select_diverse_assets(
    assets: list,
    limit: int | str,
    entity_name: str,
    selection_mode: str = "smart",
    entity_type: str = "face",
    progress_callback=None,
) -> list:
    """
    Select diverse assets using Farthest Point Sampling or time spread.

    Args:
        assets: List of asset dicts from Immich API
        limit: Number to select, or "auto" for dynamic selection
        entity_name: Name of the person/object for logging
        selection_mode: 'smart' (embedding-based) or 'time' (time spread)
        entity_type: 'face' or 'object' - determines embedding model
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of selected assets
    """
    # Fast path: fewer assets than limit
    if limit != "auto" and len(assets) <= limit:
        return assets

    # Sort by creation time
    assets = sorted(assets, key=lambda x: x.get("fileCreatedAt", ""))

    if selection_mode != "smart" or not is_embedding_available(entity_type):
        if selection_mode == "smart":
            model_name = "InsightFace" if entity_type == "face" else "SigLIP"
            logger.warning(f"{model_name} unavailable. Falling back to time spread.")
        return _select_time_spread(assets, limit)

    try:
        return _select_by_embedding(assets, limit, entity_type, progress_callback)
    except Exception as e:
        logger.error(f"Smart Diversity failed: {e}. Falling back to time spread.")
        return _select_time_spread(assets, limit)


def _fetch_thumbnail(asset_id: str, timeout: int = 10) -> Image.Image | None:
    """Fetch thumbnail from Immich API."""
    try:
        url = f"{Config.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview&format=JPEG"
        resp = requests.get(url, headers=get_headers(), timeout=timeout)
        return Image.open(BytesIO(resp.content)) if resp.ok else None
    except Exception:
        return None


def _select_by_embedding(
    assets: list,
    limit: int | str,
    entity_type: str,
    progress_callback=None,
) -> list:
    """Select assets using embedding-based Farthest Point Sampling."""
    # Determine candidate pool (cap at 3000 for performance)
    effective_limit = 30 if limit == "auto" else limit
    pool_size = min(3000, max(effective_limit * 20, len(assets)))

    # Subsample if needed (evenly distributed in time)
    if len(assets) > pool_size:
        indices = np.linspace(0, len(assets) - 1, pool_size, dtype=int)
        candidates = [assets[i] for i in indices]
    else:
        candidates = assets

    # Compute embeddings
    embeddings, valid_candidates = [], []
    for i, asset in enumerate(candidates):
        if progress_callback:
            progress_callback(i, len(candidates))

        img = _fetch_thumbnail(asset["id"])
        if img is None:
            continue

        emb = get_embedding(img, entity_type)
        if emb is not None:
            embeddings.append(emb)
            valid_candidates.append(asset)

    if progress_callback:
        progress_callback(len(candidates), len(candidates))

    if not embeddings:
        logger.warning("No valid embeddings found. Falling back to time spread.")
        return _select_time_spread(assets, limit)

    if limit != "auto" and len(valid_candidates) < limit:
        logger.warning(f"Only {len(valid_candidates)} valid embeddings. Returning all.")
        return valid_candidates

    # Farthest Point Sampling with vectorized distance computation
    return _farthest_point_sampling(
        embeddings, valid_candidates, limit, auto_threshold=0.15
    )


def _farthest_point_sampling(
    embeddings: list,
    candidates: list,
    limit: int | str,
    auto_threshold: float = 0.15,
) -> list:
    """Vectorized Farthest Point Sampling."""
    emb_matrix = np.vstack(embeddings)  # (N, D)
    n = len(emb_matrix)

    # Normalize for cosine distance (cosine_dist = 1 - cosine_sim)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_normed = emb_matrix / np.maximum(norms, 1e-8)

    # Start from median-time sample
    selected = [n // 2]
    min_dists = np.full(n, np.inf)

    target = 500 if limit == "auto" else limit

    while len(selected) < target:
        # Update min distances with last selected point
        last_emb = emb_normed[selected[-1]]
        dists_to_last = 1 - emb_normed @ last_emb  # Cosine distance
        min_dists = np.minimum(min_dists, dists_to_last)
        min_dists[selected[-1]] = -np.inf  # Exclude already selected

        # Find farthest point
        best_idx = np.argmax(min_dists)
        best_dist = min_dists[best_idx]

        if best_dist == -np.inf:
            break  # All points selected

        if limit == "auto" and best_dist < auto_threshold:
            logger.info(
                f"Auto-stop: Next best image {best_dist:.3f} away (threshold {auto_threshold})."
            )
            break

        selected.append(best_idx)

    logger.info(f"Smart selection complete. Picked {len(selected)} diverse images.")
    return [candidates[i] for i in selected]


def _select_time_spread(assets: list, limit: int | str) -> list:
    """Select N assets evenly distributed in time."""
    if limit == "auto":
        limit = 30

    logger.info(f"Selecting {limit} images using time spread.")

    if len(assets) <= limit:
        return assets

    indices = np.linspace(0, len(assets) - 1, limit, dtype=int)
    return [assets[i] for i in np.unique(indices)]
