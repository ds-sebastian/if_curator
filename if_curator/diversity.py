"""Object diversity selection and shared K-Medoids; faces use the target-bound pipeline."""

import logging

import numpy as np
from PIL import Image

from .config import Config
from .embeddings import get_embedding, is_embedding_available
from .immich_api import fetch_preview_image

logger = logging.getLogger(__name__)


def select_diverse_assets(
    assets: list,
    limit: int | str,
    entity_name: str,
    selection_mode: str = "smart",
    entity_type: str = "face",
    progress_callback=None,
    *,
    person_id: str | None = None,
    staging_dir=None,
) -> list:
    """
    Select object assets, or prepared FaceCandidates when entity_type is face.

    Args:
        assets: List of asset dicts from Immich API
        limit: Number to select, or "auto" for dynamic selection
        entity_name: Name of the person/object for logging
        selection_mode: 'smart' (embedding-based) or 'time' (time spread)
        entity_type: 'face' or 'object' - determines embedding model
        progress_callback: Optional callback(current, total) for progress

    Returns:
        Selected asset dicts for objects; FaceCandidates for faces.
    """
    if entity_type == "face":
        from .faces import prepare_face_candidates, select_face_candidates

        if not person_id or staging_dir is None:
            raise ValueError("Face selection requires person_id and a staging_dir")
        records, _ = prepare_face_candidates(assets, person_id, staging_dir, progress_callback)
        return select_face_candidates(records, Config.FACE_MAX_IMAGES if limit == "auto" else limit, selection_mode)

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


# =============================================================================
# Thumbnail & Metadata Helpers
# =============================================================================


def _fetch_thumbnail(asset_id: str, timeout: int = 10) -> Image.Image | None:
    """Fetch the same oriented RGB previews used by other image consumers."""
    return fetch_preview_image(asset_id, timeout=timeout)


# =============================================================================
# Embedding Collection
# =============================================================================


def _select_by_embedding(
    assets: list,
    limit: int | str,
    entity_type: str,
    progress_callback=None,
) -> list:
    """Select object assets using SigLIP, K-Medoids and FPS."""
    # Determine candidate pool (cap at 3000 for performance)
    effective_limit = 30 if limit == "auto" else limit
    pool_size = min(3000, max(effective_limit * 20, len(assets)))

    # Subsample if needed (evenly distributed in time)
    if len(assets) > pool_size:
        indices = np.linspace(0, len(assets) - 1, pool_size, dtype=int)
        candidates = [assets[i] for i in indices]
    else:
        candidates = assets

    # --- Phase 1: Concurrent thumbnail download ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    thumbnail_map: dict[str, Image.Image] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_thumbnail, a["id"]): a for a in candidates}
        for i, future in enumerate(as_completed(futures)):
            if progress_callback:
                progress_callback(i, len(candidates))
            asset = futures[future]
            try:
                img = future.result()
                if img is not None:
                    thumbnail_map[asset["id"]] = img
            except Exception:
                continue

    # Embed whole object previews (existing object behavior).
    embeddings, valid_candidates = [], []

    for asset in candidates:
        img = thumbnail_map.get(asset["id"])
        if img is None:
            continue

        embed_img = img

        emb = get_embedding(embed_img, entity_type, asset_id=asset["id"])
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

    # --- Phase 5: Cluster-aware selection ---
    return _cluster_aware_selection(
        embeddings,
        valid_candidates,
        limit,
        entity_type=entity_type,
    )


# =============================================================================
# K-Medoids (Lightweight Implementation)
# =============================================================================


def _kmedoids(dist_matrix: np.ndarray, k: int, max_iter: int = 50) -> tuple[list[int], np.ndarray]:
    """Lightweight K-Medoids clustering using cosine distance matrix.

    Args:
        dist_matrix: (N, N) pairwise distance matrix
        k: Number of clusters
        max_iter: Maximum iterations for swap step

    Returns:
        (medoid_indices, cluster_labels) tuple
    """
    n = dist_matrix.shape[0]
    if n == 0 or k <= 0:
        return [], np.empty(0, dtype=int)
    k = min(k, n)
    rng = np.random.default_rng(42)

    # Initialize medoids: first = most central point, rest = farthest from chosen
    total_dist = dist_matrix.sum(axis=1)
    medoids = [int(np.argmin(total_dist))]

    for _ in range(k - 1):
        dists_to_chosen = dist_matrix[:, medoids].min(axis=1)
        dists_to_chosen[medoids] = -np.inf
        medoids.append(int(np.argmax(dists_to_chosen)))

    # Iterative swap step
    medoids = list(medoids)
    labels = np.argmin(dist_matrix[:, medoids], axis=1)
    cost = sum(dist_matrix[i, medoids[labels[i]]] for i in range(n))

    for _ in range(max_iter):
        improved = False
        # Try swapping each medoid with a random non-medoid
        non_medoids = [i for i in range(n) if i not in medoids]
        if not non_medoids:
            break

        for m_idx in range(k):
            candidates = rng.choice(non_medoids, size=min(10, len(non_medoids)), replace=False)
            for cand in candidates:
                new_medoids = medoids.copy()
                new_medoids[m_idx] = cand
                new_labels = np.argmin(dist_matrix[:, new_medoids], axis=1)
                new_cost = sum(dist_matrix[i, new_medoids[new_labels[i]]] for i in range(n))
                if new_cost < cost:
                    medoids = new_medoids
                    labels = new_labels
                    cost = new_cost
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return medoids, labels


# =============================================================================
# Cluster-Aware Selection (K-Medoids + FPS Hybrid)
# =============================================================================


def _compute_adaptive_threshold(emb_normed: np.ndarray, entity_type: str) -> float:
    """Compute adaptive FPS stop threshold based on actual embedding distribution.

    Instead of a hardcoded threshold, samples pairwise distances and sets
    the threshold as a fraction of the median pairwise distance.
    """
    n = len(emb_normed)
    sample_size = min(200, n)
    rng = np.random.default_rng(42)
    indices = rng.choice(n, sample_size, replace=False) if n > sample_size else np.arange(n)
    sample = emb_normed[indices]

    # Compute pairwise cosine distances for the sample
    pairwise = 1 - sample @ sample.T
    upper_tri = pairwise[np.triu_indices(len(sample), k=1)]
    median_dist = float(np.median(upper_tri))

    # Faces: 20% of median (tighter — want fewer, more distinct images)
    # Objects: 10% of median (wider — want more diversity)
    fraction = 0.20 if entity_type == "face" else 0.10
    threshold = max(0.05, median_dist * fraction)

    logger.info(
        f"Adaptive threshold: {threshold:.4f} (median_dist={median_dist:.4f}, fraction={fraction}, type={entity_type})"
    )
    return threshold


def _cluster_aware_selection(
    embeddings: list,
    candidates: list,
    limit: int | str,
    entity_type: str = "object",
) -> list:
    """Object selection: cluster medoids followed by farthest-point filling."""
    emb_matrix = np.vstack(embeddings)  # (N, D)
    n = len(emb_matrix)

    # Normalize for cosine distance
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_normed = emb_matrix / np.maximum(norms, 1e-8)

    # Compute adaptive threshold for auto mode
    auto_threshold = _compute_adaptive_threshold(emb_normed, entity_type) if limit == "auto" else 0.0
    target = Config.MAX_AUTO_IMAGES if limit == "auto" else limit

    # --- Stage 1: K-Medoids clustering ---
    k = min(max(5, target // 4), n // 3, n)  # e.g., 5-20 clusters
    logger.info(f"Clustering {n} embeddings into {k} groups (K-Medoids)...")

    # Compute full cosine distance matrix
    dist_matrix = 1 - emb_normed @ emb_normed.T

    medoid_indices, cluster_labels = _kmedoids(dist_matrix, k)
    selected = list(medoid_indices)

    logger.info(f"Selected {len(selected)} cluster medoids as initial picks.")

    # --- Stage 2: FPS ---
    min_dists = np.full(n, np.inf)

    # Initialize min distances from all medoids
    for idx in selected:
        dists = dist_matrix[idx]
        min_dists = np.minimum(min_dists, dists)
    for idx in selected:
        min_dists[idx] = -np.inf

    while len(selected) < target:
        best_idx = int(np.argmax(min_dists))
        best_dist = min_dists[best_idx]  # Use unweighted for threshold comparison

        if best_dist == -np.inf:
            break  # All points selected

        if limit == "auto" and best_dist < auto_threshold:
            logger.info(f"Auto-stop: Next best image {best_dist:.3f} away (adaptive threshold {auto_threshold:.4f}).")
            break

        selected.append(best_idx)

        # Update min distances
        dists_to_new = dist_matrix[best_idx]
        min_dists = np.minimum(min_dists, dists_to_new)
        min_dists[best_idx] = -np.inf

    logger.info(f"Selection complete: {len(selected)} diverse images.")

    return [candidates[i] for i in selected]


# =============================================================================
# Time Spread Fallback
# =============================================================================


def _select_time_spread(assets: list, limit: int | str) -> list:
    """Select N assets evenly distributed in time."""
    if limit == "auto":
        limit = 30

    logger.info(f"Selecting {limit} images using time spread.")

    if len(assets) <= limit:
        return assets

    indices = np.linspace(0, len(assets) - 1, limit, dtype=int)
    return [assets[i] for i in np.unique(indices)]
