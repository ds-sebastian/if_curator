"""Immich API client for fetching people, assets, and face data."""

import logging
from datetime import datetime, timedelta, timezone
from io import BytesIO

import requests
from PIL import Image

from .config import Config, get_headers

logger = logging.getLogger(__name__)


def get_people() -> list[dict]:
    """Fetch all people from Immich."""
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/people",
            headers=get_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("people", [])
    except requests.RequestException as e:
        logger.error(f"Failed to fetch people: {e}")
        return []


def fetch_all_assets(person: dict) -> list[dict]:
    """Fetch all assets for a person with pagination."""
    name = person.get("name", "Unknown")
    person_id = person["id"]
    url = f"{Config.IMMICH_URL}/api/search/metadata"
    page_size = 1000

    logger.info(f"Fetching assets for {name}...")

    assets = []
    for page in range(1, 1000):  # Safety limit
        try:
            resp = requests.post(
                url,
                json={"personIds": [person_id], "size": page_size, "page": page},
                headers=get_headers(),
                timeout=30,
            )

            if not resp.ok:
                logger.error(f"Error fetching assets for {name} (page {page}): {resp.status_code}")
                break

            page_assets = resp.json().get("assets", [])
            if isinstance(page_assets, dict):
                page_assets = page_assets.get("items", [])

            if not page_assets:
                break

            assets.extend(page_assets)
            logger.debug(f"Fetched page {page}, total: {len(assets)}")

            if len(page_assets) < page_size:
                break

        except requests.RequestException as e:
            logger.error(f"Exception fetching assets for {name}: {e}")
            break

    return assets


def fetch_full_image(asset_id: str, timeout: int = 60) -> Image.Image | None:
    """Fetch full-resolution image from Immich, falling back to preview thumbnail.

    The /original endpoint may return HEIC, RAW, or video files that PIL
    cannot open directly. In that case, we fall back to the JPEG thumbnail.
    """
    # Try original first
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/assets/{asset_id}/original",
            headers=get_headers(),
            timeout=timeout,
        )
        if resp.ok:
            try:
                return Image.open(BytesIO(resp.content))
            except Exception:
                logger.debug(f"PIL can't open original for {asset_id}, falling back to preview")
    except requests.RequestException:
        logger.debug(f"Original request failed for {asset_id}, falling back to preview")

    # Fall back to preview thumbnail (always JPEG)
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview&format=JPEG",
            headers=get_headers(),
            timeout=30,
        )
        if resp.ok:
            return Image.open(BytesIO(resp.content))
    except Exception as e:
        logger.error(f"Failed to fetch image {asset_id}: {e}")

    return None


def filter_recent_assets(assets: list[dict], years: int | None = None) -> list[dict]:
    """Filter assets to keep only those from the last N years."""
    years = years or Config.YEARS_FILTER
    cutoff = datetime.now(timezone.utc) - timedelta(days=365 * years)

    logger.debug(f"Filtering assets older than {years} years ({cutoff})")

    recent, skipped = [], 0
    for asset in assets:
        created_at_str = asset.get("fileCreatedAt")
        if not created_at_str:
            continue

        try:
            # Handle ISO8601 with 'Z' suffix
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            if created_at > cutoff:
                recent.append(asset)
            else:
                skipped += 1
        except ValueError:
            continue

    logger.info(f"Retained {len(recent)} assets (filtered {skipped} old assets).")
    return recent


def resolve_face_metadata(asset: dict, person_id: str) -> dict:
    """Return exactly one target face. Missing nested metadata triggers API lookup."""
    if not person_id:
        raise ValueError("person_id is required")
    fields = {"imageWidth", "imageHeight", "boundingBoxX1", "boundingBoxX2", "boundingBoxY1", "boundingBoxY2"}
    matches = [
        face for person in asset.get("people", []) if person.get("id") == person_id for face in person.get("faces", [])
    ]
    if len(matches) > 1:
        raise ValueError("ambiguous_target_face")
    if len(matches) == 1 and fields <= matches[0].keys():
        return matches[0]
    resp = requests.get(f"{Config.IMMICH_URL}/api/faces", params={"id": asset["id"]}, headers=get_headers(), timeout=10)
    resp.raise_for_status()
    matches = [face for face in resp.json() if (face.get("person") or {}).get("id") == person_id]
    if len(matches) != 1:
        raise ValueError("missing_or_ambiguous_target_face")
    if not fields <= matches[0].keys():
        raise ValueError("incomplete_face_metadata")
    return matches[0]


def fetch_image_source(asset_id: str, use_original: bool = True) -> tuple[Image.Image, str]:
    """Fully decode and orient an image before declaring a download successful."""
    from PIL import ImageOps

    endpoints = []
    if use_original:
        endpoints.append((f"/api/assets/{asset_id}/original", "original"))
    endpoints.append((f"/api/assets/{asset_id}/thumbnail?size=preview&format=JPEG", "preview"))
    for endpoint, source in endpoints:
        try:
            resp = requests.get(f"{Config.IMMICH_URL}{endpoint}", headers=get_headers(), timeout=60)
            resp.raise_for_status()
            with Image.open(BytesIO(resp.content)) as image:
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB"), source
        except (requests.RequestException, OSError, ValueError):
            continue
    raise ValueError("image_download_or_decode_failed")
